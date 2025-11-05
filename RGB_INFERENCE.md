# RGB Inference Quickstart

This note summarizes how to load a released VideoMAE classifier checkpoint and run a forward pass on raw RGB frames.

## 1. Load the Model
- The second-stage checkpoints are built for `videomae_classifier_small_patch16_224`, which returns both a 384-D embedding and action logits (`models/videomae_classifier.py`).
- Instantiate the model and supply the checkpoint path; the loader maps the stored keys into the classifier automatically.

```python
import torch
from models.videomae_classifier import videomae_classifier_small_patch16_224

ckpt_path = "/abs/path/to/rgb_checkpoint.pt"
model = videomae_classifier_small_patch16_224(
    ckpt_pth=ckpt_path,
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes_action=204,   # match the dataset used for training
    use_mean_pooling=True,
)
model.eval().cuda()
```

## 2. Prepare RGB Clips
- All inference clips must contain 16 RGB frames sampled at the same stride used during training (default target sampling rate from `configs/data_module/mm_distill_data_module.yaml`).
- Input tensors are shaped `(batch, channels, frames, height, width)` = `(1, 3, 16, 224, 224)`.
- Reuse the evaluation transform to ensure identical normalization.

```python
from types import SimpleNamespace
from PIL import Image
from datamodule.utils.augmentation import DataAugmentationForVideoMAERGB

cfg = SimpleNamespace(mask_type="tube", mask_ratio=0.75)
transform = DataAugmentationForVideoMAERGB(
    cfg,
    num_frames=16,
    input_size=224,
    patch_size=[16, 16],
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    multi_scale_crop=False,   # deterministic center crop
)

frame_paths = sorted(list_of_16_paths)  # chronological order
frames = [Image.open(p).convert("RGB") for p in frame_paths]

clip_tensor, _ = transform((frames, None))      # -> (3*T, H, W)
clip_tensor = clip_tensor.view(16, 3, 224, 224) # -> (T, C, H, W)
clip_tensor = clip_tensor.permute(1, 0, 2, 3)   # -> (C, T, H, W)
clip_tensor = clip_tensor.unsqueeze(0).cuda()   # add batch dim
```

## 3. Run Inference
- Masks are optional for evaluation; pass `None` for a standard forward.
- The method returns both the pooled embedding (`features`) and action logits.

```python
with torch.no_grad():
    features, logits = model(clip_tensor, mask=None)

prob = logits.softmax(dim=1)
topk = prob.topk(k=5)
```

## 4. Output Interpretation
- `features` is the 384-dimensional representation used in few-shot logistic regression (`models/lit_MMDistillTrainer.py`).
- `logits` correspond to the action labels the checkpoint was trained on (Ego4D/EPIC/etc.); map indices using the same label list from training.
