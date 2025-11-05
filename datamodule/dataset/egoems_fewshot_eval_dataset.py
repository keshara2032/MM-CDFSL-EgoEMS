import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class EgoEMSFewshotEvalDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform, mask_gen, num_frames, mode="RGB"):
        super(EgoEMSFewshotEvalDataset, self).__init__()
        self.cfg = cfg
        self.transform = transform
        self.mask_gen = mask_gen
        self.mode = mode
        self.num_frames = num_frames
        self._construct_loader(cfg)

    def _construct_loader(self, cfg):
        # initialization
        self._dir_to_img_frame = []
        self._uid = []
        self._participant_id = []
        self._video_id = []
        self._start_frame = []
        self._stop_frame = []
        self._action_label = []
        self._action_idx = []

        # read annotation json file
        with open(cfg.fewshot_eval_json_path) as f:
            data = json.load(f)
        assert data["split"] == "val"

        self.num_actions = data["num_actions"]

        for i, clip_dict in enumerate(data["clips"]):
            uid = clip_dict["uid"]
            participant_id = clip_dict["participant_id"]
            video_id = clip_dict["video_id"]
            start_frame = int(clip_dict["start_frame"])
            verb_label = int(clip_dict["verb_label"])
            noun_label = int(clip_dict["noun_label"])
            action_idx = int(clip_dict["action_idx"])
            action_label = (verb_label, noun_label)

            dir_to_img_frame = Path(
                cfg.target_data_dir,
                "egoems/rgb/",
                video_id,
            )
            self._dir_to_img_frame.append(dir_to_img_frame)
            self._uid.append(uid)
            self._start_frame.append(start_frame)
            self._action_label.append(action_label)
            self._action_idx.append(action_idx)

        logger.info(f"Constructing EPIC dataloader (size: {len(self._uid)})")

    def _get_frame(self, dir_to_img_frame, frame_name, mode, frames):
        if mode == "RGB":
            path = dir_to_img_frame / Path(f"frame_{str(frame_name).zfill(10)}.jpg")
            if path.exists():
                frame = Image.open(str(path))
            else:
                frame = frames[-1]
        elif mode == "flow":
            dir_to_flow_frame = str(dir_to_img_frame).replace("rgb", "flow")
            path = Path(dir_to_flow_frame, f"frame_{str(frame_name).zfill(10)}.npy")
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        elif mode == "pose":
            dir_to_flow_frame = str(dir_to_img_frame).replace("rgb", "pose")
            path = Path(dir_to_flow_frame, f"frame_{str(frame_name).zfill(10)}.npy")
            if path.exists():
                frame = np.load(str(path))
            else:
                frame = frames[-1]
        return frame

    def _get_input(self, dir_to_img_frame, clip_start_frame):
        frame_names = [
            max(1, clip_start_frame + self.cfg.target_sampling_rate * i)
            for i in range(self.num_frames)
        ]
        frames = []
        for frame_name in frame_names:
            frame = self._get_frame(dir_to_img_frame, frame_name, self.mode, frames)
            frames.append(frame)

        # [T, H, W, C] -> [T*C, H, W] -> [C, T, H, W]
        if self.mode == "RGB":
            frames, _ = self.transform((frames, None))
            frames = frames.view((self.num_frames, 3) + frames.size()[-2:]).transpose(
                0, 1
            )
        elif self.mode == "flow":
            frames, _ = self.transform(frames)
            frames = frames.view((self.num_frames, 2) + frames.size()[-2:]).transpose(
                0, 1
            )
        elif self.mode == "pose":
            frames, _ = self.transform(frames)
            frames = frames.view((self.num_frames, 21) + frames.size()[-2:]).transpose(
                0, 1
            )

        # mask generation
        mask = self.mask_gen()

        return frames, mask

    def __getitem__(self, index):
        input = {}

        dir_to_img_frame = self._dir_to_img_frame[index]
        clip_start_frame = self._start_frame[index]
        action_label = self._action_label[index]
        action_idx = self._action_idx[index]

        # load frames
        frames, mask = self._get_input(dir_to_img_frame, clip_start_frame)

        input["frames"] = frames
        input["mask"] = mask
        input["action_label"] = action_label
        input["action_idx"] = action_idx
        input["verb_label"] = action_label[0]
        input["noun_label"] = action_label[1]

        return input, index

    def __len__(self):
        return len(self._uid)

# ----------------- TEST BLOCK (self-contained) -----------------
if __name__ == "__main__":
    import logging
    from types import SimpleNamespace
    import torch
    import numpy as np
    from PIL import Image

    logging.basicConfig(level=logging.INFO)

    # ---- Minimal config (EDIT THESE PATHS) ----
    cfg = SimpleNamespace(
        fewshot_eval_json_path="/standard/UVA-DSA/Keshara/MM-CDFSL/cdfsl/target/egoems/annotations/ego_val_mm_cdfsl.json",   # your JSON
        target_data_dir="/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/MM-CDFSL_Format/Test/",                    # root that contains egoems/rgb/<video_id>/frame_*.jpg
        target_sampling_rate=8,                                    # frames stride inside each clip
    )

    # ---- Minimal transform matching your Dataset's expectations ----
    # Dataset expects: transform((frames, None)) -> tensor shaped [T*3, H, W] for RGB,
    # because it later reshapes to [C, T, H, W].
    class SimpleTransform:
        def __init__(self, size=224):
            self.size = size

        def __call__(self, data):
            # data is (frames, None) for RGB in your code
            frames, _ = data if isinstance(data, tuple) else (data, None)
            tensors = []
            for im in frames:
                # ensure PIL.Image
                if not isinstance(im, Image.Image):
                    # handles a fallback path if im is a numpy array
                    im = Image.fromarray(np.array(im))
                im = im.convert("RGB").resize((self.size, self.size))
                t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0  # [3,H,W]
                tensors.append(t)
            # Stack to [T,3,H,W] then flatten to [T*3,H,W] (what your dataset expects)
            stacked = torch.stack(tensors, dim=0)                  # [T,3,H,W]
            flat = stacked.view(-1, self.size, self.size)          # [T*3,H,W]
            return flat, None

    # ---- Dummy mask generator (model-agnostic) ----
    # Your dataset just passes this through; shape/content won't matter for the test.
    def mask_gen():
        return torch.zeros(1, dtype=torch.bool)

    # ---- Instantiate dataset ----
    NUM_FRAMES = 16  # choose T you want the dataset to load per clip
    dataset = EgoEMSFewshotEvalDataset(
        cfg=cfg,
        transform=SimpleTransform(size=224),
        mask_gen=mask_gen,
        num_frames=NUM_FRAMES,
        mode="RGB",  # or "flow"/"pose" if your directories exist
    )

    print(f"Dataset size: {len(dataset)} clips")
    if len(dataset) == 0:
        raise SystemExit("No samples found. Check cfg.fewshot_eval_json_path and frame directories.")

    # ---- Inspect one item ----
    sample, idx = dataset[0]
    print(f"Index: {idx}")
    print(f"Frames shape: {sample['frames'].shape}   (expected [3, T, H, W])")
    print(f"Mask shape:   {sample['mask'].shape}")
    print(f"Action idx:   {sample['action_idx']}")
    print(f"Verb/Noun:    {sample['verb_label']}, {sample['noun_label']}")

    # ---- Optional: iterate with a DataLoader ----
    from torch.utils.data import DataLoader

    def collate(batch):
        # batch is list of (input_dict, idx); stack frames and masks
        inputs, indices = zip(*batch)
        frames = torch.stack([x["frames"] for x in inputs], dim=0)   # [B, 3, T, H, W]
        masks  = torch.stack([x["mask"]   for x in inputs], dim=0)   # [B, ...]
        action_idx = torch.tensor([x["action_idx"] for x in inputs], dtype=torch.long)
        verb = torch.tensor([x["verb_label"] for x in inputs], dtype=torch.long)
        noun = torch.tensor([x["noun_label"] for x in inputs], dtype=torch.long)
        return {
            "frames": frames,
            "mask": masks,
            "action_idx": action_idx,
            "verb_label": verb,
            "noun_label": noun,
        }, torch.tensor(indices, dtype=torch.long)

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)

    for batch, idxs in loader:
        print(f"\nBatch frames: {batch['frames'].shape}  (B,3,T,H,W)")
        print(f"Batch masks:  {batch['mask'].shape}")
        print(f"Action idx:   {batch['action_idx']}")
        break  # just one batch for the smoke test
