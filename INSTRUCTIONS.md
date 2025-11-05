# Few-Shot Training & Testing Guide

This note collects the key steps for adapting MM-CDFSL to new video data and evaluating few-shot performance.

## 1. Environment Setup
- Create and activate a virtual environment, then install dependencies with `pip install -r requirements.txt`.
- All commands assume execution from the repository root.
- If you only have one GPU, override the Hydra defaults at launch (e.g. `python3 lit_main_mmdistill.py devices=[0] strategy=null`).

## 2. Data Preparation
- Mirror the folder hierarchy in `DATA_STRUCTURE.md`; every modality (RGB / flow / pose) must be placed where the loaders expect to find frames.
- Produce annotation JSONs that match the bundled examples such as `cdfsl/target/epic/annotations/epic-55_val.json`. Fields like `clips[]`, `action_idx`, and `start_frame` are required by the few-shot datasets in `datamodule/dataset/`.
- For the unlabeled split used during distillation, follow the schema consumed by the `*_UnlabelLoader` classes. The helper `get_frame_str` in each loader shows the expected filename pattern.
- If your dataset does not follow the EPIC/MECCANO/WEAR layouts, create a new YAML in `configs/data_module/dataset/` and update both `netscripts/get_fewshot_eval_dataset.py` and `netscripts/get_unlabel_loader.py` to recognize the new key.

## 3. Configuration Updates
- Edit `configs/data_module/mm_distill_data_module.yaml`:
  - Set `dataset:` to the dataset YAML you want (existing or custom).
  - Update `source_*` fields (Ego4D paths, annotation JSON) and modality-specific normalization if required.
  - Adjust `n_way`, `k_shot`, `q_sample`, or `episodes` to match your evaluation protocol.
- Edit the target dataset YAML (e.g. `configs/data_module/dataset/epic.yaml`) so that `target_data_dir`, `fewshot_eval_json_path`, and `unlabel_json_path` point to your data.
- Replace the placeholder checkpoint entries in `configs/trainer/mm_distill_trainer.yaml` with the absolute paths to the RGB/flow/pose VideoMAE classifier checkpoints you plan to distill from.

## 4. Multimodal Distillation (Few-Shot Adaptation)
Run:
```bash
python3 lit_main_mmdistill.py train=True test=False \
  trainer.ckpt_path='["/abs/path/rgb.pt","/abs/path/flow.pt","/abs/path/pose.pt"]' \
  data_module.dataset=<dataset_key> \
  data_module.n_way=<N> data_module.k_shot=<K> data_module.episodes=<episodes>
```
- The script constructs the Lightning data module (`lit_unlabel_combined_mm_data_module.py`) and trainer (`models/lit_MMDistillTrainer.py`) and begins training.
- Lightning logs to `tensor_board/` and checkpoints to `checkpoints/`. Adjust `devices`, `strategy`, or `batch_size` on the command line if your hardware differs from the defaults.

## 5. Few-Shot Evaluation
Evaluate any checkpoint (either the distilled one or a provided baseline) via:
```bash
python3 lit_main_mmdistill.py train=False test=True \
  ckpt_path=/path/to/lightning.ckpt \
  data_module.n_way=<N> data_module.k_shot=<K> data_module.episodes=<episodes>
```
- The model fits a logistic-regression head on each episode and reports average accuracy (`models/lit_MMDistillTrainer.py`).
- Adjust `episodes` to control the number of sampled few-shot tasks (600 by default).

## 6. Additional Notes
- Ensure each class has enough labeled clips to support `k_shot + q_sample` samples per episode; the sampler reshuffles but cannot invent clips.
- scikit-learn (installed via `requirements.txt`) is required for the logistic-regression evaluator.
- After verifying results, export or archive the distilled student checkpoint for downstream usage.

```bash
python lit_main_mmdistill.py train=False test=True ckpt_path=/standard/UVA-DSA/Keshara/MM-CDFSL/checkpoints/ours_ego4d-epic.pt 
```