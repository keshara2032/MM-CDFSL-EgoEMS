<div align="center">

# Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition (ECCV'24)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

**[[Paper](https://masashi-hatano.github.io/assets/pdf/mm-cdfsl.pdf)][[Supplementary](https://masashi-hatano.github.io/assets/pdf/mm-cdfsl_supp.pdf)][[Project Page](https://masashi-hatano.github.io/MM-CDFSL/)][[Poster](https://masashi-hatano.github.io/assets/pdf/mm-cdfsl_poster.pdf)][[Data](https://huggingface.co/datasets/masashi-hatano/MM-CDFSL/tree/main)]**

</div>

This is the official code release for our ECCV 2024 paper \
"Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition".

## 🔨 Installation
```bash
# Create a virtual environment
python3 -m venv mm-cdfsl
source mm-cdfsl/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

## 📂 Data Preparation
### Training Split
You can find the train/val split files for all three target datasets in `cdfsl` folders.

### Data Structure
Please follow the data structure as detailed in [DATA_STRUCTURE.md](https://github.com/masashi-hatano/MM-CDFSL/blob/main/DATA_STRUCTURE.md).

### Pre-processed Data
You can download the pre-processed data from the [hub](https://huggingface.co/datasets/masashi-hatano/MM-CDFSL/tree/main).

## 📍 Model Zoo
You can brouse the checkpoints of pre-trained model, comparison methods, and our models in this [folder](https://keiojp0-my.sharepoint.com/:f:/g/personal/hatano1210_keio_jp/EoqKPBpzMDRAqalq6m2zjvEBdOywFzpMnW6CwWfeTNb95A?e=4ZQctt) or directly download from the following links.

### Pre-Train

<div align="center">

|  Method  | Source Dataset | Target Dataset | Modality | Ckpt |
| :------: | :------------: | :------------: | :------: | :--: |
| VideoMAE | Kinetics-400 | - | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EXaK54A5D7RBkCT4S7ARcNYBtUfb2P3K0AApujNuHBy6Ig?e=IzEwXK) |
| VideoMAE | Ego4D | - | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EbsfvRKG-0BJiTda6-NViN0BIbft7lMmJ24lrqgBT9XvHA?e=yr3rkR) |
| VideoMAE w/ classifier | Ego4D | EPIC | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EdbvlOrr8ZdFjEtBO0q0i_QBtKS8t8UijQFP3eceZKsaEw?e=XzU9ZJ) |
| VideoMAE w/ classifier | Ego4D | EPIC | flow | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EWBJmu4hY6ZGq-DVdi6j2YcBZmIc8VZgwxcJXpqtLjqZVA?e=tkfUY3) |
| VideoMAE w/ classifier | Ego4D | EPIC | pose | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EfR3OzJdiM5KhV9okCp2EPABLW0fps2u9ae2V-MpNA0iaA?e=pPtyyK) |
| VideoMAE w/ classifier | Ego4D | MECCANO | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/ERWg5T4tvdxCirQeHvYowRABVMzY6SuLSoA-K9bBOE9fQw?e=wWFo1m) |
| VideoMAE w/ classifier | Ego4D | MECCANO | flow | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EaTh3eApwkJPl2_yyZQNrfwBGkDGflCOKPYqDQjiePclVQ?e=ZpXY7E) |
| VideoMAE w/ classifier | Ego4D | MECCANO | pose | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/ESHqY2VvflxCtLBulultKFkB4idHAxoGj2vJKSnWShx5wQ?e=EqdJSl) |
| VideoMAE w/ classifier | Ego4D | WEAR | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EW6x78JK9qRCnl61BbpLFGIBWufJWzo36yzk1Hpmq1ngcA?e=NxXq23) |
| VideoMAE w/ classifier | Ego4D | WEAR | flow | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/ESyhgb-YVJNBtFXTRuuLHoQBVuPANrK06H9hoIzd0sCbcQ?e=DTrAp2) |
| VideoMAE w/ classifier | Ego4D | WEAR | pose | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EcLT4rjhtZBKoykNxeTtKEoBN0hSDxW18zcYAZe5LgV--A?e=GPUU44) |

</div>


### 2nd Stage

<div align="center">

|  Method  | Source Dataset | Target Dataset | Modality | Ckpt |
| :------: | :------------: | :------------: | :------: | :--: |
| STARTUP++ | Ego4D | EPIC | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EZ1U6bqBp8hHgCUC0Xe47aMBfaIcDh3t4vztqfoz62ZJZg?e=SmjJV6) |
| STARTUP++ | Ego4D | MECCANO | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/ETa92sOiNYZJtV5j-sa20QUBrKPumQY-Vau-fg8jpz2qKA?e=3SOobP) |
| STARTUP++ | Ego4D | WEAR | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EXsICTNcFNZCkS7iBsdfR5EB_UK3MAQRTlxasEjVCIglxQ?e=jLkCZe) |
| Dynamic Distill++ | Ego4D | EPIC | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EctXZUmdEmlElij5DSQN7VUBJGHWKaAEf0J6BopgRdXNNA?e=gwHLf1) |
| Dynamic Distill++ | Ego4D | MECCANO | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EVP8YSbPhKNFthpaRJ-VsesBSOyfixyPQYB3TBL8jsHI9Q?e=GZBLKM) |
| Dynamic Distill++ | Ego4D | WEAR | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/Ec5U1DltaXFFvzauQxS9280Bo6vjPXDtqXlMGsb7ToUCGQ?e=Q9jMoe) |
| CDFSL-V | Ego4D | EPIC | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EdhYd-oNWtpGqqBKgcNydJQByUTpEhrZMNd7hgpTRtikJw?e=4CaDc6) |
| CDFSL-V | Ego4D | MECCANO | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/ERZ-1-p89qNDuYI_HFn7pVwBRPXAodkvkftDAHdRf5semQ?e=8NpMeL) |
| CDFSL-V | Ego4D | WEAR | RGB | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EeliefqhUypBhvd0kmv7oDoB5uxRty80PG3NzR9VmuRu9Q?e=8uSndq) |
| Ours | Ego4D | EPIC | RGB, flow, pose | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/Edy0Y9TM3P9PgCtEVUxsd8IBSU8ev02kcVjr1TqNEFZX5w?e=bQMbk8) |
| Ours | Ego4D | MECCANO | RGB, flow, pose | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EYW6FcUc0mhOghEy-rK3fHQBPLbnQvzz2p6FjK9cSz_LAw?e=Abcr1b) |
| Ours | Ego4D | WEAR | RGB, flow, pose | [checkpoint](https://keiojp0-my.sharepoint.com/:u:/g/personal/hatano1210_keio_jp/EXz18Z6qbolIhUIvbqTpzqEBxQ8Z7PU0du_xe5reNtdhOw?e=qWD3sT) |

</div>

## 🔥 Training
### 1. Pre-training
Please make sure that you set modality (e.g., rgb) and dataset (e.g., epic) in `configs/trainer/pretrain_trainer.yaml` and `confings/data_module/pretrain_data_module.yaml`.
```bash
python3 lit_main_pretrain.py train=True test=False
```

### 2. Multimodal Distillation
Please make sure that you set dataset (e.g., epic) in `confings/data_module/mm_distill_data_module.yaml`.
Also, you need to set the ckpt path of all modalities in `configs/trainer/mm_distill_trainer.yaml`.
```bash
python3 lit_main_mmdistill.py train=True test=False 
```

## 🔍 Evaluation
To evaluate the model in 5-way 5-shot setting with 600 runs, please run the following command.
```bash
python3 lit_main_mmdistill.py train=False test=True data_module.n_way=5 data_module.k_shot=5 data_module.episodes=600
```

## ✍️ Citation
If you use this code for your research, please cite our paper.
```bib
@inproceedings{Hatano2024MMCDFSL,
  author = {Hatano, Masashi and Hachiuma, Ryo and Fujii, Ryo and Saito, Hideo},
  title = {Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024},
}
```
