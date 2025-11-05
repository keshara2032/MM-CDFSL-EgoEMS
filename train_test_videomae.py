import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Repo imports
from datamodule.utils.augmentation import DataAugmentationForVideoMAERGB
from models.videomae_classifier import videomae_classifier_small_patch16_224


def build_transform(num_frames: int, input_size: int, multi_scale_crop: bool):
    # mask disabled for classifier train/eval
    aug_cfg = SimpleNamespace(mask_type="none", mask_ratio=0.0)
    return DataAugmentationForVideoMAERGB(
        aug_cfg,
        num_frames=num_frames,
        input_size=input_size,
        patch_size=[16, 16],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        multi_scale_crop=multi_scale_crop,
    )


class ClipJsonDataset(Dataset):
    """
    Generic RGB clip dataset using target JSON annotations in this repo’s style.

    Supports datasets: epic, meccano, wear, egoems.
    Each item returns tensor [3, T, H, W] and integer action index.
    """

    def __init__(
        self,
        dataset: str,
        json_path: str,
        data_root: str,
        num_frames: int = 16,
        sampling_rate: int = 8,
        transform=None,
        epic_split_dir: str = "train",
    ):
        self.dataset = dataset.lower()
        self.json_path = json_path
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.epic_split_dir = epic_split_dir  # EPIC frames subdir name under frames/rgb/

        with open(self.json_path, "r") as f:
            data = json.load(f)

        # Accept train/val/test; do not assert
        self.num_actions = data.get("num_actions", None)
        self._records = []

        for clip in data["clips"]:
            if self.dataset == "epic":
                participant_id = clip["participant_id"]
                video_id = clip["video_id"]
                start_frame = int(clip["start_frame"])
                action_idx = int(clip["action_idx"])  # 0..(C-1)
                dir_frames = self.data_root / "frames" / "rgb" / self.epic_split_dir / participant_id / video_id
            elif self.dataset == "egoems":
                video_id = clip["video_id"]
                start_frame = int(clip["start_frame"])
                action_idx = int(clip["action_idx"])  # 0..(C-1)
                dir_frames = self.data_root / "egoems" / "rgb" / video_id
            elif self.dataset == "meccano":
                video_id = clip["video_id"]
                start_frame = int(clip["start_frame"])
                action_idx = int(clip["action_id_internal"])  # 0..(C-1)
                dir_frames = self.data_root / "RGB_frames" / video_id
            elif self.dataset == "wear":
                video_id = clip["video_id"]
                start_frame = int(clip["start_frame"])
                action_idx = int(clip["label_id_internal"])  # 0..(C-1)
                dir_frames = self.data_root / "RGB_frames" / video_id
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset}")

            self._records.append(
                {
                    "dir": dir_frames,
                    "start": start_frame,
                    "label": action_idx,
                }
            )

        # Infer num_classes from labels if not provided
        if self.num_actions is None:
            max_label = max(r["label"] for r in self._records)
            self.num_actions = max_label + 1

    def __len__(self):
        return len(self._records)

    def _frame_path(self, dir_frames: Path, frame_idx: int) -> Path:
        if self.dataset == "epic" or self.dataset == "egoems":
            return dir_frames / f"frame_{str(frame_idx).zfill(10)}.jpg"
        elif self.dataset == "meccano":
            return dir_frames / f"{str(frame_idx).zfill(5)}.jpg"
        elif self.dataset == "wear":
            return dir_frames / f"{str(frame_idx).zfill(6)}.jpg"
        else:
            raise ValueError

    def __getitem__(self, idx):
        rec = self._records[idx]
        dir_frames = rec["dir"]
        start = rec["start"]
        label = rec["label"]

        # Sample frame indices
        frame_indices = [max(1, start + self.sampling_rate * i) for i in range(self.num_frames)]
        frames = []
        last = None
        for fi in frame_indices:
            p = self._frame_path(dir_frames, fi)
            if p.exists():
                im = Image.open(str(p)).convert("RGB")
                last = im
            else:
                if last is None:
                    # Backoff: try to find the smallest existing frame near start
                    last = Image.new("RGB", (224, 224), color=(0, 0, 0))
                im = last
            frames.append(im)

        # Apply repo transform: expects (frames, None) for RGB; returns [T*3,H,W]
        x, _ = self.transform((frames, None)) if self.transform is not None else frames
        # Reshape to [C,T,H,W]
        if isinstance(x, torch.Tensor):
            x = x.view(self.num_frames, 3, *x.shape[-2:]).permute(1, 0, 2, 3).contiguous()
        else:
            raise RuntimeError("Transform must return a tensor")

        return x, torch.tensor(label, dtype=torch.long)


def collate_rgb(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B,3,T,H,W]
    y = torch.stack(ys, dim=0)  # [B]
    return x, y


def train_one_epoch(model, loader, optimizer, scaler, device, log_interval=50):
    model.train()
    ce = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    correct = 0

    for it, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            _, logits = model(x, mask=None)
            loss = ce(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        total += x.size(0)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()

        if (it + 1) % log_interval == 0:
            print(f"Iter {it+1:05d} | loss {loss.item():.4f}")

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, logits = model(x, mask=None)
        pred = logits.argmax(dim=1)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Train/Test VideoMAE classifier on RGB clips")
    parser.add_argument("--dataset", type=str, required=True, choices=["epic", "meccano", "wear", "egoems"], help="Target dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Root to frames as in DATA_STRUCTURE.md")
    parser.add_argument("--train_json", type=str, required=True, help="Path to training annotation JSON")
    parser.add_argument("--test_json", type=str, required=True, help="Path to testing annotation JSON")
    parser.add_argument("--init_ckpt", type=str, required=True, help="Init checkpoint for VideoMAE encoder/classifier (see Model Zoo)")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epic_split_dir", type=str, default="train", help="EPIC frames subdir under frames/rgb (default: train)")
    parser.add_argument("--save_dir", type=str, default="checkpoints_videomae")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Transforms
    t_train = build_transform(args.num_frames, args.img_size, multi_scale_crop=True)
    t_test = build_transform(args.num_frames, args.img_size, multi_scale_crop=False)

    # Datasets
    ds_train = ClipJsonDataset(
        dataset=args.dataset,
        json_path=args.train_json,
        data_root=args.data_root,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        transform=t_train,
        epic_split_dir=args.epic_split_dir,
    )
    ds_test = ClipJsonDataset(
        dataset=args.dataset,
        json_path=args.test_json,
        data_root=args.data_root,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        transform=t_test,
        epic_split_dir=args.epic_split_dir,
    )

    # Determine num classes from train JSON if available
    num_classes = ds_train.num_actions
    print(f"Num classes: {num_classes}")

    # DataLoaders
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_rgb,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_rgb,
    )

    # Model
    model = videomae_classifier_small_patch16_224(
        ckpt_pth=args.init_ckpt,
        img_size=args.img_size,
        patch_size=16,
        in_chans=3,
        num_classes_action=num_classes,
        use_mean_pooling=True,
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and torch.cuda.is_available()) else None

    best_acc = 0.0
    best_path = None
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, dl_train, optimizer, scaler, device)
        test_acc = evaluate(model, dl_test, device)
        print(f"Train loss {train_loss:.4f} | Train acc {train_acc*100:.2f}% | Test acc {test_acc*100:.2f}%")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = os.path.join(args.save_dir, f"videomae_best_acc_{best_acc:.4f}.pt")
            torch.save({"state_dict": model.state_dict(), "num_classes": num_classes}, best_path)
            print(f"Saved new best to {best_path}")

    # Final save
    final_path = os.path.join(args.save_dir, "videomae_last.pt")
    torch.save({"state_dict": model.state_dict(), "num_classes": num_classes}, final_path)
    print(f"Final checkpoint: {final_path}")
    if best_path:
        print(f"Best test acc: {best_acc*100:.2f}% at {best_path}")


if __name__ == "__main__":
    main()

