import numpy as np
import torch
from typing import Iterator
import logging
from typing import Iterator, List, Dict

from torch.utils.data.sampler import BatchSampler


logger = logging.getLogger(__name__)

class EpisodicBatchSampler:
    def __init__(
        self, dataset, batch_sampler=None, n_way=5, k_shot=5, q_sample=15, episodes=600
    ):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.q_sample = int(q_sample)
        self.episodes = int(episodes)
        self.batch_size = self.n_way * (self.k_shot + self.q_sample)

        # Build per-class pools
        action_idx_list = torch.LongTensor(dataset._action_idx)
        action_idx_vals = action_idx_list.numpy()
        unique_classes: List[int] = sorted(list(set(action_idx_vals)))

        # Keep only classes that have at least 1 sample
        self.action_idx_to_indices: Dict[int, np.ndarray] = {}
        for a in unique_classes:
            inds = np.where(action_idx_vals == a)[0]
            if len(inds) > 0:
                inds = inds.copy()
                np.random.shuffle(inds)
                self.action_idx_to_indices[a] = inds

        self.action_idx_set: List[int] = list(self.action_idx_to_indices.keys())
        if len(self.action_idx_set) < self.n_way:
            raise ValueError(
                f"Only {len(self.action_idx_set)} classes have >=1 clip, "
                f"but n_way={self.n_way}. Reduce n_way or add data."
            )

        # Rolling pointer per class
        self.used_action_idx_indices_count: Dict[int, int] = {
            a: 0 for a in self.action_idx_set
        }

    def _choose_n_classes(self) -> List[int]:
        # choose n distinct class labels from available classes
        return list(np.random.choice(self.action_idx_set, size=self.n_way, replace=False))

    def _take_k_from_class(self, action_label: int, need: int) -> List[int]:
        """
        Return exactly `need` sample indices for this class.
        Uses a rolling pointer into the class pool; if it runs out,
        top-up with replacement from the full pool.
        """
        pool = self.action_idx_to_indices[action_label]
        ptr = self.used_action_idx_indices_count[action_label]
        n_pool = len(pool)

        # Fast path: enough remain in the current slice
        end = ptr + need
        if end <= n_pool:
            chosen = pool[ptr:end].tolist()
            self.used_action_idx_indices_count[action_label] = end
            # If we exactly hit the end, wrap and reshuffle for next time
            if self.used_action_idx_indices_count[action_label] == n_pool:
                np.random.shuffle(pool)
                self.action_idx_to_indices[action_label] = pool
                self.used_action_idx_indices_count[action_label] = 0
            return chosen

        # Slow path: not enough remain—take the tail, then top-up with replacement
        tail = pool[ptr:].tolist()
        short = need - len(tail)
        if n_pool == 0:
            raise RuntimeError(f"Class {action_label} has no samples.")
        # top-up by sampling with replacement from entire pool
        topup = np.random.choice(pool, size=short, replace=True).tolist()
        chosen = tail + topup

        # We wrapped: reshuffle pool and advance pointer by `need % n_pool`
        np.random.shuffle(pool)
        self.action_idx_to_indices[action_label] = pool
        self.used_action_idx_indices_count[action_label] = need % n_pool

        # Safety: ensure exact count
        if len(chosen) != need:
            logger.warning(
                f"[EpisodicBatchSampler] class {action_label}: expected {need}, got {len(chosen)}"
            )
        return chosen

    def __iter__(self) -> Iterator[List[int]]:
        iteration = 0
        per_class = self.k_shot + self.q_sample

        while iteration < self.episodes:
            class_labels = self._choose_n_classes()
            indices: List[int] = []

            for a in class_labels:
                picks = self._take_k_from_class(a, per_class)
                indices.extend(picks)

            # Absolute safety: guarantee exact batch size
            if len(indices) != self.batch_size:
                # Trim or pad with last index to reach exact size
                if len(indices) > self.batch_size:
                    indices = indices[: self.batch_size]
                else:
                    pad = self.batch_size - len(indices)
                    indices += [indices[-1]] * pad
                logger.warning(
                    f"[EpisodicBatchSampler] Adjusted episode size to {self.batch_size}."
                )

            yield indices
            iteration += 1

    def __len__(self):
        return self.episodes
