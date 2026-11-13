"""Vision + proprio + task-id BC policy. Images arrive uint8 in dataset orientation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_shift(x: torch.Tensor, pad: int = 6) -> torch.Tensor:
    """Pad + random crop per sample (translation only, no flips)."""
    n, c, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (n,)).tolist()
    oy = torch.randint(0, 2 * pad + 1, (n,)).tolist()
    return torch.stack([xp[i, :, oy[i]: oy[i] + h, ox[i]: ox[i] + w] for i in range(n)])


class ConvEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()

        def block(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))

        self.net = nn.Sequential(
            block(3, 32, 2), block(32, 32, 1),
            block(32, 64, 2), block(64, 64, 1),
            block(64, 128, 2), block(128, 128, 1),
            block(128, 256, 2),
        )
        self.proj = nn.Linear(256 * 2, out_dim)

    def forward(self, x):
        f = self.net(x)
        f = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return self.proj(f)


class VisionProprioPolicy(nn.Module):
    def __init__(self, proprio_dim: int, action_dim: int = 7, num_tasks: int = 1, num_cameras: int = 2, hidden: int = 512):
        super().__init__()
        self.num_cameras = num_cameras
        self.encoders = nn.ModuleList([ConvEncoder(256) for _ in range(num_cameras)])
        self.task_emb = nn.Embedding(num_tasks, 64)
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(256 * num_cameras + 64 + 64, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim),
        )
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image_u8: torch.Tensor, proprio: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        x = image_u8.float() / 255.0
        if self.training:
            x = random_shift(x)
        feats = []
        for i, enc in enumerate(self.encoders):
            xi = (x[:, 3 * i: 3 * i + 3] - self.img_mean) / self.img_std
            feats.append(enc(xi))
        h = torch.cat(feats + [self.task_emb(task_id), self.proprio(proprio)], dim=1)
        out = self.head(h)
        # translation/rotation in [-1, 1] like LIBERO actions
        return torch.tanh(out)
