#!/usr/bin/env python3
"""
Convolutional Variational Autoencoder (CVAE) for 32x32x3 face images.
- Trains on images in a directory using torchvision ImageFolder.
- Reconstructs inputs and generates new samples from latent space.
- Saves reconstructions and samples periodically.

Folder structure expected (ImageFolder):
  data/faces/train/...
  data/faces/val/...     (optional; if missing, a split from train is used)
You can also just have one folder (e.g., data/faces/all/...), and it will be split.

Dependencies: torch, torchvision, tqdm
"""
import math
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from torchvision.datasets.folder import default_loader
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm

# ----------------------
# Utilities
# ----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class TrainConfig:
    data_root: str = "C:\\Users\\Tryerand Retryer\\Datasets\\faces_32_32"
    out_dir: str = "./CAE_saves"
    image_size: int = 32
    channels: int = 3
    latent_dim: int = 64
    beta: float = 1.0  # KL weight
    batch_size: int = 128
    epochs: int = 50
    lr: float = 2e-4
    wd: float = 0.0
    num_workers: int = 4
    val_split: float = 0.1  # used if no explicit val folder
    seed: int = 42
    amp: bool = True  # mixed precision if CUDA is available
    save_every: int = 1  # epochs
    n_vis: int = 16  # number of images to visualize (recon/sample)
    checkpoint: str | None = out_dir + "/cvae.pt"
    sample_only: bool = False


# ----------------------
# Model
# ----------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), 256, 4, 4)
        logits = self.net(h)
        return logits


class CVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decoder(z)
        return recon_logits, mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.encoder.fc_mu.out_features, device=device)
        logits = self.decoder(z)
        return torch.sigmoid(logits)

class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = [os.path.join(root, f) for f in os.listdir(root)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, 0   # zweites Element ist Dummy-Label

# ----------------------
# Loss
# ----------------------

def vae_loss(recon_logits, x, mu, logvar, beta=1.0):
    b = x.size(0)
    recon = F.binary_cross_entropy_with_logits(recon_logits, x, reduction='sum') / b
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return recon + beta * kld, recon, kld


# ----------------------
# Data
# ----------------------

def build_dataloaders(cfg: TrainConfig):
    tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(cfg.data_root, 'train')
    val_dir = os.path.join(cfg.data_root, 'val')

    has_train = os.path.isdir(train_dir) and any(os.scandir(train_dir))
    has_val = os.path.isdir(val_dir) and any(os.scandir(val_dir))

    if has_train:
        train_ds = datasets.ImageFolder(train_dir, transform=tfm)
        if has_val:
            val_ds = datasets.ImageFolder(val_dir, transform=tfm)
        else:
            val_len = max(1, int(len(train_ds) * cfg.val_split))
            train_len = len(train_ds) - val_len
            train_ds, val_ds = random_split(train_ds, [train_len, val_len])
    else:
        all_ds = FlatImageDataset(cfg.data_root, transform=tfm)
        val_len = max(1, int(len(all_ds) * cfg.val_split))
        train_len = len(all_ds) - val_len
        train_ds, val_ds = random_split(all_ds, [train_len, val_len])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)
    return train_loader, val_loader


# ----------------------
# Training / Eval helpers
# ----------------------

def save_images(tensor, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(tensor, path, nrow=nrow)


def visualize_batch(x, recon_logits, out_dir, step, prefix="train"):
    with torch.no_grad():
        recon = torch.sigmoid(recon_logits)
        grid_in = vutils.make_grid(x[:64], nrow=8)
        grid_out = vutils.make_grid(recon[:64], nrow=8)
        save_images(grid_in, os.path.join(out_dir, f"{prefix}_inputs_step{step}.png"), nrow=8)
        save_images(grid_out, os.path.join(out_dir, f"{prefix}_recons_step{step}.png"), nrow=8)


@torch.no_grad()
def sample_and_save(model: CVAE, device, out_dir, epoch, n_samples=16):
    print("Speichern Bsp-Bilder")
    model.eval()
    samples = model.sample(n_samples, device)
    save_images(samples, os.path.join(out_dir, f"samples_epoch{epoch:04d}.png"), nrow=int(math.sqrt(n_samples)))


# ----------------------
# Main train loop
# ----------------------

def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = os.path.join(cfg.out_dir, timestamp())
    os.makedirs(run_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")

    train_loader, val_loader = build_dataloaders(cfg)

    model = CVAE(in_channels=cfg.channels, latent_dim=cfg.latent_dim).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == 'cuda'))

    start_epoch = 1
    if cfg.checkpoint and os.path.isfile(cfg.checkpoint):
        state = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optim'])
        start_epoch = state.get('epoch', 1)
        print(f"Loaded checkpoint from {cfg.checkpoint} (epoch {start_epoch})")

    best_val = float('inf')

    for epoch in range(start_epoch, start_epoch + cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        running = {'loss': 0.0, 'recon': 0.0, 'kld': 0.0}
        for i, (x, _) in enumerate(pbar, 1):
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == 'cuda')):
                recon_logits, mu, logvar = model(x)
                loss, recon, kld = vae_loss(recon_logits, x, mu, logvar, beta=cfg.beta)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running['loss'] += loss.item()
            running['recon'] += recon.item()
            running['kld'] += kld.item()

            if i % 50 == 0 or i == len(train_loader):
                pbar.set_postfix({
                    'loss': f"{running['loss']/i:.3f}",
                    'recon': f"{running['recon']/i:.3f}",
                    'kld': f"{running['kld']/i:.3f}",
                })

        model.eval()
        val_loss = val_recon = val_kld = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                recon_logits, mu, logvar = model(x)
                loss, recon, kld = vae_loss(recon_logits, x, mu, logvar, beta=cfg.beta)
                val_loss += loss.item()
                val_recon += recon.item()
                val_kld += kld.item()
        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kld /= len(val_loader)
        print(f"Val - loss: {val_loss:.3f} | recon: {val_recon:.3f} | kld: {val_kld:.3f}")

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            try:
                x_vis, _ = next(iter(val_loader))
                x_vis = x_vis.to(device)
                logits_vis, _, _ = model(x_vis)
                visualize_batch(x_vis, logits_vis, run_dir, step=epoch, prefix="val")
            except StopIteration:
                pass
            sample_and_save(model, device, run_dir, epoch, n_samples=cfg.n_vis)

            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'cfg': cfg.__dict__,
            }
            torch.save(state, os.path.join(run_dir, 'cvae.pt'))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__},
                       os.path.join(run_dir, 'best.pt'))

    print("Training finished. Check samples and reconstructions in:", run_dir)


@torch.no_grad()
def sample_only(cfg: TrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert cfg.checkpoint and os.path.isfile(cfg.checkpoint), "checkpoint is required for sample_only"
    payload = torch.load(cfg.checkpoint, map_location=device)
    state = payload['model'] if 'model' in payload else payload
    model = CVAE(in_channels=3, latent_dim=payload.get('cfg', {}).get('latent_dim', cfg.latent_dim))
    model.load_state_dict(state)
    model.to(device).eval()

    out_dir = os.path.join(os.path.dirname(cfg.checkpoint), 'samples_only')
    os.makedirs(out_dir, exist_ok=True)
    samples = model.sample(cfg.n_vis, device)
    save_images(samples, os.path.join(out_dir, f"samples_{timestamp()}.png"), nrow=int(math.sqrt(cfg.n_vis)))
    print("Saved:", out_dir)


if __name__ == '__main__':
    cfg = TrainConfig()  # use defaults
    if cfg.sample_only:
        sample_only(cfg)
    else:
        train(cfg)