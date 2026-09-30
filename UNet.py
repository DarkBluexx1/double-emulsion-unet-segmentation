import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import numpy as np
import random
import argparse


IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 50
VAL_SPLIT = 0.2
MODEL_SAVE_PATH = 'droplet_unet.pth'
DATA_DIR = None
SEED = 42

UNET_DEPTH = 4
UNET_START_CH = 64
UNET_DROPOUT = 0.3
UNET_BATCHNORM = True
IN_CHANNELS = 4

BCE_WEIGHT = 0.4
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
BOUNDARY_W0 = 8.0
BOUNDARY_SIGMA = 6.0

GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 12
USE_AMP = torch.cuda.is_available()

AUG_PROB = 0.8
AUG_BLUR_PROB = 0.4
AUG_NOISE_PROB = 0.3
AUG_ROTATION_MAX = 30
AUG_ZOOM_RANGE = (0.85, 1.15)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train U-Net for droplet jet segmentation.')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset root containing images/ and masks/')
    parser.add_argument('--save-path', type=str, default='droplet_unet.pth',
                        help='Where to save the best model checkpoint')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, batchnorm=True, residual=True):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)

        layers = []
        for i, (ic, oc) in enumerate([(in_ch, out_ch), (out_ch, out_ch)]):
            layers.append(nn.Conv2d(ic, oc, 3, padding=1, bias=not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0 and i == 1:
                layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out = out + x
        return out


class ImprovedUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, start_ch=64, depth=4,
                 dropout=0.3, batchnorm=True):
        super().__init__()
        self.depth = depth
        ch = [start_ch * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_ch
        for i in range(depth):
            self.encoders.append(ConvBlock(prev, ch[i], batchnorm=batchnorm, residual=False))
            self.pools.append(nn.MaxPool2d(2))
            prev = ch[i]

        self.bottleneck = ConvBlock(ch[depth - 1], ch[depth],
                                    dropout=dropout, batchnorm=batchnorm, residual=False)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2))
            self.decoders.append(ConvBlock(ch[i] * 2, ch[i], batchnorm=batchnorm, residual=False))

        self.final = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.final(x)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, probs, target):
        smooth = 1e-6
        p = probs.reshape(-1)
        t = target.reshape(-1)
        TP = (p * t).sum()
        FP = ((1 - t) * p).sum()
        FN = (t * (1 - p)).sum()
        tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        return 1 - tversky


class BoundaryWeightedTverskyLoss(nn.Module):
    def __init__(self, bce_weight=0.4, tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky = TverskyLoss(tversky_alpha, tversky_beta)

    def forward(self, logits, target, weight_map):
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight_map)
        probs = torch.sigmoid(logits)
        tv = self.tversky(probs, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * tv


def dice_score(probs, target, threshold=0.5):
    pred_bin = (probs > threshold).float()
    smooth = 1e-6
    intersection = (pred_bin * target).sum()
    return ((2 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)).item()


def compute_boundary_weight_map(mask_uint8, w0=8.0, sigma=6.0):
    edges = cv2.Canny(mask_uint8, 50, 150)
    if edges.max() == 0:
        return np.ones(mask_uint8.shape, dtype=np.float32)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    weight = 1.0 + w0 * np.exp(-(dist ** 2) / (2 * sigma ** 2))
    return weight.astype(np.float32)


class Augmenter:
    def __call__(self, image, mask):
        if random.random() > AUG_PROB:
            return image, mask

        h, w = image.shape[:2]

        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if random.random() < 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        if random.random() < 0.6:
            angle = random.uniform(-AUG_ROTATION_MAX, AUG_ROTATION_MAX)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT)

        if random.random() < 0.5:
            scale = random.uniform(*AUG_ZOOM_RANGE)
            new_h, new_w = int(h * scale), int(w * scale)
            image_r = cv2.resize(image, (new_w, new_h))
            mask_r = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if scale > 1.0:
                y0 = (new_h - h) // 2
                x0 = (new_w - w) // 2
                image = image_r[y0:y0 + h, x0:x0 + w]
                mask = mask_r[y0:y0 + h, x0:x0 + w]
            else:
                image = np.zeros((h, w, 3), dtype=np.float32)
                mask = np.zeros((h, w), dtype=np.float32)
                y0 = (h - new_h) // 2
                x0 = (w - new_w) // 2
                image[y0:y0 + new_h, x0:x0 + new_w] = image_r
                mask[y0:y0 + new_h, x0:x0 + new_w] = mask_r

        if random.random() < 0.4:
            tx = random.randint(-w // 8, w // 8)
            ty = random.randint(-h // 8, h // 8)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT,
                                  flags=cv2.INTER_NEAREST)

        if random.random() < AUG_BLUR_PROB:
            ksize = random.choice([3, 5, 7, 9])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        if random.random() < 0.2:
            k = random.choice([5, 7, 9])
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            image = cv2.filter2D(image, -1, kernel)

        if random.random() < 0.5:
            alpha = random.uniform(0.75, 1.25)
            beta = random.uniform(-0.1, 0.1)
            image = np.clip(alpha * image + beta, 0, 1)

        if random.random() < AUG_NOISE_PROB:
            sigma = random.uniform(0.01, 0.04)
            noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)

        mask = (mask > 0.5).astype(np.float32)
        return image, mask


MASK_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp']
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


class DropletDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.images = sorted(os.listdir(self.img_dir))
        self.augmenter = Augmenter() if augment else None

    def __len__(self):
        return len(self.images)

    def _find_mask_path(self, name):
        base = os.path.splitext(name)[0]
        for ext in MASK_EXTENSIONS:
            candidate = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def __getitem__(self, idx):
        name = self.images[idx]
        img_path = os.path.join(self.img_dir, name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

        mask_path = self._find_mask_path(name)
        if mask_path is None:
            raise FileNotFoundError(
                f"No mask found for '{name}' in {self.mask_dir}\n"
                f"  Tried extensions: {MASK_EXTENSIONS}"
            )
        mask_raw = cv2.imread(mask_path, 0)
        if mask_raw is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask_raw = cv2.resize(mask_raw, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

        image_f = image.astype(np.float32) / 255.0
        mask_f = (mask_raw / 255.0).astype(np.float32)

        if self.augmenter is not None:
            image_f, mask_f = self.augmenter(image_f, mask_f)

        mask_u8 = (mask_f * 255).astype(np.uint8)
        weight_map = compute_boundary_weight_map(mask_u8, BOUNDARY_W0, BOUNDARY_SIGMA)

        gray_u8 = cv2.cvtColor((image_f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        clahe_u8 = _CLAHE.apply(gray_u8)
        clahe_f = clahe_u8.astype(np.float32) / 255.0

        stacked = np.dstack([image_f, clahe_f])
        stacked = np.transpose(stacked, (2, 0, 1))
        mask_out = np.expand_dims(mask_f, axis=0)
        weight_out = np.expand_dims(weight_map, axis=0)

        return (torch.tensor(stacked, dtype=torch.float32),
                torch.tensor(mask_out, dtype=torch.float32),
                torch.tensor(weight_out, dtype=torch.float32))


if __name__ == '__main__':
    args = parse_args()
    DATA_DIR = args.data_dir
    MODEL_SAVE_PATH = args.save_path
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    SEED = args.seed

    seed_everything(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("DROPLET U-NET TRAINING")
    print("=" * 60)
    print(f"Device        : {device}")
    print(f"Input size    : {IMG_WIDTH}x{IMG_HEIGHT}  |  channels: {IN_CHANNELS}")
    print(f"U-Net depth   : {UNET_DEPTH}  (channels: "
          f"{', '.join(str(UNET_START_CH * 2**i) for i in range(UNET_DEPTH+1))})")
    print(f"Loss          : {BCE_WEIGHT} * boundary-weighted BCE + "
          f"{1-BCE_WEIGHT:.1f} * Tversky(a={TVERSKY_ALPHA}, b={TVERSKY_BETA})")
    print(f"Epochs (max)  : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"Early stop    : patience={EARLY_STOP_PATIENCE} epochs on val Dice")
    print(f"AMP           : {USE_AMP}")

    img_dir = os.path.join(DATA_DIR, 'images')
    if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
        print(f"\nNo images found in '{img_dir}'.")
        print(f"Make sure your dataset folder has images/ and masks/ subfolders.")
        exit()

    full_dataset = DropletDataset(DATA_DIR, augment=False)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_dataset = DropletDataset(DATA_DIR, augment=True)
    val_dataset = DropletDataset(DATA_DIR, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=SubsetRandomSampler(train_indices.indices),
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            sampler=SubsetRandomSampler(val_indices.indices),
                            num_workers=2, pin_memory=True, persistent_workers=True)

    print(f"\nDataset       : {n_total} images  ->  {n_train} train / {n_val} val")

    model = ImprovedUNet(in_ch=IN_CHANNELS, out_ch=1,
                         start_ch=UNET_START_CH, depth=UNET_DEPTH,
                         dropout=UNET_DROPOUT, batchnorm=UNET_BATCHNORM).to(device)
    criterion = BoundaryWeightedTverskyLoss(BCE_WEIGHT, TVERSKY_ALPHA, TVERSKY_BETA)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters    : {n_params:,}")
    print("\n" + "=" * 60)

    best_val_dice = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()
        train_loss = 0.0
        train_dice = 0.0
        for images, masks, weights in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(images)
                loss = criterion(logits, masks, weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_dice += dice_score(torch.sigmoid(logits.detach()), masks)

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks, weights in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                logits = model(images)
                val_loss += criterion(logits, masks, weights).item()
                val_dice += dice_score(torch.sigmoid(logits), masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f}  Dice: {train_dice:.3f} | "
              f"Val loss: {val_loss:.4f}  Dice: {val_dice:.3f} | "
              f"LR: {lr_now:.2e}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
                'epoch': epoch,
                'in_channels': IN_CHANNELS,
            }, MODEL_SAVE_PATH)
            print(f"  New best saved  (val Dice={val_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no val Dice improvement in "
                      f"{EARLY_STOP_PATIENCE} epochs.")
                break

    print("\n" + "=" * 60)
    print(f"Training complete.")
    print(f"Best val Dice : {best_val_dice:.4f}")
    print(f"Best val Loss : {best_val_loss:.4f}")
    print(f"Model saved   : {MODEL_SAVE_PATH}")
    print("=" * 60)