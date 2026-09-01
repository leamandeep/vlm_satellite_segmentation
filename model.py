# !pip install git+https://github.com/openai/CLIP.git


import glob
import os
import random

import numpy as np
import rasterio as rio
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.functional.segmentation import mean_iou

mask_files = glob.glob("C:\\Users\\ORSAC\\Downloads\\training_chips\\training_chips\\chip_*.mask.tif")
image_files = glob.glob("C:\\Users\\ORSAC\\Downloads\\training_chips\\training_chips\\chip_*_merged.tif")



device = "cuda"


from terratorch.models import EncoderDecoderFactory

# Initialize the model directly using EncoderDecoderFactory
factory = EncoderDecoderFactory()
from terratorch.tasks import SemanticSegmentationTask

task = SemanticSegmentationTask(
    model_factory="EncoderDecoderFactory",
    model_args={
        # "task": "segmentation",   <-- REMOVE this line
        "backbone": "prithvi_eo_v2_tiny_tl",
        "backbone_pretrained": False,
        "backbone_num_frames": 3,
        "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        "backbone_coords_encoding": [],
        "necks": [
            {"name": "SelectIndices", 
            # "indices": [5, 11, 17, 23]
            "indices": [2, 5, 8, 11]
             },
            {"name": "ReshapeTokensToImage", "effective_time_dim": 3},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        "decoder": "UNetDecoder",
        "decoder_channels": [512, 256, 128, 64],
        "head_dropout": 0.1,
        "num_classes": 13,
    },
    freeze_backbone=True,
    freeze_decoder=False,
)

import albumentations
import terratorch
from albumentations.pytorch import ToTensorV2
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
import albumentations as A





import clip
import torch
import torch.nn as nn
from clip.model import CLIP


device = "cuda" if torch.cuda.is_available() else "cpu"


clip_model = CLIP(
    embed_dim=512,

    # Image encoder: ViT-B/32
    image_resolution=224,
    vision_layers=12,
    vision_width=768,
    vision_patch_size=32,

    # Text encoder
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
).to(device)





import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl



TRAIN_PROMPTS = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans",
    "Wetlands", "Developed / Barren", "Open Water", "Winter Wheat",
    "Alfalfa", "Fallow / Idle Cropland", "Cotton", "Sorghum", "Other"
]


class_to_synonyms = {
    "Natural Vegetation": ["natural vegetation", "vegetation", "flora", "natural greenery", "greenery", "wild growth", "plant cover"],
    "Forest": ["forest", "forests", "woodland", "woods", "dense trees", "tree cover", "forested area"],
    "Corn": ["corn", "maize", "cornfield", "maize fields"],
    "Soybeans": ["soybeans", "soybean", "soy", "soy fields"],
    "Wetlands": ["wetlands", "wetland", "marsh", "swamp", "bog", "marshland"],
    "Developed / Barren": ["developed / barren", "barren land", "built-up area", "urban land", "developed area", "bare soil", "constructed zone"],
    "Open Water": ["open water", "water", "water bodies", "lakes", "lake", "rivers", "river", "ponds", "reservoir"],
    "Winter Wheat": ["winter wheat", "wheat", "wheat fields", "wheat crops"],
    "Alfalfa": ["alfalfa", "lucerne", "alfalfa field"],
    "Fallow / Idle Cropland": ["fallow / idle cropland", "fallow land", "idle cropland", "unplanted fields", "barren cropland", "resting farmland"],
    "Cotton": ["cotton", "cotton crop", "cotton fields"],
    "Sorghum": ["sorghum", "milo", "sorghum field"],
    "Other": ["other", "miscellaneous", "other land cover", "unclassified regions"]
}

TEMPLATES = [
    "a satellite photo of {}",
    "an aerial image showing {}",
    "{}",
]


class PromptablePrithviSegmentationLightning(pl.LightningModule):
    def __init__(
        self,
        full_model,
        clip_model,
        embedding_dim=512,
        image_size=224,
        lr=1e-4,
        weight_decay=1e-2,
        freeze_clip=True,
        # freeze_full_model=False,
        ignore_index=255,
        num_classes = 13
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["full_model", "clip_model"])

        self.backbone = full_model.encoder
        self.necks = full_model.neck
        self.decoder = full_model.decoder
        self.clip_model = clip_model

        self.prompts = TRAIN_PROMPTS  # Full vocabulary list (N items)
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.weights = torch.tensor([0.002, 0.0105, 0.0119, 0.0162, 0.0145, 0.0035, 0.0019, 0.04, 0.0058, 0.0025, 0.0138, 0.0021, 0.8751])



        if freeze_clip:
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False



        self.image_projection = nn.Conv2d(64, embedding_dim, kernel_size=1)
        self.register_buffer("text_center", torch.zeros(embedding_dim if False else 512))
        self._center_initialized = False

        # LSeg fixed or learnable temperature scaling (Paper sets t = 0.07 -> logit_scale ~ 14.28)
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        # CrossEntropyLoss configured with ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index , weight=self.weights)

    def _embed_raw(self, phrases):
        """Templated CLIP embedding for a list of phrases, no centering."""
        all_prompts = [t.format(p) for p in phrases for t in TEMPLATES]
        tokens = clip.tokenize(all_prompts).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_text(tokens).float()
        feats = feats.view(len(phrases), len(TEMPLATES), -1).mean(dim=1)  # avg over templates
        return feats  

    def sample_training_prompts(self):
        prompts = []
        for cls in self.prompts:
            choices = [cls] + class_to_synonyms.get(cls, [])
            prompts.append(random.choice(choices))
        return prompts

    def encode_image(self, x):
        features = self.backbone(x)
        features = self.necks(features)
        image_features = self.decoder(features)

        if isinstance(image_features, (tuple, list)):
            image_features = image_features[-1]

        # Project to text embedding dimension
        image_features = self.image_projection(image_features)
        
        # L2-normalize pixel embeddings along channel dimension (Equation 1)
        image_features = F.normalize(image_features, p=2, dim=1)
        return image_features

    def _init_text_center(self):
        # Compute the centering vector ONCE, from the canonical class vocabulary,
        # and freeze it for the lifetime of the model (saved in checkpoint via register_buffer).
        with torch.no_grad():
            raw = self._embed_raw(self.prompts)  # canonical 13 classes
            self.text_center.copy_(raw.mean(dim=0))
        self._center_initialized = True

    def encode_text(self, prompts=None):
        if prompts is None:
            prompts = self.prompts
        if not self._center_initialized:
            self._init_text_center()

        raw = self._embed_raw(prompts)
        centered = raw - self.text_center
        centered = F.normalize(centered, p=2, dim=-1)
        return centered

    def compute_logits(self, img_feats, text_feats):
        """
        Computes Word-Pixel Correlation Tensor f_ijk = I_ij . T_k
        img_feats:  [B, C, H_feat, W_feat]
        text_feats: [N, C]
        Returns:    [B, N, H_feat, W_feat]
        """
        scale = self.logit_scale.exp().clamp(max=100).to(img_feats.dtype)
        
        # Einstein summation for dot product correlation tensor
        logits = scale * torch.einsum("bchw,nc->bnhw", img_feats, text_feats)
        return logits

    def forward(self, x, prompts=None):
        x = rearrange(x, "b (c t) h w -> b c t h w", c=6, t=3)
        img_feats = self.encode_image(x)
        text_feats = self.encode_text(prompts)
        
        logits = self.compute_logits(img_feats, text_feats)

        # Upsample logits ONLY during inference / validation for dense evaluation
        logits = F.interpolate(
            logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits

    def training_step(self, batch, batch_idx):
        x, y = self._extract_batch(batch)
        y = self._prepare_mask(y)

        # 1. Image features at native decoder resolution (e.g., [B, 512, H_feat, W_feat])
        x_reshaped = rearrange(x, "b (c t) h w -> b c t h w", c=6, t=3)
        img_feats = self.encode_image(x_reshaped)


        # 2. Extract full N-class vocabulary text features
        text_feats = self.encode_text(self.sample_training_prompts())

        # 3. Compute low-resolution logit correlation matrix
        logits_lowres = self.compute_logits(img_feats, text_feats)

        # 4. Downsample Ground Truth Mask using Nearest-Neighbor to avoid smooth loss boundaries
        _, _, h_feat, w_feat = logits_lowres.shape
        y_lowres = F.interpolate(
            y.unsqueeze(1).float(),
            size=(h_feat, w_feat),
            mode="nearest"
        ).squeeze(1).long()

        # 5. Compute loss on sharp low-resolution features (Matches LSeg Eq 2)
        loss = self.criterion(logits_lowres, y_lowres)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._extract_batch(batch)
        y = self._prepare_mask(y)

        logits = self(x)  # Uses full resolution forward pass
        loss = self.criterion(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        mIoU = mean_iou(preds, y, num_classes=self.num_classes, per_class=False)
        self.log("val_mIoU", mIoU.mean(), prog_bar=True, sync_dist=True)        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def _extract_batch(self, batch):
        if isinstance(batch, dict):
            return batch["image"], batch["mask"]
        return batch[0], batch[1]

    def _prepare_mask(self, y):
        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]
        return y.long()


    def _extract_batch(self, batch):
        if isinstance(batch, dict):
            return batch["image"], batch["mask"]
        return batch[0], batch[1]

    def _prepare_mask(self, y):
        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]
        return y.long()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


prompt_model = PromptablePrithviSegmentationLightning(
    full_model=task.model,
    clip_model=clip_model,
    embedding_dim=512,
    image_size=224,
    freeze_clip=True,
)

