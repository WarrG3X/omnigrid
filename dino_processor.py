#!/usr/bin/env python
# coding: utf-8

import os
import time
import urllib
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.runner import load_checkpoint

import math
import itertools
from functools import partial
import torch.nn.functional as F

from dinov2.eval.depth.models import build_depther

# Constants
CAMERA_DIR = "data_selected/scamera5/"
DEPTH_DIR = "data_selected/depth5_grayscale/"
BACKBONE_SIZE = "large"
HEAD_DATASET = "nyu"
HEAD_TYPE = "dpt"

# Ensure output directory exists
os.makedirs(DEPTH_DIR, exist_ok=True)

# Model URLs
DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

# Load the backbone model
backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.cuda()

# Load the configuration and depth head
def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

cfg_str = load_config_from_url(head_config_url)
cfg = Config.fromstring(cfg_str, file_format=".py")

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther

model = create_depther(cfg, backbone_model, BACKBONE_SIZE, HEAD_TYPE)
load_checkpoint(model, head_checkpoint_url, map_location="cpu")
model.eval()
model.cuda()

# Image transformation pipeline
def make_depth_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3],  # Discard alpha component and scale by 255
        transforms.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
    ])

# Depth rendering
def render_depth(values, colormap_name="magma_r"):
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)
    colormap = plt.get_cmap(colormap_name)
    colors = colormap(normalized_values, bytes=True)[:, :, :3]  # Discard alpha component
    return Image.fromarray(colors)

# Process each image in the directory
t0 = time.time()
transform = make_depth_transform()
for filename in os.listdir(CAMERA_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(CAMERA_DIR, filename)
        image = Image.open(image_path).convert("RGB")
        
        transformed_image = transform(image)
        batch = transformed_image.unsqueeze(0).cuda()
        
        with torch.inference_mode():
            result = model.whole_inference(batch, img_meta=None, rescale=True)
        
        # depth_image = render_depth(result)
        result = torch.permute(result[0], (1, 2, 0)).detach().cpu().numpy()
        depth_image = Image.fromarray(result)
        print(depth_image.shape)
        depth_image.save(os.path.join(DEPTH_DIR, f"depth_{filename}"))

print("Processing complete.")
print(f"Elapsed time: {time.time() - t0:.1f} seconds")
