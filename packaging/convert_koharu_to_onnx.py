"""One-time Koharu Local Cache -> ONNX / GGUF Bridge for Vellum (Phase 10.1).

Auto-discovers Koharu's local Hugging Face cache (default:
`C:\\FILE SOFTWARE\\Koharu\\store\\hugging-face\\models\\`), resolves each model's
`snapshots/<hash>/` directory, and bridges all 4 Koharu models into Vellum's
persistent storage (`%LOCALAPPDATA%\\ComicTranslator\\models`):

1. `mayocream--koharu-layout-rfdetr-seg-2xl-1152` (`.safetensors` -> `koharu-layout-rfdetr-seg-2xl-1152.onnx`)
2. `mayocream--lama-manga` (`.safetensors` -> `lama_manga.onnx`, DirectML/CUDA/CPU safe MatMul DFT)
3. `mayocream--manga-ocr` (`.safetensors` -> `manga-ocr/encoder_model.onnx`, `decoder_model.onnx`, `vocab.txt`)
4. `PaddlePaddle--PaddleOCR-VL-1.6-GGUF` (copies `.gguf` + `-mmproj.gguf` -> `paddleocr-vl-gguf/` for LM Studio Vision-OCR)

Usage (using existing `backend/venv`):
    backend\\venv\\Scripts\\python.exe packaging\\convert_koharu_to_onnx.py
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from safetensors.torch import load_file as load_safetensors
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel


DEFAULT_KOHARU_CACHE = Path(r"C:\FILE SOFTWARE\Koharu\store\hugging-face\models")

KOHARU_FOLDERS = {
    "rfdetr": "mayocream--koharu-layout-rfdetr-seg-2xl-1152",
    "lama": "mayocream--lama-manga",
    "manga_ocr": "mayocream--manga-ocr",
    "paddleocr_vl": "PaddlePaddle--PaddleOCR-VL-1.6-GGUF",
}


def is_windows_store_python() -> bool:
    """Return True when running under Microsoft Store (MSIX/WindowsApps) Python, which virtualizes %LOCALAPPDATA%."""
    return "windowsapps" in str(sys.base_prefix).lower() or "windowsapps" in str(sys.executable).lower()


def get_default_output_dir() -> Path:
    """Resolve `%LOCALAPPDATA%\\ComicTranslator\\models`."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data).expanduser().resolve() / "ComicTranslator" / "models"
    return (Path.home() / "AppData" / "Local" / "ComicTranslator" / "models").resolve()


def resolve_koharu_snapshot(cache_root: Path, folder_name: str) -> Path | None:
    """Resolve `<cache_root>/<folder_name>/snapshots/<hash>/` preferring the newest valid snapshot."""
    snapshots_root = cache_root / folder_name / "snapshots"
    if not snapshots_root.is_dir():
        return None

    candidates: list[tuple[float, Path]] = []
    for entry in snapshots_root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        valid_files = [
            f
            for f in entry.iterdir()
            if f.is_file()
            and not f.name.startswith(".")
            and not f.name.endswith(".lock")
            and f.suffix.lower() in {".safetensors", ".onnx", ".gguf"}
            and f.stat().st_size > 1024
        ]
        if valid_files:
            newest_mtime = max(f.stat().st_mtime for f in valid_files)
            candidates.append((newest_mtime, entry))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def atomic_copy_file(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` atomically via a `.part` temporary file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(dst.name + ".part")
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _legacy_onnx_export(
    model: nn.Module,
    args: tuple[Tensor, ...] | Tensor,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    """Export a PyTorch module to a single self-contained ONNX file atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".part.onnx")
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    export_kwargs: dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "export_params": True,
        "keep_initializers_as_inputs": False,
        "do_constant_folding": True,
        "opset_version": opset_version,
    }
    if dynamic_axes is not None:
        export_kwargs["dynamic_axes"] = dynamic_axes
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    try:
        with torch.no_grad():
            torch.onnx.export(model, args, str(tmp_path), **export_kwargs)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# =============================================================================
# 1. Standalone PyTorch RF-DETR-Seg-2XL (matches rfdetr==1.7.0 weights 1:1)
# =============================================================================


class _RF_PatchEmbeddings(nn.Module):
    def __init__(self, image_size: int = 1152, patch_size: int = 12, num_channels: int = 3, hidden_size: int = 384):
        super().__init__()
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class _RF_WindowedDinov2Embeddings(nn.Module):
    def __init__(self, image_size: int = 1152, patch_size: int = 12, hidden_size: int = 384, num_windows: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_size))
        self.patch_embeddings = _RF_PatchEmbeddings(image_size, patch_size, 3, hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.patch_embeddings.num_patches + 1, hidden_size))

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch_size, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings

        if self.num_windows > 1:
            num_h = height // self.patch_size
            num_w = width // self.patch_size
            nw = self.num_windows
            cls_tok = embeddings[:, :1]
            pix_tok = embeddings[:, 1:].view(batch_size, num_h, num_w, -1)
            win_tok = pix_tok.reshape(batch_size * nw, num_h // nw, nw, num_w // nw, -1)
            win_tok = win_tok.permute(0, 2, 1, 3, 4).reshape(
                batch_size * (nw * nw), (num_h // nw) * (num_w // nw), -1
            )
            win_cls = cls_tok.repeat(nw * nw, 1, 1)
            embeddings = torch.cat((win_cls, win_tok), dim=1)
        return embeddings


class _RF_Dinov2SelfAttention(nn.Module):
    def __init__(self, hidden_size: int = 384, num_heads: int = 6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key = nn.Linear(hidden_size, hidden_size, bias=True)
        self.value = nn.Linear(hidden_size, hidden_size, bias=True)

    def _transpose(self, x: Tensor) -> Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states: Tensor) -> Tensor:
        q = self._transpose(self.query(hidden_states))
        k = self._transpose(self.key(hidden_states))
        v = self._transpose(self.value(hidden_states))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        return ctx.view(ctx.size()[:-2] + (self.all_head_size,))


class _RF_Dinov2SelfOutput(nn.Module):
    def __init__(self, hidden_size: int = 384):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.dense(hidden_states)


class _RF_Dinov2Attention(nn.Module):
    def __init__(self, hidden_size: int = 384, num_heads: int = 6):
        super().__init__()
        self.attention = _RF_Dinov2SelfAttention(hidden_size, num_heads)
        self.output = _RF_Dinov2SelfOutput(hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.output(self.attention(hidden_states))


class _RF_Dinov2LayerScale(nn.Module):
    def __init__(self, hidden_size: int = 384):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_state: Tensor) -> Tensor:
        return hidden_state * self.lambda1


class _RF_Dinov2MLP(nn.Module):
    def __init__(self, hidden_size: int = 384, mlp_ratio: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * mlp_ratio, bias=True)
        self.fc2 = nn.Linear(hidden_size * mlp_ratio, hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _RF_WindowedDinov2Layer(nn.Module):
    def __init__(self, hidden_size: int = 384, num_heads: int = 6, num_windows: int = 2):
        super().__init__()
        self.num_windows = num_windows
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = _RF_Dinov2Attention(hidden_size, num_heads)
        self.layer_scale1 = _RF_Dinov2LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = _RF_Dinov2MLP(hidden_size, 4)
        self.layer_scale2 = _RF_Dinov2LayerScale(hidden_size)

    def forward(self, hidden_states: Tensor, run_full_attention: bool = False) -> Tensor:
        shortcut = hidden_states
        nw2 = self.num_windows * self.num_windows
        if run_full_attention:
            bw, tpw, c = hidden_states.shape
            hidden_states = hidden_states.view(bw // nw2, nw2 * tpw, c)

        attn_out = self.attention(self.norm1(hidden_states))

        if run_full_attention:
            b, total_t, c = hidden_states.shape
            attn_out = attn_out.view(b * nw2, total_t // nw2, c)

        hidden_states = self.layer_scale1(attn_out) + shortcut
        mlp_out = self.layer_scale2(self.mlp(self.norm2(hidden_states)))
        return mlp_out + hidden_states


class _RF_WindowedDinov2Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 384,
        num_heads: int = 6,
        num_windows: int = 2,
        full_attn_indexes: tuple[int, ...] = (3, 6, 9),
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [_RF_WindowedDinov2Layer(hidden_size, num_heads, num_windows) for _ in range(num_layers)]
        )
        self.full_attn_indexes = set(full_attn_indexes)

    def forward(self, hidden_states: Tensor) -> list[Tensor]:
        all_states: list[Tensor] = []
        for idx, layer_mod in enumerate(self.layer):
            all_states.append(hidden_states)
            hidden_states = layer_mod(hidden_states, run_full_attention=(idx in self.full_attn_indexes))
        all_states.append(hidden_states)
        return all_states


class _RF_WindowedDinov2Backbone(nn.Module):
    def __init__(
        self,
        image_size: int = 1152,
        patch_size: int = 12,
        hidden_size: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        num_windows: int = 2,
        out_feature_indexes: tuple[int, ...] = (3, 6, 9, 12),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.out_feature_indexes = out_feature_indexes
        self.embeddings = _RF_WindowedDinov2Embeddings(image_size, patch_size, hidden_size, num_windows)
        full_attn = tuple(i for i in out_feature_indexes if i < num_layers)
        self.encoder = _RF_WindowedDinov2Encoder(num_layers, hidden_size, num_heads, num_windows, full_attn)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, pixel_values: Tensor) -> list[Tensor]:
        batch_size, _, height, width = pixel_values.shape
        num_h = height // self.patch_size
        num_w = width // self.patch_size
        nw = self.num_windows
        nw2 = nw * nw

        emb = self.embeddings(pixel_values)
        all_states = self.encoder(emb)

        feature_maps: list[Tensor] = []
        for idx in self.out_feature_indexes:
            hs = self.layernorm(all_states[idx])
            hs = hs[:, 1:]  # drop CLS token (num_register_tokens = 0)
            if nw > 1:
                bw, tpw, c = hs.shape
                hs = hs.reshape(bw // nw2, nw2 * tpw, c)
                hs = hs.reshape((bw // nw2) * nw, nw, num_h // nw, num_w // nw, c)
                hs = hs.permute(0, 2, 1, 3, 4)
            hs = hs.reshape(batch_size, num_h, num_w, -1).permute(0, 3, 1, 2).contiguous()
            feature_maps.append(hs)
        return feature_maps


class _RF_DinoV2Wrapper(nn.Module):
    def __init__(self, resolution: int = 1152):
        super().__init__()
        self.encoder = _RF_WindowedDinov2Backbone(
            image_size=resolution,
            patch_size=12,
            hidden_size=384,
            num_layers=12,
            num_heads=6,
            num_windows=2,
            out_feature_indexes=(3, 6, 9, 12),
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        return self.encoder(x)


class _RF_ChannelsFirstLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
        self.normalized_shape = (channels,)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class _RF_ConvX(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False
        )
        self.bn = _RF_ChannelsFirstLayerNorm(out_planes)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x.contiguous())))


class _RF_Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.cv1 = _RF_ConvX(c1, c2, kernel=3, stride=1)
        self.cv2 = _RF_ConvX(c2, c2, kernel=3, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.cv2(self.cv1(x))


class _RF_C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 3):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = _RF_ConvX(c1, 2 * self.c, kernel=1, stride=1)
        self.cv2 = _RF_ConvX((2 + n) * self.c, c2, kernel=1, stride=1)
        self.m = nn.ModuleList([_RF_Bottleneck(self.c, self.c) for _ in range(n)])

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        for block in self.m:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class _RF_MultiScaleProjector(nn.Module):
    def __init__(self, in_channels: int = 1536, out_channels: int = 256, num_blocks: int = 3):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    _RF_C2f(in_channels, out_channels, n=num_blocks),
                    _RF_ChannelsFirstLayerNorm(out_channels),
                )
            ]
        )

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        fused = torch.cat(feats, dim=1)
        return [self.stages[0](fused)]


class _RF_Backbone(nn.Module):
    def __init__(self, resolution: int = 1152):
        super().__init__()
        self.encoder = _RF_DinoV2Wrapper(resolution=resolution)
        self.projector = _RF_MultiScaleProjector(in_channels=384 * 4, out_channels=256, num_blocks=3)

    def forward(self, x: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        feats = self.projector(self.encoder(x))
        masks = [torch.zeros((f.shape[0], f.shape[2], f.shape[3]), dtype=torch.bool, device=f.device) for f in feats]
        return feats, masks


class _RF_PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2.0 * math.pi

    def forward(self, mask: Tensor) -> Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class _RF_Joiner(nn.Sequential):
    def __init__(self, backbone: nn.Module, pos_embed: nn.Module):
        super().__init__(backbone, pos_embed)

    def forward(self, inputs: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        feats, masks = self[0](inputs)
        poss = [self[1](m).to(f.dtype) for f, m in zip(feats, masks)]
        return feats, poss


class _RF_MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _rf_gen_sineembed_for_position(pos_tensor: Tensor, dim: int = 128) -> Tensor:
    scale = 2.0 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    coords = []
    for idx in (1, 0, 2, 3):  # y, x, w, h
        embed = pos_tensor[:, :, idx] * scale
        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        coords.append(pos)
    return torch.cat(coords, dim=2)


def _rf_gen_encoder_output_proposals(memory: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, height - 1, height, dtype=torch.float32, device=memory.device),
        torch.linspace(0, width - 1, width, dtype=torch.float32, device=memory.device),
        indexing="ij",
    )
    grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1)
    scale = torch.tensor([[[[float(width), float(height)]]]], dtype=torch.float32, device=memory.device)
    grid = (grid.unsqueeze(0) + 0.5) / scale
    wh = torch.ones_like(grid) * 0.05
    output_proposals = torch.cat((grid, wh), dim=-1).reshape(-1, height * width, 4)

    valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = output_proposals.masked_fill(~valid, 0.0)
    output_memory = memory.masked_fill(~valid, 0.0)
    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


def _rf_bilinear_grid_sample(inp: Tensor, grid: Tensor) -> Tensor:
    """Gather-based exact bilinear grid sample (100% DirectML/CUDA/CPU compatible, zero GridSample EP gaps)."""
    batch_size, channels, height, width = inp.shape
    grid_h, grid_w = grid.shape[1], grid.shape[2]

    ix = (grid[..., 0] + 1.0) * (float(width) * 0.5) - 0.5
    iy = (grid[..., 1] + 1.0) * (float(height) * 0.5) - 0.5

    ix0 = ix.floor().long()
    iy0 = iy.floor().long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    wx1 = (ix - ix0.float()).to(inp.dtype).unsqueeze(1)
    wy1 = (iy - iy0.float()).to(inp.dtype).unsqueeze(1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    in_x0 = (ix0 >= 0) & (ix0 < width)
    in_x1 = (ix1 >= 0) & (ix1 < width)
    in_y0 = (iy0 >= 0) & (iy0 < height)
    in_y1 = (iy1 >= 0) & (iy1 < height)

    ix0_c = ix0.clamp(0, width - 1)
    iy0_c = iy0.clamp(0, height - 1)
    ix1_c = ix1.clamp(0, width - 1)
    iy1_c = iy1.clamp(0, height - 1)

    flat = inp.flatten(2)  # (N, C, H*W)

    def _gather(iy_: Tensor, ix_: Tensor) -> Tensor:
        idx = (iy_ * width + ix_).flatten(1).unsqueeze(1).expand(batch_size, channels, -1)
        return flat.gather(2, idx).view(batch_size, channels, grid_h, grid_w)

    v00 = _gather(iy0_c, ix0_c) * (in_x0 & in_y0).to(inp.dtype).unsqueeze(1)
    v10 = _gather(iy0_c, ix1_c) * (in_x1 & in_y0).to(inp.dtype).unsqueeze(1)
    v01 = _gather(iy1_c, ix0_c) * (in_x0 & in_y1).to(inp.dtype).unsqueeze(1)
    v11 = _gather(iy1_c, ix1_c) * (in_x1 & in_y1).to(inp.dtype).unsqueeze(1)

    return wx0 * wy0 * v00 + wx1 * wy0 * v10 + wx0 * wy1 * v01 + wx1 * wy1 * v11


class _RF_MSDeformAttn(nn.Module):
    def __init__(self, d_model: int = 256, n_levels: int = 1, n_heads: int = 16, n_points: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, reference_points: Tensor, input_flatten: Tensor, height: int, width: int) -> Tensor:
        bs, len_q, _ = query.shape
        _, len_in, _ = input_flatten.shape
        head_dim = self.d_model // self.n_heads

        value = self.value_proj(input_flatten)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attn_weights = self.attention_weights(query).view(
            bs, len_q, self.n_heads, self.n_levels * self.n_points
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets / float(self.n_points) * reference_points[:, :, None, :, None, 2:] * 0.5
        )
        attn_weights = F.softmax(attn_weights, dim=-1)

        value = value.transpose(1, 2).contiguous().view(bs * self.n_heads, head_dim, height, width)
        sampling_grid = (2.0 * sampling_locations[:, :, :, 0] - 1.0).transpose(1, 2).flatten(0, 1)
        sampled = _rf_bilinear_grid_sample(value, sampling_grid)  # (bs*n_heads, head_dim, len_q, n_points)

        attn_w = attn_weights.transpose(1, 2).reshape(bs * self.n_heads, 1, len_q, self.n_points)
        out = (sampled * attn_w).sum(-1).view(bs, self.d_model, len_q).transpose(1, 2).contiguous()
        return self.output_proj(out)


class _RF_TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, sa_nhead: int = 8, ca_nhead: int = 16, dim_ff: int = 2048, dec_n_points: int = 2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=sa_nhead, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = _RF_MSDeformAttn(d_model, n_levels=1, n_heads=ca_nhead, n_points=dec_n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, memory: Tensor, query_pos: Tensor, reference_points: Tensor, height: int, width: int) -> Tensor:
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + tgt2)
        tgt2 = self.cross_attn(tgt + query_pos, reference_points, memory, height, width)
        tgt = self.norm2(tgt + tgt2)
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        return self.norm3(tgt + tgt2)


class _RF_TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int = 6, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        base_layer = _RF_TransformerDecoderLayer(d_model=d_model, sa_nhead=8, ca_nhead=16, dim_ff=2048, dec_n_points=2)
        self.layers = nn.ModuleList([copy.deepcopy(base_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.ref_point_head = _RF_MLP(2 * d_model, d_model, d_model, 2)

    def forward(self, tgt: Tensor, memory: Tensor, refpoints: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
        query_sine = _rf_gen_sineembed_for_position(refpoints[..., :4], self.d_model // 2)
        query_pos = self.ref_point_head(query_sine)
        refpoints_input = refpoints[:, :, None, :4]

        output = tgt
        for layer in self.layers:
            output = layer(output, memory, query_pos, refpoints_input, height, width)
        output = self.norm(output)
        return output, refpoints


class _RF_Transformer(nn.Module):
    def __init__(self, d_model: int = 256, num_queries: int = 300, num_decoder_layers: int = 6, group_detr: int = 13, num_classes: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.decoder = _RF_TransformerDecoder(num_layers=num_decoder_layers, d_model=d_model)
        self.enc_output = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(group_detr)])
        self.enc_output_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(group_detr)])
        self.enc_out_bbox_embed = nn.ModuleList([_RF_MLP(d_model, d_model, 4, 3) for _ in range(group_detr)])
        self.enc_out_class_embed = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in range(group_detr)])

    def forward(self, src: Tensor, refpoint_embed: Tensor, query_feat: Tensor) -> tuple[Tensor, Tensor]:
        bs, _, height, width = src.shape
        memory = src.flatten(2).transpose(1, 2)
        output_memory, output_proposals = _rf_gen_encoder_output_proposals(memory, height, width)

        output_memory_0 = self.enc_output_norm[0](self.enc_output[0](output_memory))
        enc_cls = self.enc_out_class_embed[0](output_memory_0)
        enc_delta = self.enc_out_bbox_embed[0](output_memory_0)
        enc_cxcy = enc_delta[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
        enc_wh = enc_delta[..., 2:].exp() * output_proposals[..., 2:]
        enc_coord = torch.cat([enc_cxcy, enc_wh], dim=-1)

        topk_idx = torch.topk(enc_cls.max(-1)[0], self.num_queries, dim=1)[1]
        ref_ts = torch.gather(enc_coord, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))

        tgt = query_feat.unsqueeze(0).expand(bs, -1, -1).contiguous()
        ref_q = refpoint_embed.unsqueeze(0).expand(bs, -1, -1).contiguous()
        ref_cxcy = ref_q[..., :2] * ref_ts[..., 2:] + ref_ts[..., :2]
        ref_wh = ref_q[..., 2:].exp() * ref_ts[..., 2:]
        ref_init = torch.cat([ref_cxcy, ref_wh], dim=-1)

        return self.decoder(tgt, memory, ref_init, height, width)


class _RF_DepthwiseConvBlock(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.pwconv1(self.norm(x)))
        return x.permute(0, 3, 1, 2) + residual


class _RF_MLPBlock(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)])

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        return x + residual


class _RF_SegmentationHead(nn.Module):
    def __init__(self, in_dim: int = 256, num_blocks: int = 6, downsample_ratio: int = 4):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.blocks = nn.ModuleList([_RF_DepthwiseConvBlock(in_dim) for _ in range(num_blocks)])
        self.spatial_features_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.query_features_block = _RF_MLPBlock(in_dim)
        self.query_features_proj = nn.Linear(in_dim, in_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, spatial_features: Tensor, query_features: Tensor, image_size: tuple[int, int]) -> Tensor:
        target_h = image_size[0] // self.downsample_ratio
        target_w = image_size[1] // self.downsample_ratio
        sf = F.interpolate(spatial_features, size=(target_h, target_w), mode="bilinear", align_corners=False)
        for block in self.blocks:
            sf = block(sf)
        sf_proj = self.spatial_features_proj(sf)  # (B, C, Hm, Wm)
        qf = self.query_features_proj(self.query_features_block(query_features))  # (B, N, C)
        b, c, hm, wm = sf_proj.shape
        masks = torch.bmm(qf, sf_proj.flatten(2)).view(b, qf.shape[1], hm, wm) + self.bias
        return masks


class KoharuRFDETRSeg2XL(nn.Module):
    """Exact standalone inference/export implementation of `mayocream/koharu-layout-rfdetr-seg-2xl-1152`."""

    def __init__(self, resolution: int = 1152, num_classes: int = 4, num_queries: int = 300, group_detr: int = 13):
        super().__init__()
        self.resolution = resolution
        self.num_queries = num_queries
        self.backbone = _RF_Joiner(_RF_Backbone(resolution=resolution), _RF_PositionEmbeddingSine(128))
        self.transformer = _RF_Transformer(
            d_model=256,
            num_queries=num_queries,
            num_decoder_layers=6,
            group_detr=group_detr,
            num_classes=num_classes + 1,
        )
        self.class_embed = nn.Linear(256, num_classes + 1)
        self.bbox_embed = _RF_MLP(256, 256, 4, 3)
        self.segmentation_head = _RF_SegmentationHead(256, num_blocks=6, downsample_ratio=4)
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, 4)
        self.query_feat = nn.Embedding(num_queries * group_detr, 256)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        srcs, _ = self.backbone(images)
        ref_w = self.refpoint_embed.weight[: self.num_queries]
        qry_w = self.query_feat.weight[: self.num_queries]
        hs, ref = self.transformer(srcs[0], ref_w, qry_w)

        delta = self.bbox_embed(hs)
        cxcy = delta[..., :2] * ref[..., 2:] + ref[..., :2]
        wh = delta[..., 2:].exp() * ref[..., 2:]
        dets = torch.cat([cxcy, wh], dim=-1)
        labels = self.class_embed(hs)
        masks = self.segmentation_head(srcs[0], hs, (self.resolution, self.resolution))
        return dets, labels, masks


# =============================================================================
# 2. Standalone PyTorch LaMa-Manga (DirectML-safe MatMul 2D Orthonormal DFT)
# =============================================================================


class _LaMa_FourierUnit(nn.Module):
    """Big-LaMa FourierUnit with exact MatMul-based 2D orthonormal rfft2/irfft2 (no ONNX DFT op)."""

    def __init__(self, in_channels: int, out_channels: int, feat_size: int = 64):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=False)
        self.feat_size = feat_size

        n = feat_size
        m = n // 2 + 1
        k_full = torch.arange(n, dtype=torch.float32)
        k_half = torch.arange(m, dtype=torch.float32)
        scale = 1.0 / math.sqrt(float(n))

        angles_w = (2.0 * math.pi / n) * torch.outer(k_full, k_half)  # (n, m)
        self.register_buffer("cos_w", torch.cos(angles_w) * scale, persistent=False)
        self.register_buffer("sin_w", -torch.sin(angles_w) * scale, persistent=False)

        angles_h = (2.0 * math.pi / n) * torch.outer(k_full, k_full)  # (n, n)
        self.register_buffer("cos_h", torch.cos(angles_h) * scale, persistent=False)
        self.register_buffer("sin_h", -torch.sin(angles_h) * scale, persistent=False)

        self.register_buffer("icos_h", (torch.cos(angles_h) * scale).t().contiguous(), persistent=False)
        self.register_buffer("isin_h", (torch.sin(angles_h) * scale).t().contiguous(), persistent=False)

        w_mult = torch.full((m, 1), 2.0, dtype=torch.float32)
        w_mult[0, 0] = 1.0
        if n % 2 == 0:
            w_mult[-1, 0] = 1.0
        angles_iw = (2.0 * math.pi / n) * torch.outer(k_half, k_full)  # (m, n)
        self.register_buffer("icos_w", torch.cos(angles_iw) * (scale * w_mult), persistent=False)
        self.register_buffer("isin_w", torch.sin(angles_iw) * (scale * w_mult), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        # 2D real orthonormal rfft2 -> (B, C, H, W//2+1) real & imag
        xr = torch.matmul(x, self.cos_w)
        xi = torch.matmul(x, self.sin_w)
        yr = torch.matmul(self.cos_h, xr) - torch.matmul(self.sin_h, xi)
        yi = torch.matmul(self.cos_h, xi) + torch.matmul(self.sin_h, xr)

        # Match Big-LaMa layout: (B, C, 2, H, W//2+1) -> (B, 2*C, H, W//2+1)
        ffted = torch.stack((yr, yi), dim=2).reshape(b, c * 2, h, w // 2 + 1)
        ffted = self.relu(self.bn(self.conv_layer(ffted)))
        ffted = ffted.reshape(b, c, 2, h, w // 2 + 1)
        fr = ffted[:, :, 0]
        fi = ffted[:, :, 1]

        # 2D orthonormal irfft2 -> (B, C, H, W)
        zr = torch.matmul(self.icos_h, fr) - torch.matmul(self.isin_h, fi)
        zi = torch.matmul(self.icos_h, fi) + torch.matmul(self.isin_h, fr)
        return torch.matmul(zr, self.icos_w) - torch.matmul(zi, self.isin_w)


class _LaMa_SpectralTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, feat_size: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=False),
        )
        self.fu = _LaMa_FourierUnit(out_channels // 2, out_channels // 2, feat_size=feat_size)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return self.conv2(x + self.fu(x))


class _LaMa_FFC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        feat_size: int = 64,
    ):
        super().__init__()
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        self.convl2l: nn.Module = (
            nn.Identity()
            if (in_cl == 0 or out_cl == 0)
            else nn.Conv2d(in_cl, out_cl, kernel_size, stride=stride, padding=padding, bias=False, padding_mode="reflect")
        )
        self.convl2g: nn.Module = (
            nn.Identity()
            if (in_cl == 0 or out_cg == 0)
            else nn.Conv2d(in_cl, out_cg, kernel_size, stride=stride, padding=padding, bias=False, padding_mode="reflect")
        )
        self.convg2l: nn.Module = (
            nn.Identity()
            if (in_cg == 0 or out_cl == 0)
            else nn.Conv2d(in_cg, out_cl, kernel_size, stride=stride, padding=padding, bias=False, padding_mode="reflect")
        )
        self.convg2g: nn.Module = (
            nn.Identity()
            if (in_cg == 0 or out_cg == 0)
            else _LaMa_SpectralTransform(in_cg, out_cg, feat_size=feat_size)
        )

    def forward(self, x: tuple[Tensor, Tensor] | Tensor) -> tuple[Tensor, Tensor]:
        x_l, x_g = x if isinstance(x, tuple) else (x, None)
        out_xl: Tensor | int = 0
        out_xg: Tensor | int = 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l)
            if x_g is not None and self.ratio_gin != 0:
                out_xl = out_xl + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l)
            if x_g is not None and self.ratio_gin != 0:
                out_xg = out_xg + self.convg2g(x_g)

        return out_xl, out_xg  # type: ignore[return-value]


class _LaMa_FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        feat_size: int = 64,
    ):
        super().__init__()
        self.ffc = _LaMa_FFC(
            in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=stride, padding=padding, feat_size=feat_size
        )
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.bn_l: nn.Module = nn.Identity() if out_cl == 0 else nn.BatchNorm2d(out_cl)
        self.bn_g: nn.Module = nn.Identity() if out_cg == 0 else nn.BatchNorm2d(out_cg)
        self.act_l: nn.Module = nn.Identity() if out_cl == 0 else nn.ReLU(inplace=False)
        self.act_g: nn.Module = nn.Identity() if out_cg == 0 else nn.ReLU(inplace=False)
        self.has_g = out_cg > 0

    def forward(self, x: tuple[Tensor, Tensor] | Tensor) -> tuple[Tensor, Tensor]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        if self.has_g:
            x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class _LaMa_FFCResnetBlock(nn.Module):
    def __init__(self, dim: int = 512, ratio_g: float = 0.75, feat_size: int = 64):
        super().__init__()
        self.conv1 = _LaMa_FFC_BN_ACT(dim, dim, kernel_size=3, padding=1, ratio_gin=ratio_g, ratio_gout=ratio_g, feat_size=feat_size)
        self.conv2 = _LaMa_FFC_BN_ACT(dim, dim, kernel_size=3, padding=1, ratio_gin=ratio_g, ratio_gout=ratio_g, feat_size=feat_size)

    def forward(self, x: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        id_l, id_g = x
        x_l, x_g = self.conv1(x)
        x_l, x_g = self.conv2((x_l, x_g))
        return id_l + x_l, id_g + x_g


class _LaMa_ConcatTupleLayer(nn.Module):
    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        return torch.cat(x, dim=1)


class LamaMangaGenerator(nn.Module):
    """Exact Big-LaMa FFCResNetGenerator matching `mayocream/lama-manga` (`lama-manga.safetensors`)."""

    def __init__(self, tile_size: int = 512):
        super().__init__()
        feat_size = tile_size // 8  # 64 for 512x512 tiles
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            _LaMa_FFC_BN_ACT(4, 64, kernel_size=7, padding=0, ratio_gin=0.0, ratio_gout=0.0),
            _LaMa_FFC_BN_ACT(64, 128, kernel_size=3, stride=2, padding=1, ratio_gin=0.0, ratio_gout=0.0),
            _LaMa_FFC_BN_ACT(128, 256, kernel_size=3, stride=2, padding=1, ratio_gin=0.0, ratio_gout=0.0),
            _LaMa_FFC_BN_ACT(256, 512, kernel_size=3, stride=2, padding=1, ratio_gin=0.0, ratio_gout=0.75),
        ]
        for _ in range(18):
            layers.append(_LaMa_FFCResnetBlock(512, ratio_g=0.75, feat_size=feat_size))
        layers.extend(
            [
                _LaMa_ConcatTupleLayer(),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, kernel_size=7, padding=0),
                nn.Sigmoid(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class LamaMangaOnnxWrapper(nn.Module):
    """Wraps `LamaMangaGenerator` to match `lama_fp32.onnx` contract (`image` [0,1], `mask` {0,1} -> `output` [0,255])."""

    def __init__(self, generator: LamaMangaGenerator):
        super().__init__()
        self.generator = generator

    def forward(self, image: Tensor, mask: Tensor) -> Tensor:
        inv_mask = 1.0 - mask
        masked_image = image * inv_mask
        inp = torch.cat([masked_image, mask], dim=1)
        pred = self.generator(inp)
        blended = mask * pred + inv_mask * image
        return torch.clamp(blended * 255.0, 0.0, 255.0)


# =============================================================================
# 3. Manga-OCR Encoder / Decoder ONNX Export Wrappers
# =============================================================================


class _MangaOcrEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.encoder(pixel_values=pixel_values, return_dict=True).last_hidden_state


class _MangaOcrDecoderWrapper(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, input_ids: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        return self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        ).logits


# =============================================================================
# Conversion Drivers
# =============================================================================


def convert_rfdetr(cache_root: Path, output_dir: Path, force: bool = False) -> dict[str, Any]:
    folder = KOHARU_FOLDERS["rfdetr"]
    snap = resolve_koharu_snapshot(cache_root, folder)
    if snap is None:
        return {"status": "missing", "folder": folder, "error": "Koharu snapshot not found"}

    out_onnx = output_dir / "koharu-layout-rfdetr-seg-2xl-1152.onnx"
    out_cfg = output_dir / "koharu-layout-rfdetr-seg-2xl-1152.config.json"

    src_cfg = snap / "inference_config.json"
    if src_cfg.is_file():
        atomic_copy_file(src_cfg, out_cfg)

    # Fast path: snapshot already contains an .onnx file
    existing_onnx = next(
        (p for p in snap.iterdir() if p.is_file() and p.suffix.lower() == ".onnx" and p.stat().st_size > 10_000_000),
        None,
    )
    if existing_onnx is not None:
        if force or not out_onnx.is_file() or out_onnx.stat().st_size != existing_onnx.stat().st_size:
            print(f"[rfdetr] Copying pre-exported ONNX from {existing_onnx.name} ...")
            atomic_copy_file(existing_onnx, out_onnx)
        return {"status": "ok", "source": str(existing_onnx), "output": str(out_onnx), "mode": "copied_onnx"}

    if not force and out_onnx.is_file() and out_onnx.stat().st_size > 100_000_000:
        print(f"[rfdetr] Already exported: {out_onnx} ({out_onnx.stat().st_size / 1e6:.1f} MB)")
        return {"status": "skipped", "output": str(out_onnx)}

    src_st = snap / "model.safetensors"
    if not src_st.is_file():
        return {"status": "error", "folder": folder, "error": f"model.safetensors not found in {snap}"}

    t0 = time.perf_counter()
    print(f"[rfdetr] Loading Koharu RF-DETR-Seg-2XL weights from {src_st} ...")
    model = KoharuRFDETRSeg2XL(resolution=1152, num_classes=4, num_queries=300, group_detr=13)
    state_dict = load_safetensors(str(src_st), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[rfdetr] Verified strict state_dict match ({len(state_dict)} tensors). Exporting to ONNX opset 17 ...")

    dummy = torch.randn(1, 3, 1152, 1152, dtype=torch.float32)
    with torch.no_grad():
        pt_dets, pt_labels, pt_masks = model(dummy)

    _legacy_onnx_export(
        model,
        dummy,
        out_onnx,
        input_names=["input"],
        output_names=["dets", "labels", "masks"],
        opset_version=17,
    )

    print("[rfdetr] Verifying ONNX Runtime numerical parity ...")
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    ort_dets, ort_labels, ort_masks = sess.run(["dets", "labels", "masks"], {"input": dummy.numpy()})
    max_diff_dets = float(np.max(np.abs(pt_dets.numpy() - ort_dets)))
    max_diff_labels = float(np.max(np.abs(pt_labels.numpy() - ort_labels)))
    max_diff_masks = float(np.max(np.abs(pt_masks.numpy() - ort_masks)))
    mean_diff_masks = float(np.mean(np.abs(pt_masks.numpy() - ort_masks)))
    elapsed = time.perf_counter() - t0
    print(
        f"[rfdetr] Exported {out_onnx.name} ({out_onnx.stat().st_size / (1024 * 1024):.2f} MB) "
        f"in {elapsed:.1f}s | max_diff dets={max_diff_dets:.2e}, labels={max_diff_labels:.2e}, "
        f"masks={max_diff_masks:.2e} (mean={mean_diff_masks:.2e})"
    )
    if max_diff_dets > 1e-2 or max_diff_labels > 2e-2 or max_diff_masks > 5e-2 or mean_diff_masks > 1e-3:
        raise RuntimeError(
            f"RF-DETR ONNX parity check failed: dets={max_diff_dets}, labels={max_diff_labels}, "
            f"masks={max_diff_masks} (mean={mean_diff_masks})"
        )

    return {
        "status": "ok",
        "source": str(src_st),
        "output": str(out_onnx),
        "config": str(out_cfg) if out_cfg.is_file() else None,
        "size_bytes": out_onnx.stat().st_size,
        "max_abs_diff": {"dets": max_diff_dets, "labels": max_diff_labels, "masks": max_diff_masks},
    }


def convert_lama(cache_root: Path, output_dir: Path, force: bool = False) -> dict[str, Any]:
    folder = KOHARU_FOLDERS["lama"]
    snap = resolve_koharu_snapshot(cache_root, folder)
    if snap is None:
        return {"status": "missing", "folder": folder, "error": "Koharu snapshot not found"}

    out_onnx = output_dir / "lama_manga.onnx"
    existing_onnx = next(
        (p for p in snap.iterdir() if p.is_file() and p.suffix.lower() == ".onnx" and p.stat().st_size > 50_000_000),
        None,
    )
    if existing_onnx is not None:
        if force or not out_onnx.is_file() or out_onnx.stat().st_size != existing_onnx.stat().st_size:
            atomic_copy_file(existing_onnx, out_onnx)
        return {"status": "ok", "source": str(existing_onnx), "output": str(out_onnx), "mode": "copied_onnx"}

    if not force and out_onnx.is_file() and out_onnx.stat().st_size > 100_000_000:
        print(f"[lama] Already exported: {out_onnx} ({out_onnx.stat().st_size / 1e6:.1f} MB)")
        return {"status": "skipped", "output": str(out_onnx)}

    src_st = next(
        (p for p in snap.iterdir() if p.is_file() and p.suffix.lower() == ".safetensors" and not p.name.startswith(".")),
        None,
    )
    if src_st is None:
        return {"status": "error", "folder": folder, "error": f"No .safetensors found in {snap}"}

    t0 = time.perf_counter()
    print(f"[lama] Loading Koharu LaMa-Manga weights from {src_st} ...")
    gen = LamaMangaGenerator(tile_size=512)
    state_dict = load_safetensors(str(src_st), device="cpu")
    gen.load_state_dict(state_dict, strict=True)
    gen.eval()
    wrapper = LamaMangaOnnxWrapper(gen).eval()
    print(f"[lama] Verified strict state_dict match ({len(state_dict)} tensors). Exporting to ONNX opset 17 ...")

    dummy_img = torch.rand(1, 3, 512, 512, dtype=torch.float32)
    dummy_mask = (torch.rand(1, 1, 512, 512, dtype=torch.float32) > 0.85).float()
    with torch.no_grad():
        pt_out = wrapper(dummy_img, dummy_mask)

    _legacy_onnx_export(
        wrapper,
        (dummy_img, dummy_mask),
        out_onnx,
        input_names=["image", "mask"],
        output_names=["output"],
        opset_version=17,
    )

    print("[lama] Verifying ONNX Runtime numerical parity ...")
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["output"], {"image": dummy_img.numpy(), "mask": dummy_mask.numpy()})[0]
    max_diff = float(np.max(np.abs(pt_out.numpy() - ort_out)))
    elapsed = time.perf_counter() - t0
    print(
        f"[lama] Exported {out_onnx.name} ({out_onnx.stat().st_size / (1024 * 1024):.2f} MB) "
        f"in {elapsed:.1f}s | max_diff={max_diff:.2e}"
    )
    if max_diff > 1e-1:
        raise RuntimeError(f"LaMa-Manga ONNX parity check failed: max_diff={max_diff}")

    return {
        "status": "ok",
        "source": str(src_st),
        "output": str(out_onnx),
        "size_bytes": out_onnx.stat().st_size,
        "max_abs_diff": max_diff,
    }


def convert_manga_ocr(cache_root: Path, output_dir: Path, force: bool = False) -> dict[str, Any]:
    folder = KOHARU_FOLDERS["manga_ocr"]
    snap = resolve_koharu_snapshot(cache_root, folder)
    if snap is None:
        return {"status": "missing", "folder": folder, "error": "Koharu snapshot not found"}

    target_dir = output_dir / "manga-ocr"
    target_dir.mkdir(parents=True, exist_ok=True)
    enc_onnx = target_dir / "encoder_model.onnx"
    dec_onnx = target_dir / "decoder_model.onnx"

    # Always sync tokenizer/config metadata files (`vocab.txt`, `config.json`, `preprocessor_config.json`)
    copied_meta: list[str] = []
    for meta_name in ("vocab.txt", "config.json", "preprocessor_config.json"):
        src_meta = snap / meta_name
        if src_meta.is_file():
            dst_meta = target_dir / meta_name
            atomic_copy_file(src_meta, dst_meta)
            copied_meta.append(meta_name)

    if (
        not force
        and enc_onnx.is_file()
        and enc_onnx.stat().st_size > 100_000_000
        and dec_onnx.is_file()
        and dec_onnx.stat().st_size > 30_000_000
        and (target_dir / "vocab.txt").is_file()
    ):
        print(f"[manga_ocr] Already exported: {target_dir}")
        return {"status": "skipped", "output_dir": str(target_dir), "meta_files": copied_meta}

    t0 = time.perf_counter()
    print(f"[manga_ocr] Loading VisionEncoderDecoderModel from {snap} ...")
    model = VisionEncoderDecoderModel.from_pretrained(str(snap), local_files_only=True)
    model.eval()

    enc_wrapper = _MangaOcrEncoderWrapper(model.encoder).eval()
    dec_wrapper = _MangaOcrDecoderWrapper(model.decoder).eval()

    dummy_pixels = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        pt_enc = enc_wrapper(dummy_pixels)

    print("[manga_ocr] Exporting encoder_model.onnx ...")
    _legacy_onnx_export(
        enc_wrapper,
        dummy_pixels,
        enc_onnx,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        opset_version=17,
        dynamic_axes={"pixel_values": {0: "batch"}, "last_hidden_state": {0: "batch"}},
    )

    dummy_ids = torch.tensor([[2, 10, 20, 30]], dtype=torch.long)
    with torch.no_grad():
        pt_dec = dec_wrapper(dummy_ids, pt_enc)

    print("[manga_ocr] Exporting decoder_model.onnx ...")
    _legacy_onnx_export(
        dec_wrapper,
        (dummy_ids, pt_enc),
        dec_onnx,
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "encoder_hidden_states": {0: "batch"},
            "logits": {0: "batch", 1: "seq_len"},
        },
    )

    print("[manga_ocr] Verifying ONNX Runtime numerical parity ...")
    enc_sess = ort.InferenceSession(str(enc_onnx), providers=["CPUExecutionProvider"])
    ort_enc = enc_sess.run(["last_hidden_state"], {"pixel_values": dummy_pixels.numpy()})[0]
    enc_diff = float(np.max(np.abs(pt_enc.numpy() - ort_enc)))

    dec_sess = ort.InferenceSession(str(dec_onnx), providers=["CPUExecutionProvider"])
    ort_dec = dec_sess.run(["logits"], {"input_ids": dummy_ids.numpy(), "encoder_hidden_states": ort_enc})[0]
    dec_diff = float(np.max(np.abs(pt_dec.numpy() - ort_dec)))

    elapsed = time.perf_counter() - t0
    print(
        f"[manga_ocr] Exported encoder ({enc_onnx.stat().st_size / (1024 * 1024):.2f} MB) + "
        f"decoder ({dec_onnx.stat().st_size / (1024 * 1024):.2f} MB) + vocab.txt in {elapsed:.1f}s "
        f"| enc_diff={enc_diff:.2e}, dec_diff={dec_diff:.2e}"
    )
    if max(enc_diff, dec_diff) > 1e-2:
        raise RuntimeError(f"Manga-OCR ONNX parity check failed: enc_diff={enc_diff}, dec_diff={dec_diff}")

    return {
        "status": "ok",
        "source_dir": str(snap),
        "output_dir": str(target_dir),
        "encoder_onnx": str(enc_onnx),
        "decoder_onnx": str(dec_onnx),
        "vocab_txt": str(target_dir / "vocab.txt"),
        "max_abs_diff": {"encoder": enc_diff, "decoder": dec_diff},
    }


def copy_paddleocr_vl_gguf(cache_root: Path, output_dir: Path, force: bool = False) -> dict[str, Any]:
    folder = KOHARU_FOLDERS["paddleocr_vl"]
    snap = resolve_koharu_snapshot(cache_root, folder)
    if snap is None:
        return {"status": "missing", "folder": folder, "error": "Koharu snapshot not found"}

    target_dir = output_dir / "paddleocr-vl-gguf"
    target_dir.mkdir(parents=True, exist_ok=True)

    gguf_files = sorted(
        p for p in snap.iterdir() if p.is_file() and p.suffix.lower() == ".gguf" and not p.name.startswith(".")
    )
    if not gguf_files:
        return {"status": "error", "folder": folder, "error": f"No .gguf files found in {snap}"}

    copied: list[dict[str, Any]] = []
    for src in gguf_files:
        dst = target_dir / src.name
        if not force and dst.is_file() and dst.stat().st_size == src.stat().st_size:
            print(f"[paddleocr_vl] Already present: {dst} ({dst.stat().st_size / (1024 * 1024):.1f} MB)")
            copied.append({"file": str(dst), "size_bytes": dst.stat().st_size, "action": "skipped"})
            continue
        t0 = time.perf_counter()
        print(f"[paddleocr_vl] Copying {src.name} ({src.stat().st_size / (1024 * 1024):.1f} MB) -> {dst} ...")
        atomic_copy_file(src, dst)
        print(f"[paddleocr_vl] Copied {src.name} in {time.perf_counter() - t0:.1f}s")
        copied.append({"file": str(dst), "size_bytes": dst.stat().st_size, "action": "copied"})

    hint = {
        "purpose": "Load PaddleOCR-VL-1.6-GGUF in LM Studio (http://localhost:1235/v1) for Vision-OCR mode",
        "files": copied,
        "source_snapshot": str(snap),
    }
    (target_dir / "lmstudio_vision_ocr.json").write_text(json.dumps(hint, indent=2) + "\n", encoding="utf-8")
    return {"status": "ok", "output_dir": str(target_dir), "files": copied}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Koharu local Hugging Face cache (.safetensors/.gguf) into Vellum ONNX/GGUF models."
    )
    parser.add_argument(
        "--koharu-cache",
        type=Path,
        default=Path(os.environ.get("KOHARU_HF_CACHE", str(DEFAULT_KOHARU_CACHE))),
        help="Path to Koharu's local Hugging Face models directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_default_output_dir(),
        help="Target persistent models directory (default: %%LOCALAPPDATA%%\\ComicTranslator\\models).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated subset: rfdetr,lama,manga_ocr,paddleocr_vl (or 'all').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export/overwrite even if target .onnx/.gguf files already exist.",
    )
    args = parser.parse_args()

    cache_root: Path = args.koharu_cache.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        {"rfdetr", "lama", "manga_ocr", "paddleocr_vl"}
        if args.models.strip().lower() == "all"
        else {m.strip().lower() for m in args.models.split(",") if m.strip()}
    )

    print("=== Vellum — Koharu 4-Model Cache Bridge ===")
    print(f"  Koharu cache : {cache_root}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Selected     : {sorted(selected)}")

    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if is_windows_store_python() and local_app_data:
        try:
            output_dir.relative_to(Path(local_app_data).expanduser().resolve())
            print(
                "\n[WARNING] Detected Microsoft Store (WindowsApps) Python writing inside %LOCALAPPDATA%.\n"
                "          Windows MSIX virtualizes writes under %LOCALAPPDATA% into LocalCache\\Local\\...,\n"
                "          which native Win32 C++ executables cannot see directly.\n"
                "          Prefer running `packaging\\convert_koharu_to_onnx.bat` (or pass --output-dir outside\n"
                "          %LOCALAPPDATA% and move into %LOCALAPPDATA%\\ComicTranslator\\models via PowerShell).\n",
                file=sys.stderr,
            )
        except ValueError:
            pass

    if not cache_root.is_dir():
        print(f"ERROR: Koharu cache directory not found: {cache_root}", file=sys.stderr)
        return 1

    results: dict[str, Any] = {}
    had_error = False

    handlers = [
        ("rfdetr", convert_rfdetr),
        ("lama", convert_lama),
        ("manga_ocr", convert_manga_ocr),
        ("paddleocr_vl", copy_paddleocr_vl_gguf),
    ]
    for key, fn in handlers:
        if key not in selected:
            continue
        try:
            res = fn(cache_root, output_dir, force=args.force)
            results[key] = res
            if res.get("status") not in ("ok", "skipped"):
                had_error = True
        except Exception as exc:
            print(f"[{key}] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            results[key] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            had_error = True

    manifest = {
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
        "koharu_cache_dir": str(cache_root),
        "output_dir": str(output_dir),
        "models": results,
    }
    manifest_path = output_dir / "koharu_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest saved to: {manifest_path}")

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
