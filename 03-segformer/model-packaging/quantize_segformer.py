"""Create FP32, FP16, and dynamic-INT8 inference checkpoints from a PyTorch model.

The source checkpoint must contain a serialised ``torch.nn.Module`` either as
the checkpoint itself or under ``model``/``ema``.  This is the usual layout for
training checkpoints that were saved with ``torch.save(model, path)`` or
``torch.save({'model': model, ...}, path)``.

Example:
    python quantize_segformer.py --input checkpoints/best.pt

All outputs are written to Quantised/ unless --output-dir is supplied.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn


def load_checkpoint(path: Path) -> Any:
    """Load a trusted local PyTorch checkpoint with a helpful corruption error."""
    try:
        # weights_only=False is required for full model checkpoints. Only use this
        # with checkpoints you trust, because PyTorch unpickling can execute code.
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(
            f"Could not load '{path}'. It must be a complete, uncorrupted PyTorch "
            "checkpoint. Re-copy or re-download the file if it is truncated or "
            "was split/compressed by another program."
        ) from error


def get_saved_model(checkpoint: Any) -> tuple[nn.Module, str | None] | None:
    """Return a serialised model if one is present in the checkpoint."""
    if isinstance(checkpoint, nn.Module):
        return checkpoint, None

    if isinstance(checkpoint, dict):
        for key in ("model", "ema"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, nn.Module):
                return candidate, key

    return None


def normalize_checkpoint_key(key: str) -> str:
    """Map the training checkpoint names to Hugging Face SegFormer names."""
    key = re.sub(
        r"^segformer\.stages\.(\d+)\.patch_embeddings\.",
        r"segformer.encoder.patch_embeddings.\1.",
        key,
    )
    key = re.sub(
        r"^segformer\.stages\.(\d+)\.blocks\.(\d+)\.",
        r"segformer.encoder.block.\1.\2.",
        key,
    )
    key = re.sub(
        r"^segformer\.stages\.(\d+)\.layer_norm\.",
        r"segformer.encoder.layer_norm.\1.",
        key,
    )
    replacements = {
        ".layernorm_before.": ".layer_norm_1.",
        ".layernorm_after.": ".layer_norm_2.",
        ".attention.q_proj.": ".attention.self.query.",
        ".attention.k_proj.": ".attention.self.key.",
        ".attention.v_proj.": ".attention.self.value.",
        ".attention.o_proj.": ".attention.output.dense.",
        ".attention.sequence_reduction.sequence_reduction.": ".attention.self.sr.",
        ".attention.sequence_reduction.layer_norm.": ".attention.self.layer_norm.",
        ".mlp.fc1.": ".mlp.dense1.",
        ".mlp.fc2.": ".mlp.dense2.",
        "decode_head.linear_projections.": "decode_head.linear_c.",
    }
    for source, target in replacements.items():
        key = key.replace(source, target)
    return key


def build_segformer_from_state_dict(checkpoint: dict[str, Any]) -> nn.Module:
    """Recreate this project's two-class SegFormer MiT-B2 model."""
    if "model_state_dict" not in checkpoint:
        raise TypeError(
            "Checkpoint does not contain a saved model or 'model_state_dict'."
        )

    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as error:
        raise ImportError(
            "This checkpoint needs the Transformers package. Install it with: "
            "pip install transformers"
        ) from error

    model_config = SegformerConfig(
        num_labels=2,
        id2label={0: "background", 1: "crack"},
        label2id={"background": 0, "crack": 1},
        hidden_sizes=[64, 128, 320, 512],
        depths=[3, 4, 6, 3],
        num_attention_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        decoder_hidden_size=768,
    )
    model = SegformerForSemanticSegmentation(model_config)
    state_dict = {
        normalize_checkpoint_key(key): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model


def make_output_payload(checkpoint: Any, model: nn.Module, precision: str) -> dict[str, Any]:
    """Save inference metadata without duplicating weights or optimizer state."""
    metadata = {}
    if isinstance(checkpoint, dict):
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model", "ema", "model_state_dict", "optimizer_state_dict",
                           "scheduler_state_dict", "scaler_state_dict"}
        }
    return {
        "model": model,
        "precision": precision,
        "checkpoint_metadata": metadata,
    }


def clone_for_inference(model: nn.Module) -> nn.Module:
    cloned = copy.deepcopy(model).cpu()
    cloned.eval()
    return cloned


def save_checkpoint(path: Path, payload: Any) -> None:
    torch.save(payload, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a SegFormer/PyTorch checkpoint as FP32, FP16, and INT8."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("checkpoints/latest.pt"),
        help="Source .pt checkpoint (default: checkpoints/latest.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Quantised"),
        help="Directory for generated checkpoints (default: Quantised).",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.input}")

    checkpoint = load_checkpoint(args.input)
    saved_model = get_saved_model(checkpoint)
    source_model = (
        saved_model[0]
        if saved_model is not None
        else build_segformer_from_state_dict(checkpoint)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    fp32_model = clone_for_inference(source_model).float()
    save_checkpoint(
        args.output_dir / f"{stem}_fp32.pt",
        make_output_payload(checkpoint, fp32_model, "fp32"),
    )

    fp16_model = clone_for_inference(source_model).half()
    save_checkpoint(
        args.output_dir / f"{stem}_fp16.pt",
        make_output_payload(checkpoint, fp16_model, "fp16"),
    )

    int8_model = clone_for_inference(source_model).float()
    # SegFormer MiT blocks use Linear layers heavily. Dynamic quantisation stores
    # their weights as INT8 and quantises activations at runtime; Conv2d remains FP32.
    int8_model = torch.ao.quantization.quantize_dynamic(
        int8_model, {nn.Linear}, dtype=torch.qint8, inplace=False
    )
    save_checkpoint(
        args.output_dir / f"{stem}_int8.pt",
        make_output_payload(checkpoint, int8_model, "int8_dynamic_linear"),
    )

    print("Done. These are inference checkpoints; keep the original latest.pt for training.")


if __name__ == "__main__":
    main()
