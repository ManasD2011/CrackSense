from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = Path("test_video.mp4")

MODEL_DIR = Path("Quantised")

FP32_PATH = MODEL_DIR / "latest_fp32.pt"
FP16_PATH = MODEL_DIR / "latest_fp16.pt"
INT8_PATH = MODEL_DIR / "latest_int8.pt"

IMG_SIZE = 224

BEST_THRESHOLD = 0.5

# Start with 30 frames.
# Increase to 100 later for a more stable benchmark.
BENCHMARK_FRAMES = 30

FRAME_SKIP = 1


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("       CRACKSENSE VIDEO QUANTIZATION BENCHMARK")
print("=" * 70)

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "FP32 and FP16 cannot be tested on the RTX 3050."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

GPU = torch.device("cuda:0")
CPU = torch.device("cpu")


# ============================================================
# TRANSFORM
# ============================================================

transform = A.Compose([
    A.Resize(
        IMG_SIZE,
        IMG_SIZE
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),

    ToTensorV2(),
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    path: Path,
    device: torch.device,
):

    print(
        f"\nLoading: {path}"
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        model = checkpoint["model"]

    elif isinstance(
        checkpoint,
        nn.Module
    ):
        model = checkpoint

    else:
        raise RuntimeError(
            f"Could not find model inside {path}"
        )

    model.eval()

    return model


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_frame(
    frame,
    dtype,
    device,
):

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    transformed = transform(
        image=frame_rgb
    )

    tensor = transformed["image"]

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        device=device,
        dtype=dtype,
    )

    return tensor


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    model,
    image_tensor,
):

    with torch.inference_mode():

        output = model(
            pixel_values=image_tensor
        )

        logits = output.logits

        logits = F.interpolate(
            logits,
            size=(
                IMG_SIZE,
                IMG_SIZE
            ),
            mode="bilinear",
            align_corners=False,
        )

        crack_logits = logits[:, 1]

        probability = torch.sigmoid(
            crack_logits
        )

        mask = (
            probability >= BEST_THRESHOLD
        )

    return mask


# ============================================================
# BENCHMARK MODEL
# ============================================================

def benchmark_model(
    name,
    model_path,
    device,
    dtype,
):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    model = load_model(
        model_path,
        device,
    )

    print(
        "Device:",
        device
    )

    print(
        "Input dtype:",
        dtype
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Video FPS: {video_fps:.2f}"
    )

    print(
        f"Total video frames: {total_frames}"
    )

    print(
        f"Benchmark frames: {BENCHMARK_FRAMES}"
    )

    # --------------------------------------------------------
    # Read first frame
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        cap.release()

        raise RuntimeError(
            "Could not read video."
        )

    image_tensor = preprocess_frame(
        frame,
        dtype,
        device,
    )

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    print(
        "\nRunning warmup..."
    )

    if device.type == "cuda":

        for _ in range(10):

            run_inference(
                model,
                image_tensor
            )

        torch.cuda.synchronize()

    else:

        for _ in range(2):

            run_inference(
                model,
                image_tensor
            )

    print(
        "Warmup complete."
    )

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    times = []

    processed = 0
    frame_number = 1

    while (
        processed < BENCHMARK_FRAMES
    ):

        if frame_number > 1:

            ret, frame = cap.read()

            if not ret:
                break

        if (
            frame_number % FRAME_SKIP
            != 0
        ):

            frame_number += 1

            continue

        # Preprocessing is NOT included
        # in the inference timing.
        image_tensor = preprocess_frame(
            frame,
            dtype,
            device,
        )

        # Synchronize GPU before timing
        if device.type == "cuda":

            torch.cuda.synchronize()

        start = time.perf_counter()

        run_inference(
            model,
            image_tensor
        )

        # CUDA operations are asynchronous.
        # Synchronize before stopping timer.
        if device.type == "cuda":

            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_ms = (
            end - start
        ) * 1000

        times.append(
            elapsed_ms
        )

        processed += 1
        frame_number += 1

    cap.release()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    times = np.array(
        times
    )

    mean_ms = float(
        np.mean(times)
    )

    median_ms = float(
        np.median(times)
    )

    min_ms = float(
        np.min(times)
    )

    max_ms = float(
        np.max(times)
    )

    fps = (
        1000.0 / mean_ms
    )

    print("\nRESULTS")

    print(
        f"Processed frames : {processed}"
    )

    print(
        f"Mean inference   : {mean_ms:.2f} ms"
    )

    print(
        f"Median inference : {median_ms:.2f} ms"
    )

    print(
        f"Minimum          : {min_ms:.2f} ms"
    )

    print(
        f"Maximum          : {max_ms:.2f} ms"
    )

    print(
        f"Approx FPS       : {fps:.2f}"
    )

    return {
        "name": name,
        "device": str(device),
        "dtype": str(dtype),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "fps": fps,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    results = []

    # ========================================================
    # 1. FP32 → RTX 3050
    # ========================================================

    results.append(
        benchmark_model(
            name="FP32 GPU",
            model_path=FP32_PATH,
            device=GPU,
            dtype=torch.float32,
        )
    )

    # ========================================================
    # 2. FP16 → RTX 3050
    # ========================================================

    results.append(
        benchmark_model(
            name="FP16 GPU",
            model_path=FP16_PATH,
            device=GPU,
            dtype=torch.float16,
        )
    )

    # ========================================================
    # 3. INT8 → CPU
    # ========================================================

    results.append(
        benchmark_model(
            name="INT8 CPU",
            model_path=INT8_PATH,
            device=CPU,
            dtype=torch.float32,
        )
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("                 FINAL COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Model':<15}"
        f"{'Device':<10}"
        f"{'Mean ms':<14}"
        f"{'Median ms':<14}"
        f"{'FPS':<10}"
    )

    print("-" * 63)

    for result in results:

        print(
            f"{result['name']:<15}"
            f"{result['device']:<10}"
            f"{result['mean_ms']:<14.2f}"
            f"{result['median_ms']:<14.2f}"
            f"{result['fps']:<10.2f}"
        )

    print()

    print("=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
    