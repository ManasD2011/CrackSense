import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "nvidia/mit-b2"
IMG_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from a video, run the trained "
            "SegFormer-B2 crack detector on every extracted "
            "frame, resize the mask back to the original "
            "frame resolution, and save the crack overlay."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video."
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Main output folder. Default: output"
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to trained checkpoint (.pt). "
            "If omitted, common checkpoint locations are searched."
        )
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames to extract and process per second. Default: 2 FPS."
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help=(
            "Extract and process exactly this many evenly spaced frames "
            "from the selected time range. If supplied, --fps is ignored."
        )
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start processing at this timestamp in seconds."
    )

    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Stop processing at this timestamp in seconds."
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process."
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help=(
            "Enable test-time augmentation (5 passes per frame: "
            "original, H-flip, V-flip, scale 1.25x, scale 0.75x, "
            "averaged). ~5x slower per frame. Default: off."
        )
    )

    return parser.parse_args()


# ============================================================
# CHECKPOINT SEARCH
# ============================================================

def find_checkpoint(user_checkpoint=None):
    if user_checkpoint:
        checkpoint = Path(user_checkpoint)

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        return checkpoint

    # Resolve defaults relative to this script, not the folder from which
    # Python happened to be launched.  The script itself lives in B2_Exp,
    # so prepending "B2_Exp" here would incorrectly look for B2_Exp/B2_Exp.
    project_dir = Path(__file__).resolve().parent
    possible_checkpoints = [
        project_dir / "Quantised" / "latest_fp32.pt",
        project_dir / "Quantised" / "latest_fp16.pt",
        project_dir / "Quantised" / "latest_int8.pt",
        project_dir / "checkpoints" / "latest.pt",
        project_dir / "checkpoints" / "best.pt",
    ]

    for checkpoint in possible_checkpoints:
        if checkpoint.exists():
            return checkpoint

    raise FileNotFoundError(
        "Could not find a trained checkpoint.\n"
        "Tried:\n"
        + "\n".join(
            f"  - {path}"
            for path in possible_checkpoints
        )
        + "\n\nUse --checkpoint path/to/model.pt"
    )


# ============================================================
# OUTPUT FOLDER
#
# output/
# ├── output 1/
# │   ├── frame1.jpg
# │   ├── frame2.jpg
# │   └── frame3.jpg
# │
# ├── output 2/
# │   ├── frame1.jpg
# │   └── frame2.jpg
# │
# └── ...
# ============================================================

def get_next_output_folder(output_dir):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    number = 1

    while True:
        candidate = output_dir / f"output {number}"

        if not candidate.exists():
            candidate.mkdir(
                parents=True,
                exist_ok=True
            )

            return candidate

        number += 1


# ============================================================
# MODEL CHECKPOINT KEY NORMALIZATION
# Same conversion used by the inference notebook.
# ============================================================

def normalize_checkpoint_key(key):
    key = re.sub(
        r"^segformer\.stages\.(\d+)\.patch_embeddings\.",
        r"segformer.encoder.patch_embeddings.\1.",
        key
    )

    key = re.sub(
        r"^segformer\.stages\.(\d+)\.blocks\.(\d+)\.",
        r"segformer.encoder.block.\1.\2.",
        key
    )

    key = re.sub(
        r"^segformer\.stages\.(\d+)\.layer_norm\.",
        r"segformer.encoder.layer_norm.\1.",
        key
    )

    key = key.replace(
        ".layernorm_before.",
        ".layer_norm_1."
    )

    key = key.replace(
        ".layernorm_after.",
        ".layer_norm_2."
    )

    key = key.replace(
        ".attention.q_proj.",
        ".attention.self.query."
    )

    key = key.replace(
        ".attention.k_proj.",
        ".attention.self.key."
    )

    key = key.replace(
        ".attention.v_proj.",
        ".attention.self.value."
    )

    key = key.replace(
        ".attention.o_proj.",
        ".attention.output.dense."
    )

    key = key.replace(
        ".attention.sequence_reduction.sequence_reduction.",
        ".attention.self.sr."
    )

    key = key.replace(
        ".attention.sequence_reduction.layer_norm.",
        ".attention.self.layer_norm."
    )

    key = key.replace(
        ".mlp.fc1.",
        ".mlp.dense1."
    )

    key = key.replace(
        ".mlp.fc2.",
        ".mlp.dense2."
    )

    key = key.replace(
        "decode_head.linear_projections.",
        "decode_head.linear_c."
    )

    return key


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(checkpoint_path):
    print("\n" + "=" * 70)
    print("LOADING CRACK DETECTION MODEL")
    print("=" * 70)

    print(f"Model       : {MODEL_NAME}")
    print(f"Input size  : {IMG_SIZE} × {IMG_SIZE}")
    print(f"Device      : {DEVICE}")
    print(f"Checkpoint  : {checkpoint_path}")

    if DEVICE.type == "cuda":
        print(
            f"GPU         : "
            f"{torch.cuda.get_device_properties(0).name}"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    # Training checkpoints contain a state dictionary.  Quantised inference
    # checkpoints created by quantize_segformer.py instead store the ready-to-
    # use model under "model" and preserve training details in
    # "checkpoint_metadata".
    packaged_model = (
        checkpoint.get("model")
        if isinstance(checkpoint, dict)
        else None
    )

    if isinstance(packaged_model, torch.nn.Module):
        model = packaged_model
        checkpoint_info = checkpoint.get("checkpoint_metadata", {})
        print(f"Checkpoint type: packaged {checkpoint.get('precision', 'unknown').upper()} model")
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label={
                0: "background",
                1: "crack"
            },
            label2id={
                "background": 0,
                "crack": 1
            },
            ignore_mismatched_sizes=True,
        )

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint must contain either a serialized 'model' or "
                "a 'model_state_dict'."
            )

        state_dict = {
            normalize_checkpoint_key(key): value
            for key, value in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(state_dict)
        checkpoint_info = checkpoint
        print("Checkpoint type: training state dictionary")

    model.to(DEVICE)
    model.eval()

    best_dice = checkpoint_info.get(
        "best_dice",
        "?"
    )

    best_epoch = checkpoint_info.get(
        "epoch",
        "?"
    )

    saved_config = checkpoint_info.get(
        "config",
        {}
    )

    print("\nCheckpoint information:")
    print(
        f"  Best epoch       : {best_epoch}"
    )
    print(
        f"  Best val Dice    : {best_dice}"
    )
    print(
        f"  Training model   : "
        f"{saved_config.get('MODEL_NAME', MODEL_NAME)}"
    )

    print("=" * 70)

    return model


# ============================================================
# INFERENCE PREPROCESSING
#
# Exactly matches the notebook:
#
# Original frame
#       ↓
# Resize 256 × 256
#       ↓
# ImageNet normalization
#       ↓
# Tensor
# ============================================================

infer_tf = A.Compose([
    A.Resize(
        IMG_SIZE,
        IMG_SIZE
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2(),
])


# ============================================================
# BINARY LOGIT
# ============================================================

def binary_logits_from_two_class(logits):
    """
    For the 2-class model:

        crack logit - background logit

    gives a single binary crack logit.
    """

    return logits[:, 1] - logits[:, 0]


# ============================================================
# RUN MODEL ON ONE FRAME
# ============================================================

def predict_frame(
    frame_bgr,
    model
):
    """
    Run the trained model on ONE video frame.

    The frame itself is never permanently resized.

    The frame is temporarily resized to 256x256 only
    for model inference.

    Returns:
        prob_map: 256x256 crack probability map
    """

    # OpenCV gives BGR.
    # The notebook/model expects RGB.
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Same preprocessing as the notebook
    # --------------------------------------------------------

    tensor_img = infer_tf(
        image=frame_rgb
    )["image"]

    input_tensor = (
        tensor_img
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        with torch.amp.autocast(
            DEVICE.type,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(
                pixel_values=input_tensor,
                return_dict=True
            )

            logits = outputs.logits

    # --------------------------------------------------------
    # Upsample decoder output to 256x256
    # --------------------------------------------------------

    logits_full = F.interpolate(
        logits,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )

    # --------------------------------------------------------
    # Convert two-class logits to binary crack logit
    # --------------------------------------------------------

    binary_logit = (
        binary_logits_from_two_class(
            logits_full.float()
        )
    )

    # --------------------------------------------------------
    # Convert logit to probability
    # --------------------------------------------------------

    prob_map = (
        torch.sigmoid(
            binary_logit
        )[0]
        .cpu()
        .numpy()
    )

    return prob_map


# ============================================================
# RUN MODEL ON ONE FRAME — WITH TEST-TIME AUGMENTATION (TTA)
#
# Same 5 passes as the notebook's tta_inference():
# Original, H-Flip, V-Flip, Scale 1.25x, Scale 0.75x — averaged.
# ============================================================

def predict_frame_tta(
    frame_bgr,
    model,
    img_size=IMG_SIZE,
    device=DEVICE
):
    """
    Run TTA (5 augmented passes) on ONE video frame.

    Returns:
        prob_map: 256x256 crack probability map, averaged
                  across all 5 passes.
    """

    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    def preprocess(image):
        t = infer_tf(image=image)["image"]
        return t.unsqueeze(0).to(device)

    def predict(tensor):
        with torch.no_grad():
            with torch.amp.autocast(
                device.type,
                enabled=(device.type == "cuda")
            ):
                logits = model(
                    pixel_values=tensor,
                    return_dict=True
                ).logits

                logits_up = F.interpolate(
                    logits,
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=False
                )

                binary_logit = binary_logits_from_two_class(
                    logits_up.float()
                )

                return torch.sigmoid(binary_logit)[0].cpu().numpy()

    def scale_up(img):
        h, w = img.shape[:2]
        new_h, new_w = int(h * 1.25), int(w * 1.25)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        return scaled[y0:y0 + h, x0:x0 + w]

    def scale_down(img):
        h, w = img.shape[:2]
        new_h, new_w = int(h * 0.75), int(w * 0.75)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_top = (h - new_h) // 2
        pad_left = (w - new_w) // 2
        padded = np.zeros_like(img)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = scaled
        return padded

    # (name, forward transform on image, reverse transform on prob map)
    tta_configs = [
        ("Original", lambda img: img, lambda p: p),
        ("H-Flip", lambda img: np.fliplr(img).copy(), lambda p: np.fliplr(p).copy()),
        ("V-Flip", lambda img: np.flipud(img).copy(), lambda p: np.flipud(p).copy()),
        ("Scale 1.25x", scale_up, lambda p: p),
        ("Scale 0.75x", scale_down, lambda p: p),
    ]

    prob_maps = []

    for _, forward_fn, reverse_fn in tta_configs:
        aug_img = forward_fn(frame_rgb)
        tensor = preprocess(aug_img)
        prob = predict(tensor)
        prob_maps.append(reverse_fn(prob))

    return np.mean(prob_maps, axis=0)


# ============================================================
# CREATE MASK
# ============================================================

def create_mask(prob_map):
    """
    Create a 256x256 binary mask using Otsu thresholding.

    The threshold is calculated separately for every frame from that frame's
    model probability map.
    """
    prob_u8 = (
        np.clip(
            prob_map,
            0.0,
            1.0
        ) * 255
    ).astype(np.uint8)

    otsu_value, _ = cv2.threshold(
        prob_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    threshold = float(otsu_value) / 255.0

    mask = (
        prob_map > threshold
    ).astype(np.uint8)

    return mask, threshold


# ============================================================
# RESIZE MASK TO ORIGINAL FRAME
# ============================================================

def resize_mask_to_original(
    mask,
    original_height,
    original_width
):
    """
    Convert 256x256 binary mask back to the
    exact original video-frame resolution.

    IMPORTANT:
    cv2.resize expects (width, height).
    """

    return cv2.resize(
        mask,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )


# ============================================================
# CREATE CRACK OVERLAY
# ============================================================

def create_overlay(
    frame_bgr,
    mask_full
):
    """
    Apply the full-resolution crack mask to
    the original frame.

    Crack pixels are highlighted in red.

    The original frame resolution is preserved.
    """

    overlay = frame_bgr.copy()

    # BGR red
    overlay[
        mask_full == 1
    ] = [0, 0, 255]

    # Blend original and highlighted mask
    result = cv2.addWeighted(
        frame_bgr,
        0.60,
        overlay,
        0.40,
        0
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    video_path = Path(
        args.video
    )

    output_dir = Path(
        args.output
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if args.fps <= 0:

        raise ValueError(
            "--fps must be greater than 0."
        )

    if args.num_frames is not None and args.num_frames <= 0:

        raise ValueError(
            "--num-frames must be greater than 0."
        )

    if args.start < 0:

        raise ValueError(
            "--start cannot be negative."
        )

    if (
        args.end is not None
        and args.end <= 0
    ):

        raise ValueError(
            "--end must be greater than 0."
        )

    if (
        args.end is not None
        and args.start >= args.end
    ):

        raise ValueError(
            "--start must be smaller than --end."
        )

    if (
        args.max_frames is not None
        and args.max_frames <= 0
    ):

        raise ValueError(
            "--max-frames must be greater than 0."
        )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = find_checkpoint(
        args.checkpoint
    )

    # --------------------------------------------------------
    # LOAD MODEL ONCE
    #
    # The model is NOT loaded again for every frame.
    # It is loaded once and then called for every frame.
    # --------------------------------------------------------

    model = load_model(checkpoint_path)

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    source_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    source_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if source_fps <= 0:

        cap.release()

        raise RuntimeError(
            "Could not determine video FPS."
        )

    duration = (
        frame_count / source_fps
    )

    # --------------------------------------------------------
    # TIME RANGE
    # --------------------------------------------------------

    start_time = max(
        0.0,
        args.start
    )

    if args.end is None:

        end_time = duration

    else:

        end_time = min(
            args.end,
            duration
        )

    if start_time >= end_time:

        cap.release()

        raise ValueError(
            "Start time must be smaller than end time."
        )

    # --------------------------------------------------------
    # FRAME SAMPLING
    # --------------------------------------------------------
    # If --num-frames is provided, select exactly that many source
    # frame indices evenly across the requested time range. This is
    # deterministic and avoids OpenCV timestamp sampling issues.
    # Otherwise, sample deterministically by source frame index.

    start_frame = int(round(start_time * source_fps))
    end_frame_exclusive = int(round(end_time * source_fps))
    end_frame_exclusive = min(end_frame_exclusive, frame_count)

    available_frames = max(0, end_frame_exclusive - start_frame)

    if available_frames <= 0:
        cap.release()
        raise ValueError("No video frames are available in the selected time range.")

    if args.num_frames is not None:
        frames_to_extract = min(args.num_frames, available_frames)
        target_indices = np.linspace(
            start_frame,
            end_frame_exclusive - 1,
            frames_to_extract,
            dtype=np.int64
        )
        target_indices = list(dict.fromkeys(int(x) for x in target_indices))
        sampling_mode = "EXACT FRAME COUNT"
        effective_fps = (
            len(target_indices) / max(end_time - start_time, 1.0 / source_fps)
        )
    else:
        frame_step = max(1, int(round(source_fps / args.fps)))
        target_indices = list(range(start_frame, end_frame_exclusive, frame_step))
        sampling_mode = "FIXED FPS"
        effective_fps = args.fps

    target_set = set(target_indices)

    # --------------------------------------------------------
    # CREATE OUTPUT FOLDER
    #
    # Example:
    #
    # output/
    #   output 1/
    #       frame1.jpg        # crack overlay
    #       frame1_mask.png   # binary crack mask
    #       frame2.jpg
    #       frame2_mask.png
    #       frame3.jpg
    #       frame3_mask.png
    #
    #   output 2/
    #       frame1.jpg
    #       frame2.jpg
    # --------------------------------------------------------

    video_output_dir = (
        get_next_output_folder(
            output_dir
        )
    )

    # --------------------------------------------------------
    # PRINT INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VIDEO CRACK DETECTION")
    print("=" * 70)

    print(
        f"Input video       : {video_path}"
    )

    print(
        f"Video resolution  : "
        f"H={source_height}, W={source_width}"
    )

    print(
        f"Original FPS      : "
        f"{source_fps:.2f}"
    )

    print(
        f"Duration          : "
        f"{duration:.2f} sec"
    )

    if args.num_frames is not None:
        print(f"Requested frames  : {args.num_frames}")
        print(f"Frames selected   : {len(target_indices)}")
        print(f"Sampling mode     : {sampling_mode}")
        print(f"Effective FPS     : {effective_fps:.2f}")
    else:
        print(f"Processing FPS    : {args.fps:.2f}")
        print(f"Frame step        : {frame_step} source frames")
        print(f"Sampling mode     : {sampling_mode}")

    print(
        f"Model input       : "
        f"H={IMG_SIZE}, W={IMG_SIZE}"
    )

    print("Threshold method  : Otsu (calculated per frame)")

    print(
        f"TTA               : "
        f"{'ON (5 passes/frame)' if args.tta else 'off'}"
    )

    print(
        f"Output directory  : "
        f"{video_output_dir}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # MOVE TO FIRST TARGET FRAME
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        target_indices[0]
    )

    saved = 0
    target_position = 0

    # --------------------------------------------------------
    # MAIN FRAME LOOP
    # --------------------------------------------------------

    while target_position < len(target_indices):

        target_frame_index = target_indices[target_position]

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)

        ret, frame = cap.read()

        if not ret:
            print(f"\nWARNING: Could not read source frame {target_frame_index}.")
            target_position += 1
            continue

        source_frame_index = target_frame_index
        timestamp_sec = source_frame_index / source_fps

        target_position += 1

        # ----------------------------------------------------
        # ORIGINAL FRAME RESOLUTION
        # ----------------------------------------------------

        original_h, original_w = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # 1. CALL THE MODEL
        #
        # The model receives a temporary 256x256 version,
        # but the original frame is never permanently resized.
        # ----------------------------------------------------

        if args.tta:
            prob_map = predict_frame_tta(
                frame,
                model
            )
        else:
            prob_map = predict_frame(
                frame,
                model
            )

        # ----------------------------------------------------
        # 2. THRESHOLD THE MODEL OUTPUT
        # ----------------------------------------------------

        mask_256, used_threshold = create_mask(prob_map)

        # ----------------------------------------------------
        # 3. RESIZE MASK TO ORIGINAL FRAME RESOLUTION
        # ----------------------------------------------------

        mask_full = resize_mask_to_original(
            mask_256,
            original_h,
            original_w
        )

        # ----------------------------------------------------
        # 4. APPLY CRACK MASK TO ORIGINAL FRAME
        # ----------------------------------------------------

        result = create_overlay(
            frame,
            mask_full
        )

        # ----------------------------------------------------
        # 5. SAVE RESULT
        #
        # frame1.jpg
        # frame2.jpg
        # frame3.jpg
        # ...
        # ----------------------------------------------------

        filename = (
            f"frame{saved + 1}.jpg"
        )

        output_path = (
            video_output_dir
            / filename
        )

        mask_filename = f"frame{saved + 1}_mask.png"
        mask_path = video_output_dir / mask_filename

        success = cv2.imwrite(
            str(output_path),
            result,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        mask_success = cv2.imwrite(
            str(mask_path),
            (mask_full * 255).astype(np.uint8),
            [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )

        if not success or not mask_success:

            if not success:
                print(
                    f"\nWARNING: Could not save overlay {output_path}"
                )

            if not mask_success:
                print(
                    f"\nWARNING: Could not save mask {mask_path}"
                )

            continue

        saved += 1

        # ----------------------------------------------------
        # CRACK PIXEL PERCENTAGE
        # ----------------------------------------------------

        crack_percentage = (
            100.0
            * mask_full.sum()
            / mask_full.size
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        print(
            f"Processed frame {saved:04d} | "
            f"time={timestamp_sec:7.2f}s | "
            f"source_frame={source_frame_index:6d} | "
            f"threshold={used_threshold:.4f} | "
            f"crack={crack_percentage:6.2f}%"
        )

        # ----------------------------------------------------
        # MAXIMUM FRAME LIMIT
        # ----------------------------------------------------

        if (
            args.max_frames is not None
            and saved >= args.max_frames
        ):

            break

    # --------------------------------------------------------
    # RELEASE VIDEO
    # --------------------------------------------------------

    cap.release()

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VIDEO CRACK DETECTION COMPLETE")
    print("=" * 70)

    print(
        f"Frames processed : {saved}"
    )

    print(
        f"Output folder    : {video_output_dir}"
    )

    print(
        f"Output resolution: "
        f"H={source_height}, W={source_width}"
    )

    print(
        "\nSaved structure:"
    )

    print(
        f"  {video_output_dir}/"
    )

    for i in range(min(saved, 3)):
        print(f"      frame{i + 1}.jpg")
        print(f"      frame{i + 1}_mask.png")

    if saved > 3:
        print(f"      ... frame{saved}.jpg")
        print(f"      ... frame{saved}_mask.png")

    print("=" * 70)


if __name__ == "__main__":
    main()
