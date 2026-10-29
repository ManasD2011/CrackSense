from pathlib import Path
import torch
from torch import nn


MODEL_DIR = Path("Quantised")


# ---------------------------------------------------------
# LOAD CHECKPOINT
# ---------------------------------------------------------

def load_model(path):
    print(f"\n{'=' * 70}")
    print(f"Loading: {path}")
    print(f"{'=' * 70}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model = checkpoint["model"]
        precision = checkpoint.get("precision", "unknown")

    elif isinstance(checkpoint, nn.Module):
        model = checkpoint
        precision = "unknown"

    else:
        raise RuntimeError(
            f"Could not find model inside {path}"
        )

    model.eval()

    print(f"Metadata precision : {precision}")
    print(f"Model type         : {type(model).__name__}")

    return model, checkpoint


# ---------------------------------------------------------
# MODEL SIZE
# ---------------------------------------------------------

def get_model_size(path):

    size_mb = path.stat().st_size / (1024 * 1024)

    return size_mb


# ---------------------------------------------------------
# PARAMETER INFORMATION
# ---------------------------------------------------------

def inspect_parameters(model):

    print("\n--- Parameter information ---")

    dtype_counts = {}
    dtype_parameters = {}

    total_parameters = 0

    for name, parameter in model.named_parameters():

        dtype = str(parameter.dtype)

        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        dtype_parameters[dtype] = (
            dtype_parameters.get(dtype, 0)
            + parameter.numel()
        )

        total_parameters += parameter.numel()

    print(f"Total parameters: {total_parameters:,}")

    print("\nParameter dtypes:")

    for dtype in dtype_counts:

        print(
            f"  {dtype:<15} "
            f"tensors={dtype_counts[dtype]:<5} "
            f"parameters={dtype_parameters[dtype]:,}"
        )


# ---------------------------------------------------------
# MODULE INFORMATION
# ---------------------------------------------------------

def inspect_modules(model):

    print("\n--- Module information ---")

    normal_linear = 0
    quantized_linear = 0

    conv2d = 0
    other = 0

    for name, module in model.named_modules():

        module_name = module.__class__.__name__
        module_module = module.__class__.__module__

        if isinstance(module, nn.Linear):

            normal_linear += 1

            print(
                f"[NORMAL LINEAR] "
                f"{name}"
            )

        elif (
            "quantized" in module_module.lower()
            or "quantized" in module_name.lower()
        ):

            quantized_linear += 1

            print(
                f"[QUANTIZED] "
                f"{name} -> "
                f"{module_module}.{module_name}"
            )

        elif isinstance(module, nn.Conv2d):

            conv2d += 1

        else:

            other += 1

    print("\nModule summary:")

    print(f"  Normal Linear layers : {normal_linear}")
    print(f"  Quantized layers     : {quantized_linear}")
    print(f"  Conv2D layers        : {conv2d}")
    print(f"  Other layers         : {other}")


# ---------------------------------------------------------
# STATE DICT
# ---------------------------------------------------------

def inspect_state_dict(model):

    print("\n--- State dictionary ---")

    state_dict = model.state_dict()

    print(f"State dict entries: {len(state_dict)}")

    shown = 0

    for name, value in state_dict.items():

        if torch.is_tensor(value):

            print(
                f"{name:<70} "
                f"shape={str(tuple(value.shape)):<25} "
                f"dtype={value.dtype}"
            )

            shown += 1

        if shown >= 15:

            print("...")

            break


# ---------------------------------------------------------
# TEST FOR NaN / INF
# ---------------------------------------------------------

def check_weights(model):

    print("\n--- Weight integrity check ---")

    bad = False
    checked = 0

    for name, parameter in model.named_parameters():

        if not torch.is_floating_point(parameter):

            continue

        checked += 1

        if torch.isnan(parameter).any():

            print(f"[ERROR] NaN found in: {name}")

            bad = True

        if torch.isinf(parameter).any():

            print(f"[ERROR] INF found in: {name}")

            bad = True

    if not bad:

        print(
            f"PASS: Checked {checked} floating-point tensors. "
            "No NaN/INF detected."
        )

    return not bad


# ---------------------------------------------------------
# FORWARD PASS
# ---------------------------------------------------------

def test_forward(model):

    print("\n--- Forward pass test ---")

    model.eval()

    # SegFormer expects:
    # [batch, channels, height, width]

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    )

    try:

        with torch.inference_mode():

            output = model(
                pixel_values=dummy_input
            )

        print("PASS: Forward pass successful.")

        if hasattr(output, "logits"):

            print(
                f"Output logits shape: "
                f"{tuple(output.logits.shape)}"
            )

            print(
                f"Output dtype: "
                f"{output.logits.dtype}"
            )

        else:

            print(
                f"Output type: "
                f"{type(output)}"
            )

        return True

    except Exception as error:

        print("FAILED: Forward pass failed.")

        print(
            f"Error: {error}"
        )

        return False


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def inspect_checkpoint(filename):

    path = MODEL_DIR / filename

    if not path.exists():

        print(f"\nWARNING: {path} does not exist.")

        return

    model, checkpoint = load_model(path)

    print(
        f"\nFile size: "
        f"{get_model_size(path):.2f} MB"
    )

    inspect_parameters(model)

    inspect_modules(model)

    inspect_state_dict(model)

    check_weights(model)

    test_forward(model)


def main():

    print("\n")
    print("=" * 70)
    print("        CRACKSENSE MODEL VERIFICATION")
    print("=" * 70)

    inspect_checkpoint("latest_fp32.pt")

    inspect_checkpoint("latest_fp16.pt")

    inspect_checkpoint("latest_int8.pt")

    print("\n")
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()