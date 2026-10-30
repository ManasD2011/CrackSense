"""Model pipeline preserved from Phase 3; FP32 is an explicit runtime option."""

def load_predictor(args):
    import cv2
    import numpy as np
    model = None
    threshold = args.threshold
    if not args.preview_only:
        import torch
        import torch.nn.functional as F
        # Import required when unpickling the user's Transformers model package.
        from transformers import SegformerForSemanticSegmentation  # noqa: F401
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Use the working Phase 1 environment.")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("Expected packaged model under 'model'; use latest_fp16.pt.")
        precision = str(checkpoint.get("precision", "")) if isinstance(checkpoint, dict) else ""
        if "int8" in precision.lower() or "quantized" in str(type(model)).lower():
            raise RuntimeError("Use the floating-point FP16 package, not dynamic INT8.")
        metadata = checkpoint.get("checkpoint_metadata", {}) if isinstance(checkpoint, dict) else {}
        if threshold is None:
            for item in (metadata, checkpoint if isinstance(checkpoint, dict) else {}):
                if isinstance(item, dict) and item.get("best_threshold") is not None:
                    threshold = float(item["best_threshold"])
                    break
        if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise RuntimeError("No valid best_threshold found. Supply --threshold using your validated value.")
        dtype = torch.float16 if args.precision == "fp16" else torch.float32
        model = model.to(device="cuda", dtype=dtype).eval()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def predict(frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
            tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
            tensor = tensor.unsqueeze(0).to(device="cuda", dtype=dtype)
            with torch.inference_mode():
                logits = model(
                                pixel_values=tensor,
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict=True,
                            ).logits
                if logits.shape[1] != 2:
                    raise RuntimeError("Expected background/crack two-class logits")
                logits = F.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=False)
                probability = torch.sigmoid(logits[:, 1].float() - logits[:, 0].float())[0]
                mask = (probability > threshold).cpu().numpy().astype(np.uint8)
            return cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        print("Warming up model...")
        for _ in range(10):
            predict(np.zeros((720, 1280, 3), dtype=np.uint8))
        torch.cuda.synchronize()

    return (None if args.preview_only else predict), threshold
