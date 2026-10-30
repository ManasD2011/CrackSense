"""Phase 3 bench receiver. Requires the trusted packaged FP16 SegFormer model."""
import argparse
import csv
import json
import os
import threading
import time
from pathlib import Path


class Recording:
    """One writer per worker; timestamps in CSV describe actual frame timing."""
    def __init__(self, directory, name, fps, cv2):
        self.directory, self.name, self.fps, self.cv2 = directory, name, fps, cv2
        self.writer = None
        self.size = None
        self.count = 0
        self.part = 0
        self.file = (directory / f"{name}_frames.csv").open("w", newline="", encoding="utf-8")
        self.csv = csv.writer(self.file)
        self.csv.writerow(["video", "video_frame_index", "generation", "receiver_frame_id",
                           "receive_perf_counter_s", "record_perf_counter_s"])

    def write(self, frame, generation, sequence, received):
        size = (frame.shape[1], frame.shape[0])
        if self.writer is None or size != self.size:
            if self.writer is not None:
                self.writer.release()
            self.part += 1
            self.filename = f"{self.name}_{self.part:03d}.avi"
            self.writer = self.cv2.VideoWriter(str(self.directory / self.filename),
                self.cv2.VideoWriter_fourcc(*"MJPG"), self.fps, size)
            if not self.writer.isOpened():
                raise IOError(f"Cannot open video writer: {self.filename}")
            self.size, self.part_index = size, 0
        self.writer.write(frame)
        self.csv.writerow([self.filename, self.part_index, generation, sequence,
                           received, time.perf_counter()])
        self.part_index += 1
        self.count += 1
        if self.count % 30 == 0:
            self.file.flush()

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.file.close()


class Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.packet = None
        self.generation = 0
        self.sequence = 0
        self.state = "CONNECTING"

    def reset(self, state):
        with self.lock:
            self.generation += 1
            self.packet = None
            self.state = state

    def publish(self, frame):
        with self.lock:
            self.sequence += 1
            self.packet = (self.generation, self.sequence, time.perf_counter(), frame)
            self.state = "LIVE"
            return self.packet

    def snapshot(self):
        with self.lock:
            return self.packet, self.generation, self.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--checkpoint", default="Quantised/latest_fp16.pt")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", default="phase3_results")
    parser.add_argument("--stale-seconds", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 means until Q")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--record-fps", type=float, default=30.0,
                        help="Fixed playback FPS for AVI files; CSV holds actual timing")
    parser.add_argument("--no-record", action="store_true", help="Only previews and metrics")
    args = parser.parse_args()
    if args.stale_seconds <= 0 or args.duration < 0 or not 0 < args.record_fps < float("inf"):
        parser.error("Invalid stale timeout or duration")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("Threshold must be between 0 and 1")

    # Set before importing cv2; timeouts also bound blocking open/read calls.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
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
        model = model.to(device="cuda", dtype=torch.float16).eval()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def predict(frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
            tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
            tensor = tensor.unsqueeze(0).to(device="cuda", dtype=torch.float16)
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

        print("Warming up FP16 model...")
        for _ in range(10):
            predict(np.zeros((720, 1280, 3), dtype=np.uint8))
        torch.cuda.synchronize()

    session = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    (session / "config.json").write_text(json.dumps({
        **vars(args), "used_threshold": threshold,
        "probability": "sigmoid(crack_logit-background_logit)",
        "timing": "decode-delivery to matched-view submission; excludes upstream delay",
        "recording": "raw: all decoded frames; processed: unique displayed results; fixed FPS playback",
        "opencv": cv2.__version__,
        "torch": None if model is None else torch.__version__,
        "gpu": None if model is None else torch.cuda.get_device_name(0)
    }, indent=2), encoding="utf-8")
    latest = Latest()
    stop = threading.Event()
    result_lock = threading.Lock()
    result = None
    failures = []
    raw_recording = None if args.no_record else Recording(session, "raw", args.record_fps, cv2)
    processed_recording = None if args.no_record or model is None else Recording(session, "processed", args.record_fps, cv2)
    completed = 0

    def receive():
        while not stop.is_set():
            cap = None
            latest.reset("CONNECTING")
            try:
                cap = cv2.VideoCapture(args.stream, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000])
                if not cap.isOpened():
                    latest.reset("DISCONNECTED")
                else:
                    while not stop.is_set():
                        ok, frame = cap.read()
                        if not ok:
                            latest.reset("DISCONNECTED")
                            break
                        published = latest.publish(frame)
                        if raw_recording is not None:
                            raw_recording.write(frame, published[0], published[1], published[2])
            except Exception as exc:
                latest.reset("DISCONNECTED")
                print("Receiver:", exc)
                if isinstance(exc, IOError):
                    failures.append(f"Recording: {exc}")
                    stop.set()
            finally:
                if cap is not None:
                    cap.release()
            stop.wait(0.5)

    def infer():
        nonlocal result, completed
        last_id = 0
        last_generation = -1
        try:
            while not stop.is_set():
                packet, generation, _ = latest.snapshot()
                if packet is None or packet[1] == last_id:
                    stop.wait(0.002)
                    continue
                gen, seq, decoded, frame = packet
                if time.perf_counter() - decoded > args.stale_seconds:
                    stop.wait(0.01)
                    continue
                skipped = max(0, seq - last_id - 1) if gen == last_generation else 0
                last_id, last_generation = seq, gen
                begin = time.perf_counter()
                mask = predict(frame)
                done = time.perf_counter()
                completed += 1
                if latest.snapshot()[1] != gen:
                    continue
                with result_lock:
                    result = (gen, seq, decoded, frame, mask, begin, done, skipped)
        except Exception as exc:
            failures.append(str(exc))
            stop.set()

    threads = [threading.Thread(target=receive, daemon=True)]
    if model is not None:
        threads.append(threading.Thread(target=infer, daemon=True))
    for thread in threads:
        thread.start()
    started = rate_start = time.perf_counter()
    received_before = displayed_before = displayed = 0
    received_fps = inference_fps = 0.0
    shown_id = -1
    inference_ms = []
    local_age_ms = []
    skipped_total = 0
    print("Q: quit | S: save current matched frame/mask/overlay")
    print("Local timing is NOT camera-to-result latency.")
    try:
        with (session / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["generation", "receiver_frame_id", "session_s", "processing_ms",
                             "decode_to_view_ms", "skipped_waiting_frames", "received_fps", "displayed_result_fps", "receive_perf_counter_s", "inference_start_perf_counter_s",
                             "result_ready_perf_counter_s", "queue_wait_ms", "decode_to_result_ms"])
            while not stop.is_set():
                now = time.perf_counter()
                packet, generation, state = latest.snapshot()
                if packet is not None and now - packet[2] > args.stale_seconds:
                    state = "STALE"
                if now - rate_start >= 1:
                    elapsed = now - rate_start
                    received_fps = (latest.sequence - received_before) / elapsed
                    inference_fps = (displayed - displayed_before) / elapsed
                    received_before, displayed_before, rate_start = latest.sequence, displayed, now
                raw = np.zeros((480, 640, 3), dtype=np.uint8) if packet is None else packet[3].copy()
                cv2.putText(raw, f"{state} | Received {received_fps:.1f} FPS", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Raw preview", raw)
                with result_lock:
                    current = result
                overlay = None
                valid = current is not None and current[0] == generation and state == "LIVE"
                if valid:
                    gen, seq, decoded, frame, mask, begin, done, skipped = current
                    valid = now - decoded <= args.stale_seconds
                if valid:
                    overlay = frame.copy()
                    pixels = mask.astype(bool)
                    overlay[pixels] = (0.55 * overlay[pixels] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
                    process_ms = (done - begin) * 1000
                    new_result = seq != shown_id
                    if new_result:
                        age = (time.perf_counter() - decoded) * 1000
                        writer.writerow([gen, seq, now - started, process_ms, age, skipped,
                                         received_fps, inference_fps, decoded, begin, done,
                                         (begin - decoded) * 1000, (done - decoded) * 1000])
                        file.flush()
                        shown_id = seq
                        displayed += 1
                        inference_ms.append(process_ms)
                        local_age_ms.append(age)
                        skipped_total += skipped
                    cv2.putText(overlay, f"FP16 | frame {seq} | {process_ms:.1f} ms | {inference_fps:.1f} FPS",
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if new_result and processed_recording is not None:
                        processed_recording.write(overlay, gen, seq, decoded)
                    cv2.imshow("Matched segmentation", overlay)
                elif model is not None:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "STALE / WAITING FOR RESULT", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    cv2.imshow("Matched segmentation", blank)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or (args.duration and now - started >= args.duration):
                    break
                if key == ord("s") and valid and overlay is not None:
                    for name, image in (("original", frame), ("mask", mask * 255), ("overlay", overlay)):
                        path = session / f"g{gen}_frame{seq}_{name}.png"
                        if not cv2.imwrite(str(path), image):
                            raise IOError(f"Could not save {path}")
                    print(f"Saved matched frame {seq}")
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C")
    except Exception as exc:
        failures.append(f"Main loop: {exc}")
        raise
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=6)
        # Capture open/read calls have bounded timeouts. Wait for writer owner to finish.
        for thread in threads:
            if thread.is_alive():
                thread.join()
        for recording in (raw_recording, processed_recording):
            if recording is not None:
                recording.close()
        cv2.destroyAllWindows()
        def stats(values):
            return {} if not values else {"count": len(values), "mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
                                          "p95_ms": float(np.percentile(values, 95)), "max_ms": float(max(values))}
        summary = {"processing": stats(inference_ms), "decode_to_view": stats(local_age_ms),
                   "skipped_waiting_frames": skipped_total, "errors": failures,
                   "full_camera_latency_measured": False,
                   "elapsed_s": time.perf_counter() - started,
                   "received_frames": latest.sequence, "completed_inferences": completed,
                   "displayed_results": displayed,
                   "raw_recorded_frames": 0 if raw_recording is None else raw_recording.count,
                   "processed_recorded_frames": 0 if processed_recording is None else processed_recording.count,
                   "average_received_fps": latest.sequence / max(time.perf_counter() - started, 1e-9),
                   "average_completed_inference_fps": completed / max(time.perf_counter() - started, 1e-9)}
        (session / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Results:", session)
    if failures:
        raise RuntimeError(failures[0])


if __name__ == "__main__":
    main()
