"""Phase 1: trusted SegFormer packages, CUDA FP32 vs FP16, no TTA."""
import argparse
import csv
import gc
import json
import platform
import subprocess
import time
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=Path('Quantised'))
    p.add_argument('--output', type=Path, default=Path('phase1_results'))
    p.add_argument('--frames', type=int, default=100)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--threshold', type=float, help='Override saved best_threshold for BOTH models')
    p.add_argument('--size', type=int, default=256)
    a = p.parse_args()
    if min(a.frames, a.stride, a.warmup, a.repeats, a.size) < 1:
        p.error('frames, stride, warmup, repeats and size must be positive')
    if a.threshold is not None and not 0 <= a.threshold <= 1:
        p.error('threshold must be between 0 and 1')
    if not a.video.is_file():
        p.error('Video file not found')
    for name in ['latest_fp32.pt', 'latest_fp16.pt']:
        if not (a.model_dir / name).is_file():
            p.error(f'Missing model: {a.model_dir / name}')
    return a


def main():
    a = arguments()
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check NVIDIA driver and CUDA-enabled PyTorch installation.')
    a.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    env = dict(python=platform.python_version(), platform=platform.platform(),
               torch=torch.__version__, transformers=transformers.__version__,
               numpy=np.__version__, opencv=cv2.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), gpu_total_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
               arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
               tf32=False, timing_scope='decoded-frame preprocessing through CPU probability/mask; excludes capture, decode, network, display and saving')
    try:
        env['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        env['nvidia_smi'] = 'Unavailable'
    (a.output / 'environment.json').write_text(json.dumps(env, indent=2))
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise RuntimeError('Cannot decode video')
    frames, ids = [], []
    idx = 0
    try:
        while len(frames) < a.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % a.stride == 0:
                # Store only model-resolution input to bound host memory.
                frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (a.size, a.size), interpolation=cv2.INTER_LINEAR))
                ids.append(idx)
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError('Video contains no readable frames')
    print(f'GPU: {env["gpu"]}; using {len(frames)} identical sampled frames at {a.size}x{a.size}')
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    rows, summaries, reference = [], [], []
    threshold = a.threshold
    def sync():
        torch.cuda.synchronize()
    def tensor(frame, dtype):
        # RGB conversion and resize already performed during sampling.
        x = (frame.astype(np.float32) / 255 - mean) / std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0).to('cuda', dtype=dtype)
    def prediction(model, x):
        logits = model(pixel_values=x, return_dict=True).logits
        if logits.shape[1] != 2:
            raise RuntimeError('Expected two-class crack model')
        logits = F.interpolate(logits.float(), (a.size,a.size), mode='bilinear', align_corners=False)
        return torch.sigmoid(logits[:,1] - logits[:,0])[0].cpu().numpy()
    for precision, dtype in [('fp32', torch.float32), ('fp16', torch.float16)]:
        print(f'Loading {precision}...')
        path = a.model_dir / f'latest_{precision}.pt'
        # Full-model pickle: use only the user's trusted local packages.
        package = torch.load(path, map_location='cpu', weights_only=False)
        model = package.get('model') if isinstance(package,dict) else package
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(f'{path} does not contain a packaged torch model')
        if precision == 'fp32' and threshold is None:
            saved = package.get('checkpoint_metadata', {}).get('best_threshold') if isinstance(package,dict) else None
            threshold = float(saved) if saved is not None else .5
            if saved is None:
                print('WARNING: no saved best_threshold; using 0.5, not a validated calibration')
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise RuntimeError('Invalid saved threshold; supply --threshold')
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is not None and "_output_attentions" not in vars(config):
                config.output_attentions = False
        model = model.to(device='cuda', dtype=dtype).eval()
        with torch.inference_mode():
            x = tensor(frames[0], dtype)
            for _ in range(a.warmup):
                prediction(model, x)
            sync()
            torch.cuda.reset_peak_memory_stats()
            start_rows = len(rows)
            for repeat in range(a.repeats):
                for i, frame in enumerate(frames):
                    sync(); t0 = time.perf_counter()
                    x = tensor(frame,dtype)
                    sync(); t1 = time.perf_counter()
                    outputs = model(pixel_values=x,return_dict=True).logits
                    sync(); t2 = time.perf_counter()
                    if outputs.shape[1] != 2:
                        raise RuntimeError('Expected two logits per pixel')
                    logits = F.interpolate(outputs.float(), (a.size,a.size),mode='bilinear',align_corners=False)
                    prob = torch.sigmoid(logits[:,1]-logits[:,0])[0].cpu().numpy()
                    mask = prob > threshold
                    sync(); t3 = time.perf_counter()
                    if not np.isfinite(prob).all():
                        raise RuntimeError(f'Nonfinite prediction: {precision} frame {ids[i]}')
                    row = dict(precision=precision,repeat=repeat,frame_id=ids[i],preprocess_transfer_ms=(t1-t0)*1000,
                               forward_ms=(t2-t1)*1000,postprocess_transfer_ms=(t3-t2)*1000,total_ms=(t3-t0)*1000)
                    if repeat == 0:
                        if precision == 'fp32':
                            reference.append(prob.copy())
                        else:
                            ref = reference[i]; rm = ref > threshold
                            intersection = int((mask & rm).sum()); union = int((mask | rm).sum())
                            denom = int(mask.sum()+rm.sum())
                            row.update(probability_mae=float(np.abs(prob-ref).mean()),
                                       probability_max_error=float(np.abs(prob-ref).max()),
                                       mask_disagreement_fraction=float((mask != rm).mean()),
                                       mask_iou_vs_fp32=intersection/union if union else 1.,
                                       mask_dice_vs_fp32=2*intersection/denom if denom else 1.,
                                       fp32_crack_pixels=int(rm.sum()),fp16_crack_pixels=int(mask.sum()))
                            if i < 12:
                                cv2.imwrite(str(a.output / f'frame_{ids[i]}_fp32_mask.png'),rm.astype(np.uint8)*255)
                                cv2.imwrite(str(a.output / f'frame_{ids[i]}_fp16_mask.png'),mask.astype(np.uint8)*255)
                                cv2.imwrite(str(a.output / f'frame_{ids[i]}_input.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    rows.append(row)
                print(f'{precision}: repeat {repeat+1}/{a.repeats} done')
        measured = rows[start_rows:]
        total = np.array([r['total_ms'] for r in measured])
        forward = np.array([r['forward_ms'] for r in measured])
        summary = dict(precision=precision,threshold=threshold,samples=len(measured),
                       forward_mean_ms=float(forward.mean()),forward_p95_ms=float(np.percentile(forward,95)),
                       total_mean_ms=float(total.mean()),total_p50_ms=float(np.median(total)),
                       total_p95_ms=float(np.percentile(total,95)),processing_fps=1000/float(total.mean()),
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if precision == 'fp16':
            comparisons=[r for r in measured if 'probability_mae' in r]
            for key in ['probability_mae','mask_disagreement_fraction','mask_iou_vs_fp32','mask_dice_vs_fp32']:
                summary['mean_'+key]=float(np.mean([r[key] for r in comparisons]))
        summaries.append(summary)
        print(json.dumps(summary,indent=2))
        del model, package, x, outputs, logits
        gc.collect(); torch.cuda.empty_cache()
    fields = sorted({k for row in rows for k in row})
    with (a.output / 'frame_timings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    report=dict(environment=env,frames_used=len(frames),results=summaries,
                fp16_processing_speedup=summaries[0]['total_mean_ms']/summaries[1]['total_mean_ms'],
                limitations=['Not camera-to-display latency','RGB conversion/resize, decoding, saving and display excluded',
                             'Agreement with FP32 is not ground-truth accuracy','No sustained thermal/endurance test'])
    (a.output / 'summary.json').write_text(json.dumps(report,indent=2))
    print(f'Results saved to {a.output.resolve()}')

if __name__ == '__main__':
    main()
