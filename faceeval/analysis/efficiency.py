"""
faceeval.analysis.efficiency
============================
Standalone runtime and memory measurement for the deep face-embedding models (FaceNet, ArcFace, ...).

Why this exists
---------------
``faceeval.evaluation.profiler.ModelProfiler`` measures memory with ``tracemalloc``, which only sees Python-heap
allocations. PyTorch and ONNX Runtime allocate native memory that ``tracemalloc`` does not count, so it badly
under-reports a deep model. This module measures the PROCESS peak resident set size (``ru_maxrss``) and runs each
model in its OWN subprocess (a peak is process-wide and never decreases, so models must not share a process).

Protocol (per model, on the images you point it at; every phase is reported separately)
---------------------------------------------------------------------------------------
1. initialisation   ``load_s``: wall time of backbone loading (``fit_embedding``). ``rss_before_load_mb`` /
                    ``rss_after_load_mb`` / ``rss_peak_after_load_mb`` bracket it, so the load peak is visible on its own.
2. cold calls       ``first_preprocess_ms`` and ``first_embed_ms``: the very first call of each, before any warm-up.
3. warm-up          ``--warmup`` untimed calls of preprocessing, of single-image embedding and of every batch size.
4. preprocessing    ``preprocess_ms_{mean,p50,p95}`` per image (``pipeline_for_model(...).process_path``), after warm-up.
5. latency          ``embed_ms_{mean,p50,p95}``: per-image embedding at batch size 1, after warm-up.
6. throughput       ``batched``: for each ``--batch-sizes`` value B, ``images_per_s`` and ``ms_per_image`` of
                    ``extract_features([B images])`` after warm-up. The harness calls the model's own list interface; a
                    model that loops internally (ONNX Runtime models, dlib) will show no gain over batch size 1, which is
                    the honest answer about that implementation, not a harness defect.
7. memory           ``rss_peak_mb``: PROCESS peak resident set size over the whole run (``ru_maxrss``; native memory
                    included). GPU memory only where it is genuinely measurable, see below.

Devices (``--device``)
----------------------
``cpu``; ``cuda``; ``mps`` (Apple Silicon); ``auto`` (exactly what the experiments use: CUDA when torch sees one, else
CPU, never MPS); ``best`` (CUDA, else MPS, else CPU). Selection logic is the pure function :func:`select_device`.
An explicitly requested device that is not available is REFUSED, never silently replaced. Only torch-device models
(FaceNet) can be moved; ONNX Runtime models (ArcFace, InsightFace) and dlib are asked for ``cpu``/``auto`` only, and
their device is recorded (``device_used``, ``device_detail``). Every row records ``device_kind``; the JSON carries a
``comparability`` block listing the conditions that differ between rows. Do not compare latency across rows whose
conditions differ.

GPU memory
----------
* CUDA: ``gpu_peak_allocated_mb`` = PyTorch ``max_memory_allocated`` over the timed embedding calls (peak; excludes the
  CUDA context and non-PyTorch allocations such as ONNX Runtime).
* MPS: PyTorch exposes no peak counter, so only ``gpu_current_allocated_mb_after`` (current allocation after the timed
  calls, ``torch.mps.current_allocated_memory``) is recorded and it is NOT a peak.
* Anything else: NOT_RECORDED. Accelerator calls are asynchronous, so every timed call is followed by a device
  synchronise (``torch.cuda.synchronize`` / ``torch.mps.synchronize``).

Provenance: platform, machine, processor, CPU count, Python, NumPy, PyTorch version, CUDA/MPS availability, GPU name.
Anything that cannot be determined is the string ``NOT_RECORDED``.

Limits (read before citing a number)
------------------------------------
* The numbers describe ONE machine, ONE run, and the images given; no confidence interval is computed. Repeat the
  measurement and report the spread yourself.
* A model whose backbone is not a genuine pretrained one (stub fallback) is REFUSED unless ``--allow-stub`` is given;
  stub timings are meaningless.
* Only models that need no training data are supported (the frozen deep backbones). Classical models are not covered.
* The ``synthetic`` pseudo-model exists only to self-test this harness (a matrix product on random arrays, not a face
  model). Its results carry ``"synthetic": true`` and must never be reported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np

NOT_RECORDED = "NOT_RECORDED"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SYNTHETIC = "synthetic"
DEFAULT_BATCH_SIZES = (8, 32)
TORCH_DEVICE_MODELS = ("facenet",)           # models whose device can be chosen (torch); the others are ONNX Runtime / dlib
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps", "best")


class EfficiencyError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------ device selection
class DeviceChoice(NamedTuple):
    name: str          # "cpu", "cuda", "cuda:1" or "mps"; "auto" is passed to the model unchanged (its own resolution)
    reason: str


def select_device(requested: str, *, cuda_available: bool, mps_available: bool) -> DeviceChoice:
    """Pure device-selection logic (no hardware access, so it is testable anywhere).

    ``auto`` reproduces the experiments' rule exactly (CUDA if available, else CPU; MPS is never chosen implicitly).
    ``best`` is CUDA, else MPS, else CPU. ``cpu`` always works. ``cuda`` / ``cuda:N`` / ``mps`` must be available or the
    request is refused (no silent fallback: a measurement taken on another device would be mislabelled)."""
    r = (requested or "auto").strip().lower()
    if r == "auto":
        return DeviceChoice("cuda" if cuda_available else "cpu", "auto: CUDA if available else CPU (MPS is never implicit)")
    if r == "best":
        if cuda_available:
            return DeviceChoice("cuda", "best: CUDA available")
        if mps_available:
            return DeviceChoice("mps", "best: MPS available, no CUDA")
        return DeviceChoice("cpu", "best: no accelerator available")
    if r == "cpu":
        return DeviceChoice("cpu", "requested cpu")
    if r == "cuda" or (r.startswith("cuda:") and r[5:].isdigit()):
        if not cuda_available:
            raise EfficiencyError(f"--device {requested}: CUDA is not available on this machine")
        return DeviceChoice(r, f"requested {r}")
    if r == "mps":
        if not mps_available:
            raise EfficiencyError("--device mps: the MPS backend is not available on this machine")
        return DeviceChoice("mps", "requested mps")
    raise EfficiencyError(f"unknown --device {requested!r} (choose from {', '.join(DEVICE_CHOICES)} or cuda:N)")


def detect_accelerators(torch_module: Any = None) -> tuple[bool, bool]:
    """(cuda_available, mps_available) as torch reports them; (False, False) if torch is missing or errors."""
    try:
        torch = torch_module if torch_module is not None else __import__("torch")
        cuda = bool(torch.cuda.is_available())
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps = bool(backend is not None and backend.is_available() and backend.is_built())
        return cuda, mps
    except Exception:
        return False, False


def device_kind(label: Any) -> str:
    """'cuda' | 'mps' | 'cpu' | 'other' from a device label."""
    t = str(label or "").lower()
    if t.startswith("cuda"):
        return "cuda"
    if t.startswith("mps"):
        return "mps"
    if t.startswith("cpu"):
        return "cpu"
    return "other"


def device_label(dev: Any) -> str:
    """Label of the device a model really ran on. Only torch CUDA / MPS devices get accelerator accounting."""
    if isinstance(dev, str) and (dev.startswith("cuda") or dev == "mps"):
        return dev
    if dev == "cpu":
        return "cpu"
    return f"{dev or 'unknown'} (not a torch device; accelerator memory is NOT measured)"


class DeviceHooks(NamedTuple):
    sync: Callable[[], None]
    reset_peak: Callable[[], None]
    peak_mb: Callable[[], float | str]            # CUDA only
    current_mb: Callable[[], float | str]         # MPS only (no peak counter exists)


def device_hooks(kind: str, torch_module: Any = None) -> DeviceHooks:
    """Synchronise / memory hooks for a device kind. Without the matching torch backend every hook is a no-op or
    NOT_RECORDED. ``torch_module`` can be a stand-in object (tests)."""
    noop = (lambda: None)
    nr = (lambda: NOT_RECORDED)
    try:
        torch = torch_module if torch_module is not None else __import__("torch")
        if kind == "cuda":
            return DeviceHooks(torch.cuda.synchronize, torch.cuda.reset_peak_memory_stats,
                               lambda: torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), nr)
        if kind == "mps":
            return DeviceHooks(torch.mps.synchronize, noop, nr,
                               lambda: torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
    except Exception:
        pass
    return DeviceHooks(noop, noop, nr, nr)


def gpu_hooks() -> tuple[Callable[[], None], Callable[[], float | str], Callable[[], None]]:
    """(synchronize, peak_allocated_mb, reset_peak) for CUDA if it is available, else no-ops / NOT_RECORDED."""
    cuda, _mps = detect_accelerators()
    h = device_hooks("cuda" if cuda else "cpu")
    return h.sync, h.peak_mb, h.reset_peak


# ------------------------------------------------------------------------------------------ measurement helpers
def peak_rss_mb() -> float:
    """Process peak resident set size in MB (``ru_maxrss`` is bytes on macOS, kilobytes on Linux)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(v) / (1024.0 * 1024.0) if sys.platform == "darwin" else float(v) / 1024.0


def current_rss_mb() -> float | str:
    """Current resident set size in MB, or NOT_RECORDED when the platform offers no cheap way."""
    try:
        with open("/proc/self/statm") as f:                              # Linux
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil                                                    # optional dependency
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return NOT_RECORDED


def summarise_ms(values_s: list[float]) -> dict[str, float]:
    a = np.asarray(values_s, dtype=np.float64) * 1000.0
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def time_calls(fn: Callable[[Any], Any], items: list[Any], warmup: int,
               after: Callable[[], None] | None = None) -> list[float]:
    """Seconds per call of ``fn(item)``; ``warmup`` untimed calls (cycling over ``items``) come first. ``after`` (e.g. a
    device synchronise) runs inside the timed region after every call."""
    if not items:
        raise EfficiencyError("no items to time")
    for i in range(max(0, warmup)):
        fn(items[i % len(items)])
        if after:
            after()
    out = []
    for it in items:
        t0 = time.perf_counter()
        fn(it)
        if after:
            after()
        out.append(time.perf_counter() - t0)
    return out


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(),
                            "processor": platform.processor() or NOT_RECORDED, "cpu_count": os.cpu_count(),
                            "python": platform.python_version(), "numpy": np.__version__,
                            "torch": NOT_RECORDED, "torch_cuda_available": NOT_RECORDED, "torch_mps_available": NOT_RECORDED,
                            "gpu_name": NOT_RECORDED, "cuda_version": NOT_RECORDED, "onnxruntime": NOT_RECORDED,
                            "onnxruntime_providers": NOT_RECORDED}
    try:
        import torch
        info["torch"] = torch.__version__
        cuda, mps = detect_accelerators(torch)
        info["torch_cuda_available"], info["torch_mps_available"] = cuda, mps
        if cuda:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = str(torch.version.cuda)
    except Exception:
        pass
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
        info["onnxruntime_providers"] = list(onnxruntime.get_available_providers())
    except Exception:
        pass
    return info


def list_images(images_dir: Path, n: int) -> list[Path]:
    files = sorted(p for p in Path(images_dir).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(files) < n:
        raise EfficiencyError(f"{images_dir} holds {len(files)} images, fewer than the {n} requested")
    return files[:n]


class Setup(NamedTuple):
    prep: Callable[[Any], Any]
    embed_one: Callable[[Any], Any]
    embed_batch: Callable[[Sequence[Any]], Any]
    items: list[Any]
    synthetic: bool
    load_s: float
    device_used: str
    device_detail: str


def _synthetic_setup(n: int) -> Setup:
    rng = np.random.default_rng(0)
    w = rng.normal(size=(64 * 64, 128)).astype(np.float32)
    imgs = [rng.normal(size=(64, 64)).astype(np.float32) for _ in range(n)]
    return Setup(lambda x: x, lambda x: x.reshape(1, -1) @ w,
                 lambda xs: np.stack([x.reshape(-1) for x in xs]) @ w, imgs, True, 0.0, "cpu (numpy)", "numpy matrix product")


def _real_setup(model_name: str, images: list[Path], allow_stub: bool, device: str = "auto") -> Setup:
    from faceeval.core.registry import model_registry
    from faceeval.preprocessing.pipeline import alignment_for_config, pipeline_for_model
    import faceeval.models  # noqa: F401  (registers the models)
    cuda, mps = detect_accelerators()
    choice = select_device(device, cuda_available=cuda, mps_available=mps)       # refuses an unavailable explicit device
    torch_model = model_name in TORCH_DEVICE_MODELS
    if not torch_model and device.strip().lower() not in ("auto", "cpu"):
        raise EfficiencyError(f"{model_name} runs on ONNX Runtime / dlib / scikit-learn, not on a torch device: "
                              f"--device {device} cannot be honoured; use --device cpu (or auto) for this model")
    pipeline = pipeline_for_model(model_name, skip_detection_for_pre_cropped=True,
                                  alignment=alignment_for_config(model_name, None))      # same pipeline as the experiments
    model = model_registry.get(model_name)
    if torch_model and device.strip().lower() != "auto":
        model._device_str = choice.name                                  # explicit request: set before the backbone loads
    t0 = time.perf_counter()
    model.fit_embedding([], [])                                          # frozen backbone: loads the weights
    load_s = time.perf_counter() - t0
    if hasattr(model, "is_stub_backbone") and model.is_stub_backbone() and not allow_stub:
        raise EfficiencyError(f"{model_name}: backbone is a stub, not a genuine pretrained model (refused)")

    def prep(path: Path):
        r = pipeline.process_path(str(path), image_id=path.name)
        return r.array if r.success else None

    dev = getattr(model, "_device_str", None)
    label = device_label(dev) if torch_model else (
        "cpu" if model_name not in ("arcface", "insightface") else "onnxruntime (provider chosen by the model pack)")
    detail = f"requested={device}; model._device_str={dev}; {choice.reason}"
    if model_name in ("arcface", "insightface"):
        detail += f"; model providers={getattr(model, '_providers', NOT_RECORDED)}"
    return Setup(prep, (lambda arr: model.extract_features([arr])), (lambda arrs: model.extract_features(list(arrs))),
                 list(images), False, load_s, label, detail)


def time_batches(embed_batch: Callable[[Sequence[Any]], Any], arrays: list[Any], batch_size: int, warmup: int,
                 after: Callable[[], None] | None = None) -> dict[str, Any]:
    """Throughput of ``embed_batch`` over consecutive batches of ``batch_size`` (a final short batch is dropped)."""
    if batch_size < 1:
        raise EfficiencyError("batch size must be >= 1")
    n_batches = len(arrays) // batch_size
    if n_batches < 1:
        return {"batch_size": batch_size, "skipped": f"only {len(arrays)} usable images (< {batch_size})"}
    batches = [arrays[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]
    t = time_calls(embed_batch, batches, warmup, after)
    ms = summarise_ms(t)
    return {"batch_size": batch_size, "n_batches": n_batches, "ms_per_batch_mean": ms["mean"], "ms_per_batch_p95": ms["p95"],
            "ms_per_image_mean": ms["mean"] / batch_size, "images_per_s": 1000.0 * batch_size / ms["mean"]}


def run_worker(model_name: str, images_dir: Path | None, n_images: int, warmup: int, allow_stub: bool,
               device: str = "auto", batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES) -> dict[str, Any]:
    """Measure ONE model in THIS process (call it in a fresh process; see ``main``)."""
    if any(int(b) < 1 for b in batch_sizes):
        raise EfficiencyError("batch sizes must be >= 1")
    rss0 = current_rss_mb()
    if model_name == SYNTHETIC:
        cuda, mps = detect_accelerators()
        select_device(device if device in ("auto", "cpu") else "cpu", cuda_available=cuda, mps_available=mps)
        if device not in ("auto", "cpu"):
            raise EfficiencyError("the synthetic pseudo-model runs on the CPU only")
        su = _synthetic_setup(n_images)
        image_list_sha = NOT_RECORDED
    else:
        if images_dir is None:
            raise EfficiencyError("--images-dir is required for a real model")
        files = list_images(images_dir, n_images)
        image_list_sha = hashlib.sha256("\n".join(p.name for p in files).encode()).hexdigest()
        su = _real_setup(model_name, files, allow_stub, device)
    rss_loaded = current_rss_mb()
    rss_peak_after_load = peak_rss_mb()
    kind = device_kind(su.device_used)
    hooks = device_hooks(kind)

    t0 = time.perf_counter()
    first = su.prep(su.items[0])                                         # cold preprocessing call
    first_prep_ms = (time.perf_counter() - t0) * 1000.0
    t_prep = time_calls(su.prep, su.items, warmup)                       # warmed per-image preprocessing
    arrays = [su.prep(it) for it in su.items]
    ok = [a for a in arrays if a is not None]
    n_failed_prep = len(arrays) - len(ok)
    if not ok:
        raise EfficiencyError("preprocessing produced no usable image")
    hooks.reset_peak()
    t0 = time.perf_counter()
    su.embed_one(ok[0])                                                  # cold embedding call (includes lazy init)
    hooks.sync()
    first_embed_ms = (time.perf_counter() - t0) * 1000.0
    t_embed = time_calls(su.embed_one, ok, warmup, after=hooks.sync)
    batched = [time_batches(su.embed_batch, ok, int(b), warmup, after=hooks.sync) for b in batch_sizes]
    prep_ms, emb_ms = summarise_ms(t_prep), summarise_ms(t_embed)
    gpu_peak = hooks.peak_mb() if kind == "cuda" else NOT_RECORDED
    gpu_current = hooks.current_mb() if kind == "mps" else NOT_RECORDED
    del first
    return {"model": model_name, "synthetic": su.synthetic, "n_images": len(su.items), "n_preprocess_failed": n_failed_prep,
            "n_embedded": len(ok), "warmup_calls": warmup, "batch_sizes": [int(b) for b in batch_sizes],
            "image_list_sha256": image_list_sha,
            "load_s": su.load_s, "first_preprocess_ms": first_prep_ms, "first_embed_ms": first_embed_ms,
            "preprocess_ms_mean": prep_ms["mean"], "preprocess_ms_p50": prep_ms["p50"], "preprocess_ms_p95": prep_ms["p95"],
            "embed_ms_mean": emb_ms["mean"], "embed_ms_p50": emb_ms["p50"], "embed_ms_p95": emb_ms["p95"],
            "images_per_s": 1000.0 / (prep_ms["mean"] + emb_ms["mean"]), "batched": batched,
            "rss_before_load_mb": rss0, "rss_after_load_mb": rss_loaded, "rss_peak_after_load_mb": rss_peak_after_load,
            "rss_peak_mb": peak_rss_mb(), "device_requested": device, "device_used": su.device_used, "device_kind": kind,
            "device_detail": su.device_detail, "gpu_peak_allocated_mb": gpu_peak, "gpu_current_allocated_mb_after": gpu_current,
            "hardware": hardware_info()}


COMPARABILITY_FIELDS = ("device_kind", "n_images", "image_list_sha256", "warmup_calls", "batch_sizes")
COMPARABILITY_HW = ("platform", "machine", "python", "torch", "onnxruntime")


def comparability(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Which measurement conditions differ between rows. Latency and memory of rows that differ are NOT comparable."""
    differing = []
    for f in COMPARABILITY_FIELDS:
        if len({json.dumps(r.get(f), sort_keys=True) for r in results}) > 1:
            differing.append(f)
    for f in COMPARABILITY_HW:
        if len({json.dumps(r["hardware"].get(f), sort_keys=True) for r in results}) > 1:
            differing.append(f"hardware.{f}")
    return {"all_same_conditions": not differing, "differing_fields": differing,
            "note": ("Rows measured under different conditions must not be compared with each other." if differing
                     else "All rows were measured with the same device kind, images, warm-up, batch sizes and software.")}


def _worker_cmd(model: str, images_dir: Path | None, n: int, warmup: int, allow_stub: bool, out: Path,
                device: str, batch_sizes: Sequence[int]) -> list[str]:
    cmd = [sys.executable, "-m", "faceeval.analysis.efficiency", "--worker", "--model", model,
           "--n-images", str(n), "--warmup", str(warmup), "--worker-out", str(out), "--device", device,
           "--batch-sizes", *[str(b) for b in batch_sizes]]
    if images_dir is not None:
        cmd += ["--images-dir", str(images_dir)]
    return cmd + (["--allow-stub"] if allow_stub else [])


def run_all(models: list[str], images_dir: Path | None, n_images: int, warmup: int, out_dir: Path,
            allow_stub: bool = False, timeout_s: int = 3600, overwrite: bool = False,
            device: str = "auto", batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES) -> list[dict[str, Any]]:
    out_dir = Path(out_dir)
    if out_dir.is_dir() and any(out_dir.iterdir()) and not overwrite:
        raise EfficiencyError(f"refusing to write into the non-empty directory {out_dir} (new directory or --overwrite)")
    if len(set(models)) != len(models):
        raise EfficiencyError("duplicate model names")
    results = []
    for m in models:                                                     # one fresh process per model
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "r.json"
            p = subprocess.run(_worker_cmd(m, images_dir, n_images, warmup, allow_stub, tmp, device, batch_sizes),
                               capture_output=True, text=True, timeout=timeout_s)
            if p.returncode != 0 or not tmp.is_file():
                raise EfficiencyError(f"{m}: worker failed (exit {p.returncode}): {(p.stderr or p.stdout).strip()[-400:]}")
            results.append(json.loads(tmp.read_text()))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "efficiency_results.json").write_text(json.dumps(
        {"results": results, "comparability": comparability(results),
         "note": "One machine, one run, the stated images; no confidence intervals. "
                 "synthetic=true rows are harness self-tests and must not be reported."}, indent=2) + "\n")
    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("hardware", "batched")}
        row.update({f"hw_{k}": v for k, v in r["hardware"].items()})
        for b in r["batched"]:
            row[f"batch{b['batch_size']}_images_per_s"] = b.get("images_per_s", NOT_RECORDED)
            row[f"batch{b['batch_size']}_ms_per_image"] = b.get("ms_per_image_mean", NOT_RECORDED)
        flat.append(row)
    cols = list(dict.fromkeys(k for row in flat for k in row))
    with open(out_dir / "efficiency_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval=NOT_RECORDED)
        w.writeheader()
        w.writerows(flat)
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=("Per-model initialisation time, preprocessing time, single-image latency, batched throughput and "
                     "process-peak memory of the deep models, each model in its own subprocess. Measures this machine only; "
                     "no experiment result is read or modified."),
        epilog=("Example:\n  python -m faceeval.analysis.efficiency --models facenet arcface "
                "--images-dir data/lfw --n-images 200 --out <NEW_OUTPUT_DIRECTORY>"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, help="registered model names (e.g. facenet arcface)")
    ap.add_argument("--images-dir", type=Path, default=None, help="folder of face images (searched recursively)")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5, help="untimed calls before timing (preprocessing, embedding, each batch size)")
    ap.add_argument("--device", default="auto", help="auto (experiment rule) | cpu | cuda | cuda:N | mps | best")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES),
                    help="batch sizes for the throughput measurement (single-image latency is always measured)")
    ap.add_argument("--out", type=Path, default=None, help="NEW output directory")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-stub", action="store_true", help="measure a stub backbone anyway (meaningless numbers)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds allowed per model")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.n_images < 1 or a.warmup < 0:
            raise EfficiencyError("--n-images must be >= 1 and --warmup >= 0")
        if a.worker:
            res = run_worker(a.model, a.images_dir, a.n_images, a.warmup, a.allow_stub, a.device, a.batch_sizes)
            a.worker_out.write_text(json.dumps(res))
            return 0
        if not a.models or a.out is None:
            raise EfficiencyError("--models and --out are required")
        results = run_all(a.models, a.images_dir, a.n_images, a.warmup, a.out, a.allow_stub, a.timeout, a.overwrite,
                          a.device, a.batch_sizes)
        for r in results:
            print(f"{r['model']} [{r['device_kind']}]: load {r['load_s']:.2f} s, embed {r['embed_ms_mean']:.2f} ms/img, "
                  f"peak RSS {r['rss_peak_mb']:.0f} MB" + (" [SYNTHETIC SELF-TEST]" if r["synthetic"] else ""))
        cmp_ = comparability(results)
        if not cmp_["all_same_conditions"]:
            print("WARNING: rows differ in " + ", ".join(cmp_["differing_fields"]) + "; do not compare them with each other.")
        print(f"written: {a.out}")
        return 0
    except EfficiencyError as exc:
        print(f"EFFICIENCY BENCHMARK REFUSED: {exc}", file=sys.stderr if a.worker else sys.stdout)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
