#!/usr/bin/env python3
"""Server 3 TDTracker/RITnet microbenchmark.

This intentionally measures only a fixed-address GPU inference hot path.  It
does not start cameras or the Pupil runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SHARED_MODULES = ROOT / "pupil_src" / "shared_modules"
PLUGIN_DIR = SHARED_MODULES / "pupil_detector_plugins"
sys.path.insert(0, str(SHARED_MODULES))

TD_CHECKPOINT = PLUGIN_DIR / "best_checkpoint.pth"
RGB_CHECKPOINT = PLUGIN_DIR / "best_model.pkl"
DEVICE = torch.device("cuda:0")


class TDTrackerWithSimDR(nn.Module):
    """TDTracker forward plus the repository's SimDR coordinate/confidence decode."""

    def __init__(self) -> None:
        super().__init__()
        from pupil_detector_plugins.hybrid_runtime import _load_tdtracker

        self.model = _load_tdtracker(torch, str(TD_CHECKPOINT), DEVICE)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output_x, output_y = self.model(input_tensor)
        preds_x = output_x.argmax(dim=2)
        preds_y = output_y.argmax(dim=2)
        pooled_x = torch.nn.functional.avg_pool1d(
            output_x, kernel_size=10, stride=1, padding=1
        )
        pooled_y = torch.nn.functional.avg_pool1d(
            output_y, kernel_size=10, stride=1, padding=1
        )
        x_prob = torch.softmax(pooled_x, dim=2).amax(dim=2)
        y_prob = torch.softmax(pooled_y, dim=2).amax(dim=2)
        coordinates = torch.stack(
            (preds_x.float() / 80.0, preds_y.float() / 60.0), dim=2
        )
        return coordinates, x_prob + y_prob


class RITnetWithArgmax(nn.Module):
    """RITnet forward plus GPU argmax."""

    def __init__(self) -> None:
        super().__init__()
        from pupil_detector_plugins.models import model_dict

        self.model = model_dict["densenet"]().to(DEVICE).eval()
        state = torch.load(RGB_CHECKPOINT, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(state)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor).argmax(dim=1)


@dataclass
class Measurement:
    case: int
    condition: str
    method: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    hz: float
    improvement_vs_eager_pct: float = 0.0
    best_in_condition: bool = False


@dataclass
class MethodRunners:
    td: Callable[[], Any]
    rgb: Callable[[], Any] | None
    keepalive: list[Any]


def run_repeated(fn: Callable[[], Any], count: int) -> None:
    with torch.inference_mode():
        for _ in range(count):
            fn()
    torch.cuda.synchronize(DEVICE)


def capture_external(
    module: nn.Module | Callable[[torch.Tensor], Any],
    input_tensor: torch.Tensor,
) -> tuple[Callable[[], Any], list[Any]]:
    graph = torch.cuda.CUDAGraph()
    # In PyTorch 2.6 inference mode must already be active when capture_begin
    # touches generator state associated with inference tensors.
    with torch.inference_mode():
        with torch.cuda.graph(graph):
            static_output = module(input_tensor)

    def replay() -> Any:
        graph.replay()
        return static_output

    return replay, [graph, static_output, module]


def make_method_runners(
    method: str,
    td_input: torch.Tensor,
    rgb_input: torch.Tensor,
    preparation: int,
) -> MethodRunners:
    print(f"\nPreparing method: {method}", flush=True)
    td_module = TDTrackerWithSimDR().to(DEVICE).eval()
    rgb_module = RITnetWithArgmax().to(DEVICE).eval()
    keepalive: list[Any] = [td_module, rgb_module]

    if method == "Eager":
        td = lambda: td_module(td_input)
        rgb = lambda: rgb_module(rgb_input)
    elif method == "torch.compile":
        td_compiled = torch.compile(td_module, mode="reduce-overhead")
        rgb_compiled = torch.compile(rgb_module, mode="reduce-overhead")
        td = lambda: td_compiled(td_input)
        rgb = lambda: rgb_compiled(rgb_input)
        keepalive += [td_compiled, rgb_compiled]
        print(f"  compile preparation: {preparation} TD + RGB iterations", flush=True)
        run_repeated(td, preparation)
        run_repeated(rgb, preparation)
    elif method == "Eager + external CUDA Graph":
        td_eager = lambda: td_module(td_input)
        rgb_eager = lambda: rgb_module(rgb_input)
        print(f"  capture preparation: {preparation} TD + RGB iterations", flush=True)
        run_repeated(td_eager, preparation)
        run_repeated(rgb_eager, preparation)
        td, td_objects = capture_external(td_module, td_input)
        rgb, rgb_objects = capture_external(rgb_module, rgb_input)
        keepalive += td_objects + rgb_objects
    elif method == "torch.compile + external CUDA Graph":
        compile_options = {"triton.cudagraphs": False}
        td_compiled = torch.compile(td_module, options=compile_options)
        rgb_compiled = torch.compile(rgb_module, options=compile_options)
        td_compiled_call = lambda: td_compiled(td_input)
        rgb_compiled_call = lambda: rgb_compiled(rgb_input)
        print(f"  compile/capture preparation: {preparation} TD + RGB iterations", flush=True)
        run_repeated(td_compiled_call, preparation)
        run_repeated(rgb_compiled_call, preparation)
        td, td_objects = capture_external(td_compiled, td_input)
        rgb, rgb_objects = capture_external(rgb_compiled, rgb_input)
        keepalive += [td_compiled, rgb_compiled] + td_objects + rgb_objects
    else:
        raise ValueError(method)

    torch.cuda.synchronize(DEVICE)
    return MethodRunners(td=td, rgb=rgb, keepalive=keepalive)


def measure(
    case: int,
    condition: str,
    method: str,
    td: Callable[[], Any],
    rgb: Callable[[], Any] | None,
    warmup: int,
    iterations: int,
) -> Measurement:
    if rgb is None:
        warmup_fn = td
    else:
        def warmup_fn() -> None:
            rgb()
            torch.cuda.synchronize(DEVICE)
            td()

    print(
        f"  Case {case}: measurement warm-up {warmup}, timed iterations {iterations}",
        flush=True,
    )
    run_repeated(warmup_fn, warmup)

    elapsed_ms = np.empty(iterations, dtype=np.float64)
    with torch.inference_mode():
        for index in range(iterations):
            if rgb is not None:
                rgb()
                torch.cuda.synchronize(DEVICE)
            else:
                torch.cuda.synchronize(DEVICE)
            started_ns = time.perf_counter_ns()
            td()
            torch.cuda.synchronize(DEVICE)
            elapsed_ms[index] = (time.perf_counter_ns() - started_ns) / 1e6

    mean_ms = float(elapsed_ms.mean())
    return Measurement(
        case=case,
        condition=condition,
        method=method,
        mean_ms=mean_ms,
        p50_ms=float(np.percentile(elapsed_ms, 50)),
        p95_ms=float(np.percentile(elapsed_ms, 95)),
        hz=1000.0 / mean_ms,
    )


def dynamo_diagnostics(td_input: torch.Tensor) -> dict[str, Any]:
    print("\nCollecting TDTracker Dynamo diagnostics", flush=True)
    module = TDTrackerWithSimDR().to(DEVICE).eval()
    with torch.inference_mode():
        explanation = torch._dynamo.explain(module)(td_input)
    reasons = []
    for reason in explanation.break_reasons:
        reasons.append(str(reason))
    result = {
        "graph_count": int(explanation.graph_count),
        "graph_break_count": int(explanation.graph_break_count),
        "break_reasons": reasons,
    }
    del module
    torch.cuda.empty_cache()
    return result


def nvidia_smi() -> str:
    return subprocess.run(
        ["nvidia-smi"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "server3_tdtracker_results.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)

    start_smi = nvidia_smi()
    print("=== nvidia-smi at measurement start ===", flush=True)
    print(start_smi, flush=True)

    # These two objects remain alive and retain the same data_ptr for every case.
    td_input = torch.rand(
        (1, 8, 2, 60, 80), dtype=torch.float32, device=DEVICE
    ).mul_(15.0)
    rgb_input = torch.rand(
        (1, 1, 400, 400), dtype=torch.float32, device=DEVICE
    )
    input_metadata = {
        "td": {
            "shape": list(td_input.shape),
            "dtype": str(td_input.dtype),
            "data_ptr": td_input.data_ptr(),
            "min": float(td_input.min().item()),
            "max": float(td_input.max().item()),
        },
        "rgb": {
            "shape": list(rgb_input.shape),
            "dtype": str(rgb_input.dtype),
            "data_ptr": rgb_input.data_ptr(),
            "min": float(rgb_input.min().item()),
            "max": float(rgb_input.max().item()),
        },
    }

    methods = [
        "Eager",
        "torch.compile",
        "Eager + external CUDA Graph",
        "torch.compile + external CUDA Graph",
    ]
    measurements: list[Measurement] = []
    for method_index, method in enumerate(methods):
        runners = make_method_runners(
            method, td_input, rgb_input, args.preparation
        )
        measurements.append(
            measure(
                case=method_index + 1,
                condition="TDTracker-only",
                method=method,
                td=runners.td,
                rgb=None,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )
        measurements.append(
            measure(
                case=method_index + 5,
                condition="RGB immediately before TDTracker",
                method=method,
                td=runners.td,
                rgb=runners.rgb,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )
        del runners
        torch.cuda.synchronize(DEVICE)
        torch.cuda.empty_cache()

    for condition in ("TDTracker-only", "RGB immediately before TDTracker"):
        rows = [row for row in measurements if row.condition == condition]
        eager_mean = next(row.mean_ms for row in rows if row.method == "Eager")
        best_mean = min(row.mean_ms for row in rows)
        for row in rows:
            row.improvement_vs_eager_pct = (
                (eager_mean - row.mean_ms) / eager_mean * 100.0
            )
            row.best_in_condition = row.mean_ms == best_mean

    diagnostics = dynamo_diagnostics(td_input)
    measurements.sort(key=lambda row: row.case)
    payload = {
        "environment": {
            "hostname": platform.node(),
            "cpu": cpu_model(),
            "gpu": torch.cuda.get_device_name(DEVICE),
            "device": str(DEVICE),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "driver_nvidia_smi": start_smi.splitlines()[2].strip()
            if len(start_smi.splitlines()) > 2
            else "",
            "pid": os.getpid(),
        },
        "settings": {
            "preparation_iterations": args.preparation,
            "measurement_warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "timing": "perf_counter_ns; synchronize before start and after TDTracker",
            "td_checkpoint": str(TD_CHECKPOINT),
            "rgb_checkpoint": str(RGB_CHECKPOINT),
        },
        "inputs": input_metadata,
        "measurements": [asdict(row) for row in measurements],
        "tdtracker_dynamo": diagnostics,
        "nvidia_smi_at_start": start_smi,
        "scope_notes": [
            "Not measured with real cameras or the Pupil runtime.",
            "RGB and TDTracker ran serially, never concurrently.",
            "RGB execution time is excluded from RGB + Event results.",
            "H2D, D2H, FIFO, DAVIS, BinaRep, IPC, pye3d, and UI are excluded.",
            "Hz is theoretical serial throughput of the timed TDTracker section, not camera output rate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.output}", flush=True)
    print(json.dumps(payload["measurements"], indent=2), flush=True)
    print("Dynamo:", json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
