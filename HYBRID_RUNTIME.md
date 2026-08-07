# Hybrid RITnet + DVS runtime

The Hybrid detector keeps NIR/RITnet as the ellipse and pye3d anchor while a
spawned worker owns the complete DAVIS hot path:

```text
DAVIS -> 1 ms slice -> BinaRep -> 8-frame stack
      -> TDTracker CUDA Graph -> async pinned result ring
      -> latest-only parent state -> 1000 Hz publish
```

## Run

The default TDTracker checkpoint is
`pupil_src/shared_modules/pupil_detector_plugins/best_checkpoint.pth`.

```bash
./run_pupil_hybrid_graph.sh
```

The launcher builds `_native_binarep.so` when needed. Set `CC` to select a
different C compiler and `PUPIL_PYTHON` to select the Python interpreter.

Important runtime settings:

```text
PUPIL_HYBRID_TDTRACKER_MODE=graph  # graph, compile, eager, or auto
PUPIL_HYBRID_ASYNC_RESULT=1
PUPIL_HYBRID_ASYNC_RESULT_SLOTS=8
PUPIL_HYBRID_CPP_BINAREP=1
PUPIL_HYBRID_CONF_THRESHOLD=0.3
PUPIL_HYBRID_DVS_EYE_ID=0
PUPIL_HYBRID_DVS_WIDTH=346
PUPIL_HYBRID_DVS_HEIGHT=260
PUPIL_HYBRID_PROFILE_SYNC=0
```

`auto` falls back in the order graph -> compile -> eager. Keep
`PUPIL_HYBRID_PROFILE_SYNC=0` in normal operation; synchronization is only for
stage timing.

Optional affinity variables accept comma-separated CPUs and ranges:

```text
PUPIL_CPU_AFFINITY_MAIN
PUPIL_CPU_AFFINITY_WORLD
PUPIL_CPU_AFFINITY_EYE
PUPIL_CPU_AFFINITY_EYE0
PUPIL_CPU_AFFINITY_EYE1
PUPIL_CPU_AFFINITY_DVS
```

Choose these values only after checking GPU/NUMA topology on the target host.

## Validation

Do not use publish rate alone as the success signal. Run the full RITnet,
pye3d, and UI workload with the actual DAVIS/NIR devices for at least five
minutes and confirm:

```text
slice >= 990 Hz
submit >= 990 Hz
infer >= 990 Hz
parent state >= 990 Hz
publish >= 990 Hz
drop = 0
state_age_ms does not grow continuously
```

Published pupil data contains `seq_id`, all submit/ready/receive/publish
monotonic timestamps, derived latency fields, `fresh`, `source`, and
`state_age_ms`. `fresh` and `seq_id` distinguish a new TDTracker state from a
timer resend.
