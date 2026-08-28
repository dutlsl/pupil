# KETI support scripts

This directory groups the DAVIS-only preview and the Eye0 NIR ↔ DAVIS
checkerboard calibration workflow. The launchers intentionally reuse the
project's existing calibration implementation; no Pupil or TDTracker process
is started by `event_camera_preview.py`.

Run from `pupil_src` with the environment that provides `dv_processing`:

```bash
python KETI_supp/event_camera_preview.py
python KETI_supp/show_checkerboard.py
python KETI_supp/capture_nir_event_probe.py --eye0-device /dev/video0
python KETI_supp/calibrate_nir_event.py \
  --eye0-dir samples/eye0 \
  --event-dir samples/event \
  --square-size-mm 10 \
  --output KETI_supp/event_from_eye0.json
```

For calibration, save matched checkerboard poses with the same filename stem:

```text
samples/eye0/0001.png
samples/event/0001.npz
```

The event `.npz` can contain a reconstructed image (`image` or `frame`) or raw
`x`, `y`, and `p` event arrays from one checkerboard inversion interval.
