"""Integrated Eye0 Pye3D producer for the NIR + event-camera runtime.

This plugin deliberately keeps the normal :class:`Pye3DPlugin` detection path
intact.  Its extra responsibility is small: after each NIR result it updates a
numeric eye-model snapshot for the independent TDTracker process, then hands a
possible NIR fallback datum to the Eye0 3D broker.  The broker, rather than
this plugin or the DVS process, is the sole publisher of final Eye0 3D data.
"""

from __future__ import annotations

import logging

from .pye3d_plugin import Pye3DPlugin


logger = logging.getLogger(__name__)


class HybridPye3DPlugin(Pye3DPlugin):
    """Normal NIR Pye3D plus shared-state/fallback hand-off for Eye0."""

    label = "Hybrid NIR Pye3D"

    def detect(self, frame, **kwargs):
        datum = super().detect(frame, **kwargs)

        # Only Eye0 in main_int.py has these attributes. Eye1 intentionally
        # remains an ordinary NIR/Pye3D producer even though this class is
        # registered for both eye processes.
        store = getattr(self.g_pool, "hybrid_pye3d_snapshot_store", None)
        broker = getattr(self.g_pool, "hybrid_pye3d_broker", None)
        if store is None or broker is None:
            return datum

        try:
            snapshot = store.update_from_pye3d_datum(
                datum,
                focal_length=self.camera.focal_length,
                frame_size=(frame.width, frame.height),
            )
        except Exception:
            # A malformed Pye3D datum must never take down NIR tracking.  The
            # next valid datum replaces the shared state; meanwhile the broker
            # still has a normal NIR fallback candidate to publish.
            logger.exception("Could not update the Eye0 Pye3D fusion snapshot")
            snapshot = None

        datum["source"] = "nir_pye3d_fallback"
        datum["source_2d"] = "nir_ritnet"
        datum["hybrid_output_owner"] = "eye0_3d_broker"
        if snapshot is not None:
            datum["pye3d_snapshot_generation"] = snapshot.generation

        # eye.py must not publish this datum directly: the broker serializes it
        # with DVS-fused data and makes the fallback decision in one place.  A
        # latest-only queue drop is preferable to reopening a duplicate Eye0
        # stream, so this remains external even if a candidate is superseded.
        datum["_published_externally"] = True
        try:
            broker.submit_nir_fallback(datum)
        except Exception:
            logger.exception("Could not enqueue Eye0 NIR/Pye3D fallback datum")

        return datum
