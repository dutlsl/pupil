"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import traceback
import typing as T

import numpy as np
import scipy.spatial
from calibration_choreography import (
    ChoreographyAction,
    ChoreographyMode,
    ChoreographyNotification,
)
from gaze_mapping import gazer_classes_by_class_name, registered_gazer_classes
from gaze_mapping.notifications import (
    CalibrationResultNotification,
    CalibrationSetupNotification,
)
from gaze_mapping.utils import closest_matches_monocular
from plugin import Plugin


logger = logging.getLogger(__name__)


class CalculationResult(T.NamedTuple):
    result: float
    num_used: int
    num_total: int


class CorrelatedAndCoordinateTransformedResult(T.NamedTuple):
    """Holds result from correlating reference and gaze data and their respective
    transformations into norm, image, and camera coordinate systems.
    """

    norm_space: np.ndarray  # shape: 2*n, 2
    image_space: np.ndarray  # shape: 2*n, 2
    camera_space: np.ndarray  # shape: 2*n, 3

    @staticmethod
    def empty() -> "CorrelatedAndCoordinateTransformedResult":
        return CorrelatedAndCoordinateTransformedResult(
            norm_space=np.ndarray([]),
            image_space=np.ndarray([]),
            camera_space=np.ndarray([]),
        )

    @property
    def is_valid(self) -> bool:
        if len(self.norm_space.shape) != 2:
            return False
        # TODO: Make validity check exhaustive
        return True


class CorrelationError(ValueError):
    pass


class AccuracyPrecisionResult(T.NamedTuple):
    accuracy: CalculationResult
    precision: CalculationResult
    error_lines: np.ndarray
    correlation: CorrelatedAndCoordinateTransformedResult

    @staticmethod
    def failed() -> "AccuracyPrecisionResult":
        return AccuracyPrecisionResult(
            accuracy=CalculationResult(0.0, 0, 0),
            precision=CalculationResult(0.0, 0, 0),
            error_lines=np.array([]),
            correlation=CorrelatedAndCoordinateTransformedResult.empty(),
        )

    @property
    def is_valid(self) -> bool:
        if not self.correlation.is_valid:
            return False
        # TODO: Make validity check exhaustive
        return True


class ValidationInput:
    def __init__(self):
        self.clear()

    @property
    def gazer_class(self) -> T.Optional[T.Any]:
        return self.__gazer_class

    @property
    def gazer_params(self) -> T.Optional[T.Any]:
        return self.__gazer_params

    @property
    def gazer_class_name(self) -> T.Optional[str]:
        return self.__gazer_class.__name__ if self.__gazer_class is not None else None

    @property
    def pupil_list(self) -> T.Optional[T.Any]:
        return self.__pupil_list

    @property
    def ref_list(self) -> T.Optional[T.Any]:
        return self.__ref_list

    @property
    def is_complete(self) -> bool:
        return None not in (
            self.pupil_list,
            self.ref_list,
            self.gazer_class,
            self.gazer_params,
        )

    def clear(self):
        self.__pupil_list = None
        self.__ref_list = None
        self.__gazer_class = None
        self.__gazer_params = None

    def update(
        self, gazer_class_name: str, gazer_params=..., pupil_list=..., ref_list=...
    ):
        if (
            self.gazer_class_name is not None
            and self.gazer_class_name != gazer_class_name
        ):
            logger.debug(
                f'Overwriting gazer_class_name from "{self.gazer_class_name}" to '
                f'"{gazer_class_name}" and resetting the input.'
            )
            self.clear()

        self.__gazer_class = self.__gazer_class_from_name(gazer_class_name)

        if gazer_params is not ...:
            self.__gazer_params = gazer_params

        if pupil_list is not ...:
            self.__pupil_list = pupil_list

        if ref_list is not ...:
            self.__ref_list = ref_list

    @staticmethod
    def __gazer_class_from_name(gazer_class_name: str) -> T.Optional[T.Any]:
        gazers_by_name = gazer_classes_by_class_name(registered_gazer_classes())

        try:
            gazer_cls = gazers_by_name[gazer_class_name]
        except KeyError:
            logger.error(f'Unknown gazer "{gazer_class_name}"')
            return None

        return gazer_cls


class Accuracy_Visualizer(Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites not between
    """

    order = 0.8
    icon_chr = chr(0xEC11)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        outlier_threshold=1.3,
        vis_mapping_error=True,
        vis_calibration_area=True,
        enable_5stack_summary=False,
    ):
        super().__init__(g_pool)
        self.vis_mapping_error = vis_mapping_error
        self.vis_calibration_area = vis_calibration_area
        self.calibration_area = None
        self.accuracy = None
        self.precision = None
        self.error_lines = None

        self.recent_input = ValidationInput()

        # .5 degrees, used to remove outliers from precision calculation
        self.succession_threshold = np.cos(np.deg2rad(0.5))
        self._outlier_threshold = outlier_threshold  # in degrees

        # Structured experiment reporting state
        self.calibration_counter = 0
        self.validation_counter_under_current_calib = 0
        self.total_validation_counter = 0
        self.last_calib_accuracy = None
        self.last_calib_precision = None
        self._current_eval_mode = "unknown"

        # 5-Stack Validation Demo Options & Accumulator State
        # Check if g_pool already holds a user-toggled enable_5stack_summary
        if hasattr(g_pool, "enable_5stack_summary"):
            self._enable_5stack_summary = bool(g_pool.enable_5stack_summary)
        else:
            self._enable_5stack_summary = bool(enable_5stack_summary)
            if hasattr(g_pool, "enable_5stack_summary"):
                g_pool.enable_5stack_summary = self._enable_5stack_summary

        self.stack_target_count = 5

        # Persistent session state attached to g_pool
        if not hasattr(g_pool, "_accuracy_vis_val_stack"):
            g_pool._accuracy_vis_val_stack = []
        if not hasattr(g_pool, "_accuracy_vis_val_stack_details"):
            g_pool._accuracy_vis_val_stack_details = []
        if not hasattr(g_pool, "_accuracy_vis_stack_active"):
            g_pool._accuracy_vis_stack_active = False
        if not hasattr(g_pool, "_accuracy_vis_stack_calib_id"):
            g_pool._accuracy_vis_stack_calib_id = None
        if not hasattr(g_pool, "_accuracy_vis_stack_calib_score"):
            g_pool._accuracy_vis_stack_calib_score = None
        if not hasattr(g_pool, "_accuracy_vis_stack_model_name"):
            g_pool._accuracy_vis_stack_model_name = "Unknown Model"

        self._val_stack = g_pool._accuracy_vis_val_stack
        self._val_stack_details = g_pool._accuracy_vis_val_stack_details

    def init_ui(self):
        from pyglui import ui

        self.add_menu()
        self.menu.label = "Accuracy Visualizer"

        mapping_error_help = """The mapping error (orange line) is the angular
                             distance between mapped pupil positions (red) and
                             their corresponding reference points (blue).
                             """.replace(
            "\n", " "
        ).replace(
            "  ", ""
        )

        calib_area_help = """The calibration area (green) is defined as the
                          convex hull of the reference points that were used
                          for calibration. 2D mapping looses accuracy outside
                          of this area. It is recommended to calibrate a big
                          portion of the subject's field of view.
                          """.replace(
            "\n", " "
        ).replace(
            "  ", ""
        )
        self.menu.append(ui.Info_Text(calib_area_help))
        self.menu.append(
            ui.Switch("vis_mapping_error", self, label="Visualize mapping error")
        )

        self.menu.append(ui.Info_Text(mapping_error_help))
        self.menu.append(
            ui.Switch("vis_calibration_area", self, label="Visualize calibration area")
        )

        general_help = """Measure gaze mapping accuracy and precision using samples
                          that were collected during calibration. The outlier threshold
                          discards samples with high angular errors.""".replace(
            "\n", " "
        ).replace(
            "  ", ""
        )
        self.menu.append(ui.Info_Text(general_help))

        # self.menu.append(ui.Info_Text(''))
        self.menu.append(
            ui.Text_Input(
                "outlier_threshold", self, label="Outlier Threshold [degrees]"
            )
        )

        accuracy_help = """Accuracy is calculated as the average angular
                        offset (distance) (in degrees of visual angle)
                        between fixation locations and the corresponding
                        locations of the fixation targets.""".replace(
            "\n", " "
        ).replace(
            "  ", ""
        )

        precision_help = """Precision is calculated as the Root Mean Square (RMS)
                            of the angular distance (in degrees of visual angle)
                            between successive samples during a fixation.""".replace(
            "\n", " "
        ).replace(
            "  ", ""
        )

        def ignore(_):
            pass

        self.menu.append(ui.Info_Text(accuracy_help))
        self.menu.append(
            ui.Text_Input(
                "accuracy",
                self,
                "Angular Accuracy",
                setter=ignore,
                getter=lambda: f"{self.accuracy.result:.3f} deg. Samples used: "
                f"{self.accuracy.num_used} / {self.accuracy.num_total}"
                if self.accuracy is not None
                else "Not available",
            )
        )
        self.menu.append(ui.Info_Text(precision_help))
        self.menu.append(
            ui.Text_Input(
                "precision",
                self,
                "Angular Precision",
                setter=ignore,
                getter=lambda: f"{self.precision.result:.3f} deg. Samples used: "
                f"{self.precision.num_used} / {self.precision.num_total}"
                if self.precision is not None
                else "Not available",
            )
        )

        demo_help = """5-Stack Validation Demo Summary: Triggered upon calibration
                       start, accumulates 5 validation rounds (including fails)
                       and outputs the Mean Accuracy, Standard Deviation, and Peak
                       metrics to the console.""".replace(
            "\n", " "
        ).replace(
            "  ", ""
        )
        self.menu.append(ui.Info_Text(demo_help))
        self.menu.append(
            ui.Switch(
                "enable_5stack_summary",
                self,
                label="5-Stack Summary Demo Output",
            )
        )

    def deinit_ui(self):
        self.remove_menu()

    @property
    def outlier_threshold(self):
        return self._outlier_threshold

    @outlier_threshold.setter
    def outlier_threshold(self, value):
        self._outlier_threshold = value
        self.notify_all(
            {"subject": "accuracy_visualizer.outlier_threshold_changed", "delay": 0.5}
        )

    @property
    def enable_5stack_summary(self) -> bool:
        if hasattr(self, "g_pool") and hasattr(self.g_pool, "enable_5stack_summary"):
            return bool(self.g_pool.enable_5stack_summary)
        return getattr(self, "_enable_5stack_summary", False)

    @enable_5stack_summary.setter
    def enable_5stack_summary(self, val: bool):
        val = bool(val)
        self._enable_5stack_summary = val
        if hasattr(self, "g_pool") and self.g_pool is not None:
            self.g_pool.enable_5stack_summary = val

    @property
    def _stack_active(self) -> bool:
        if hasattr(self, "g_pool") and hasattr(self.g_pool, "_accuracy_vis_stack_active"):
            return bool(self.g_pool._accuracy_vis_stack_active)
        return False

    @_stack_active.setter
    def _stack_active(self, val: bool):
        if hasattr(self, "g_pool") and self.g_pool is not None:
            self.g_pool._accuracy_vis_stack_active = bool(val)

    @property
    def _stack_calib_id(self):
        if hasattr(self, "g_pool") and hasattr(self.g_pool, "_accuracy_vis_stack_calib_id"):
            return self.g_pool._accuracy_vis_stack_calib_id
        return None

    @_stack_calib_id.setter
    def _stack_calib_id(self, val):
        if hasattr(self, "g_pool") and self.g_pool is not None:
            self.g_pool._accuracy_vis_stack_calib_id = val

    @property
    def _stack_calib_score(self):
        if hasattr(self, "g_pool") and hasattr(self.g_pool, "_accuracy_vis_stack_calib_score"):
            return self.g_pool._accuracy_vis_stack_calib_score
        return None

    @_stack_calib_score.setter
    def _stack_calib_score(self, val):
        if hasattr(self, "g_pool") and self.g_pool is not None:
            self.g_pool._accuracy_vis_stack_calib_score = val

    @property
    def _stack_model_name(self):
        if hasattr(self, "g_pool") and hasattr(self.g_pool, "_accuracy_vis_stack_model_name"):
            return self.g_pool._accuracy_vis_stack_model_name
        return "Unknown Model"

    @_stack_model_name.setter
    def _stack_model_name(self, val):
        if hasattr(self, "g_pool") and self.g_pool is not None:
            self.g_pool._accuracy_vis_stack_model_name = val

    def _init_5stack_session(self, reason: str = "Calibration Start"):
        """Initializes a new 5-stack validation accumulation session upon calibration start."""
        discarded = len(self._val_stack)
        self._val_stack.clear()
        self._val_stack_details.clear()
        self._stack_active = True

        calib_counter = getattr(self.g_pool, "calibration_counter", self.calibration_counter)
        self._stack_calib_id = calib_counter + 1
        self._stack_calib_score = None

        if self.enable_5stack_summary:
            discard_msg = f" (Previous incomplete stack of {discarded} rounds discarded)" if discarded > 0 else ""
            print(
                f"\n{'='*80}\n"
                f"🚀 [5-STACK DEMO] New Evaluation Session Initialized (Target: {self.stack_target_count} Validation Rounds)\n"
                f"   • Trigger: {reason}{discard_msg}\n"
                f"   • Trigger Calibration ID: #{self._stack_calib_id}\n"
                f"{'='*80}\n",
                flush=True,
            )

    def _reset_5stack_if_incomplete(self, reason: str = "Interrupting Event"):
        """Resets the 5-stack validation session if an interrupting event occurs before reaching 5 rounds."""
        if self._stack_active and 0 < len(self._val_stack) < self.stack_target_count:
            discarded = len(self._val_stack)
            self._val_stack.clear()
            self._val_stack_details.clear()
            self._stack_active = False
            if self.enable_5stack_summary:
                print(
                    f"\n⚠️ [5-STACK DEMO RESET] Validation stack RESET! ({discarded}/{self.stack_target_count} rounds discarded)\n"
                    f"   • Reason: {reason}\n",
                    flush=True,
                )

    def _print_5stack_summary(self, calib_id, val_pattern, val_pts, sample_dur):
        """Calculates and prints the Mean Accuracy and Standard Deviation summary across 5 validation rounds."""
        valid_accs = [x for x in self._val_stack if not np.isnan(x)]
        valid_precs = [d["precision"] for d in self._val_stack_details if not np.isnan(d["precision"])]

        num_total = len(self._val_stack)
        num_valid = len(valid_accs)
        num_fail = num_total - num_valid

        if num_valid > 0:
            mean_acc = float(np.mean(valid_accs))
            std_acc = float(np.std(valid_accs, ddof=1)) if num_valid > 1 else (0.0 if num_valid == 1 else np.nan)
            peak_acc = float(np.min(valid_accs))
            mean_acc_str = f"{mean_acc:.3f}°  ({mean_acc:.4f} deg)"
            std_acc_str = f"{std_acc:.3f}°  ({std_acc:.4f} deg)"
            peak_acc_str = f"{peak_acc:.3f}°  ({peak_acc:.4f} deg)"
        else:
            mean_acc_str = "N/A (All 5 Rounds Failed)"
            std_acc_str = "N/A"
            peak_acc_str = "N/A"
            mean_acc = np.nan
            std_acc = np.nan
            peak_acc = np.nan

        if len(valid_precs) > 0:
            mean_prec = float(np.mean(valid_precs))
            mean_prec_str = f"{mean_prec:.3f}°  ({mean_prec:.4f} deg)"
        else:
            mean_prec_str = "N/A"

        calib_score_str = f"{self._stack_calib_score:.3f}°" if self._stack_calib_score is not None else "N/A (Bypass / Uncalibrated)"

        rounds_lines = []
        for d in self._val_stack_details:
            r_num = d["round"]
            if d["is_fail"]:
                r_line = f"    [{r_num}/{self.stack_target_count}]  Val Round #{r_num}:  FAIL (No Valid Samples / All Outliers) ❌"
            else:
                is_peak = (d["accuracy"] == peak_acc)
                peak_badge = " 🌟 [PEAK BEST]" if is_peak and num_valid > 1 else ""
                r_line = f"    [{r_num}/{self.stack_target_count}]  Val Round #{r_num}:  {d['accuracy']:.3f}°  (Precision: {d['precision']:.3f}°, Used: {d['samples_used']}/{d['samples_total']}) ✅{peak_badge}"
            rounds_lines.append(r_line)
        rounds_block = "\n".join(rounds_lines)

        summary_msg = (
            f"\n{'='*88}\n"
            f"🏆 [DEMO REPORT] 5-STACK VALIDATION ACCURACY & STD STATISTICAL SUMMARY\n"
            f"{'='*88}\n"
            f"  📋 Experiment Context:\n"
            f"    • Active 2D Model:         {self._stack_model_name}\n"
            f"    • Parent Calibration:      #{calib_id} (Calibration Score: {calib_score_str})\n"
            f"    • Outlier Threshold:       {self.outlier_threshold}°\n"
            f"    • Validation Pattern:      {val_pattern} [{val_pts} Points]\n"
            f"    • Sample Duration:         {sample_dur} frames/point (~{sample_dur/60.0:.1f}s)\n"
            f"{'-'*88}\n"
            f"  📊 5 Validation Rounds (Fails included in count):\n"
            f"{rounds_block}\n"
            f"{'-'*88}\n"
            f"  🎯 5-Stack Statistical Summary (시연 결과 요약):\n"
            f"    ★ Mean Accuracy (평균값):         {mean_acc_str}\n"
            f"    ★ Std Deviation (표준편차 σ):     {std_acc_str}\n"
            f"    ★ Peak (최고) Accuracy:          {peak_acc_str}\n"
            f"    ★ Mean Precision (정밀도):        {mean_prec_str}\n"
            f"    ★ Evaluation Rate:               {num_valid} / {num_total} Passed ({num_fail} Failed)\n"
            f"{'='*88}\n"
        )
        print(summary_msg, flush=True)

    def on_notify(self, notification):
        subj = str(notification.get("subject", "")).lower()

        # Calibration Start Events -> Trigger/Initialize 5-Stack Session
        is_calib_start = (
            subj in ("calibration.should_start", "calibration.started", "calibration.setup")
            or (subj.startswith("calibration.") and subj.endswith((".should_start", ".started", ".setup")))
        )

        # Interrupting Events -> Reset incomplete 5-Stack session
        is_interrupting = (
            subj in (
                "calibration.should_stop",
                "calibration.stopped",
                "calibration.failed",
                "calibration.set_enabled",
                "accuracy_visualizer.outlier_threshold_changed",
            )
        )

        if is_calib_start:
            self._init_5stack_session(reason=f"Calibration Start Event ('{subj}')")
        elif is_interrupting:
            self._reset_5stack_if_incomplete(reason=f"Interrupting Event ('{subj}')")

        if self.__handle_calibration_setup_notification(notification):
            return

        if self.__handle_calibration_result_notification(notification):
            return

        if self.__handle_validation_data_notification(notification):
            return

        if notification.get("subject") == "accuracy_visualizer.outlier_threshold_changed":
            if self.recent_input.is_complete:
                self.recalculate()

        if notification.get("subject") == "calibration.set_enabled":
            enabled = bool(notification.get("enabled", True))
            if hasattr(self, "g_pool") and self.g_pool is not None:
                self.g_pool.enable_calibration = enabled
            if self.recent_input.is_complete:
                if self.recent_input.gazer_params is not None:
                    self.recent_input.gazer_params["enable_calibration"] = enabled
                self.recalculate()

    def __handle_calibration_setup_notification(self, note_dict: dict) -> bool:
        try:
            note = CalibrationSetupNotification.from_dict(note_dict)
        except ValueError:
            return False

        self.recent_input.update(
            gazer_class_name=note.gazer_class_name,
            pupil_list=note.calib_data["pupil_list"],
            ref_list=note.calib_data["ref_list"],
        )
        return True

    def __handle_calibration_result_notification(self, note_dict: dict) -> bool:
        try:
            note = CalibrationResultNotification.from_dict(note_dict)
        except ValueError:
            return False

        self.calibration_counter += 1
        self.validation_counter_under_current_calib = 0
        self._current_eval_mode = "calibration"
        self._stack_calib_id = self.calibration_counter

        self.recent_input.update(
            gazer_class_name=note.gazer_class_name,
            gazer_params=note.params,
        )

        self.recalculate()
        return True

    def __handle_validation_data_notification(self, note_dict: dict) -> bool:
        try:
            note = ChoreographyNotification.from_dict(note_dict)
            assert note.mode == ChoreographyMode.VALIDATION
            assert note.action == ChoreographyAction.DATA
        except (AssertionError, ValueError):
            return False

        self.validation_counter_under_current_calib += 1
        self.total_validation_counter += 1
        self._current_eval_mode = "validation"

        self.recent_input.clear()
        self.recent_input.update(
            gazer_class_name=note_dict["gazer_class_name"],
            gazer_params=note_dict["gazer_params"],
            pupil_list=note_dict["pupil_list"],
            ref_list=note_dict["ref_list"],
        )

        self.recalculate()
        return True

    def recalculate(self):
        NOT_ENOUGH_DATA_COLLECTED_ERR_MSG = (
            "Did not collect enough data to estimate gaze mapping accuracy."
        )

        if not self.recent_input.is_complete:
            logger.warning(NOT_ENOUGH_DATA_COLLECTED_ERR_MSG)
            return

        results = self.calc_acc_prec_errlines(
            gazer_class=self.recent_input.gazer_class,
            g_pool=self.g_pool,
            gazer_params=self.recent_input.gazer_params,
            pupil_list=self.recent_input.pupil_list,
            ref_list=self.recent_input.ref_list,
            intrinsics=self.g_pool.capture.intrinsics,
            outlier_threshold=self.outlier_threshold,
            succession_threshold=self.succession_threshold,
        )

        # Extract experiment metadata for structured reporting
        pupil_samples = self.recent_input.pupil_list
        active_model = "Unknown Model"
        if pupil_samples and len(pupil_samples) > 0:
            active_model = pupil_samples[0].get("method", "Unknown Model")

        g_params = self.recent_input.gazer_params or {}
        calib_enabled = bool(g_params.get("enable_calibration", getattr(self.g_pool, "enable_calibration", True)))

        calib_id = g_params.get("calibration_id", getattr(self.g_pool, "calibration_counter", self._stack_calib_id or (1 if calib_enabled else 0)))
        val_round = g_params.get("validation_round", getattr(self.g_pool, "validation_counter_under_current_calib", self.validation_counter_under_current_calib if self.validation_counter_under_current_calib > 0 else 1))
        total_val = g_params.get("total_validation_count", getattr(self.g_pool, "total_validation_counter", self.total_validation_counter if self.total_validation_counter > 0 else 1))

        calib_pattern = g_params.get("calibration_pattern", getattr(self.g_pool, "last_calib_pattern", "12-Point (4x3 Dense Grid / New)"))
        calib_pts = g_params.get("calibration_points_count", getattr(self.g_pool, "last_calib_points", 12))

        val_pattern = g_params.get("validation_pattern", getattr(self.g_pool, "last_val_pattern", "Diamond (Inward Cross / Default)"))
        val_pts = g_params.get("validation_points_count", getattr(self.g_pool, "last_val_points", 4))
        sample_dur = g_params.get("sample_duration", getattr(self.g_pool, "sample_duration", 60))

        intr = getattr(self.g_pool.capture, "intrinsics", None)
        res_str = str(intr.resolution) if intr and hasattr(intr, "resolution") else "(1280, 720)"

        is_calc_valid = results.is_valid and not np.isnan(results.accuracy.result)
        acc_val = results.accuracy.result if is_calc_valid else np.nan
        prec_val = results.precision.result if (results.is_valid and not np.isnan(results.precision.result)) else np.nan

        if not is_calc_valid:
            self.accuracy = None
            logger.warning("Not enough data available for angular accuracy calculation.")
        else:
            self.accuracy = results.accuracy
            logger.info(f"Angular accuracy: {results.accuracy.result:.3f} degrees")

        if not (results.is_valid and not np.isnan(results.precision.result)):
            self.precision = None
            logger.warning("Not enough data available for angular precision calculation.")
        else:
            self.precision = results.precision
            logger.info(f"Angular precision: {results.precision.result:.3f} degrees")

        # Format metrics strings
        acc_str = f"{acc_val:.3f}°" if not np.isnan(acc_val) else "FAIL (All Outliers / No Correlation)"
        prec_str = f"{prec_val:.3f}°" if not np.isnan(prec_val) else "FAIL (N/A)"
        used_acc = results.accuracy.num_used if is_calc_valid else 0
        total_acc = results.accuracy.num_total if is_calc_valid else (len(self.recent_input.ref_list) if self.recent_input.ref_list else 0)
        used_prec = results.precision.num_used if (results.is_valid and not np.isnan(results.precision.result)) else 0
        total_prec = results.precision.num_total if (results.is_valid and not np.isnan(results.precision.result)) else 0

        if self._current_eval_mode == "calibration":
            if not np.isnan(acc_val):
                self.last_calib_accuracy = acc_val
                self._stack_calib_score = acc_val
            if not np.isnan(prec_val):
                self.last_calib_precision = prec_val

            report_msg = (
                f"\n{'='*80}\n"
                f"🎯 [EXPERIMENT REPORT] CALIBRATION #{calib_id}\n"
                f"{'-'*80}\n"
                f"  • Model & Setup:\n"
                f"    - Active 2D Model:       {active_model}\n"
                f"    - Calibration Status:    ENABLED (New Calibration Fitted)\n"
                f"    - Calibration Pattern:   {calib_pattern} [{calib_pts} Points]\n"
                f"    - Sample Duration:       {sample_dur} frames/point (~{sample_dur/60.0:.1f}s)\n"
                f"  • Experimental Conditions:\n"
                f"    - World Resolution:      {res_str}\n"
                f"    - Outlier Threshold:     {self.outlier_threshold}°\n"
                f"  • Calibration Metrics:\n"
                f"    - Angular Accuracy:      {acc_str}  (Used Samples: {used_acc}/{total_acc})\n"
                f"    - Angular Precision:     {prec_str}  (Used Samples: {used_prec}/{total_prec})\n"
                f"    - Status:                {'CALIBRATION COMPLETED ✅' if is_calc_valid else 'CALIBRATION FAILED ❌'}\n"
                f"{'='*80}"
            )
            print(report_msg, flush=True)

        elif self._current_eval_mode == "validation":
            if calib_enabled:
                calib_mode_str = "ENABLED (Calibrated Gaze Mapping)"
                parent_info = f"Calibration #{calib_id}"
                round_info = f"Round {val_round} under Calibration #{calib_id} (Total Validation #{total_val})"
            else:
                calib_mode_str = "DISABLED (Bypass Raw Pupil Vector / Uncalibrated)"
                parent_info = "None (Raw Baseline)"
                round_info = f"Round {val_round} (Uncalibrated Baseline)"

            report_msg = (
                f"\n{'='*80}\n"
                f"📊 [EXPERIMENT REPORT] VALIDATION #{val_round} (under Calibration #{calib_id if calib_enabled else 0})\n"
                f"{'-'*80}\n"
                f"  • Experiment Hierarchy:\n"
                f"    - Active 2D Model:       {active_model}\n"
                f"    - Calibration Mode:      {calib_mode_str}\n"
                f"    - Parent Calibration:    {parent_info}\n"
                f"    - Validation Attempt:    {round_info}\n"
                f"  • Pattern & Target Points:\n"
                f"    - Calibration Pattern:   {calib_pattern} [{calib_pts} Points] ({'Active' if calib_enabled else 'Bypassed'})\n"
                f"    - Validation Pattern:    {val_pattern} [{val_pts} Points]\n"
                f"    - Sample Duration:       {sample_dur} frames/point (~{sample_dur/60.0:.1f}s)\n"
                f"  • Experimental Conditions:\n"
                f"    - World Resolution:      {res_str}\n"
                f"    - Outlier Threshold:     {self.outlier_threshold}°\n"
                f"  • Validation Test Metrics:\n"
                f"    - Angular Accuracy:      {acc_str}  (Used Samples: {used_acc}/{total_acc})\n"
                f"    - Angular Precision:     {prec_str}  (Used Samples: {used_prec}/{total_prec})\n"
                f"    - Status:                {'VALIDATION TEST COMPLETED ✅' if is_calc_valid else 'VALIDATION TEST FAILED ❌'}\n"
                f"{'='*80}"
            )
            print(report_msg, flush=True)

            # 5-Stack Validation Demo Accumulation
            if self.enable_5stack_summary:
                if not self._stack_active:
                    self._stack_active = True
                    self._stack_calib_id = calib_id

                curr_round = len(self._val_stack) + 1
                self._val_stack.append(acc_val)
                self._val_stack_details.append({
                    "round": curr_round,
                    "accuracy": acc_val,
                    "precision": prec_val,
                    "is_fail": not is_calc_valid,
                    "samples_used": used_acc,
                    "samples_total": total_acc,
                })
                self._stack_model_name = active_model

                if len(self._val_stack) < self.stack_target_count:
                    status_text = f"{acc_val:.3f}° ✅" if is_calc_valid else "FAIL ❌"
                    print(
                        f"📌 [5-Stack Demo Progress] Round #{curr_round}/{self.stack_target_count} Stacked: {status_text} "
                        f"({self.stack_target_count - curr_round} rounds remaining until 5-stack statistical summary)\n",
                        flush=True,
                    )
                elif len(self._val_stack) >= self.stack_target_count:
                    self._print_5stack_summary(
                        calib_id=calib_id,
                        val_pattern=val_pattern,
                        val_pts=val_pts,
                        sample_dur=sample_dur,
                    )
                    # Reset stack for the next sequence
                    self._val_stack.clear()
                    self._val_stack_details.clear()

        self.error_lines = results.error_lines
        if results.correlation.is_valid and len(results.correlation.norm_space.shape) == 2 and len(results.correlation.norm_space) >= 2:
            ref_locations = results.correlation.norm_space[1::2, :]
            if len(ref_locations) >= 3:
                try:
                    # requires at least 3 points
                    hull = scipy.spatial.ConvexHull(ref_locations)
                    self.calibration_area = hull.points[hull.vertices, :]
                except scipy.spatial.qhull.QhullError:
                    logger.warning("Calibration area could not be calculated")
                    logger.debug(traceback.format_exc())

    @staticmethod
    def calc_acc_prec_errlines(
        g_pool,
        gazer_class,
        gazer_params,
        pupil_list,
        ref_list,
        intrinsics,
        outlier_threshold,
        succession_threshold=np.cos(np.deg2rad(0.5)),
    ) -> AccuracyPrecisionResult:
        gazer = gazer_class(g_pool, params=gazer_params)
        if isinstance(gazer_params, dict) and "enable_calibration" in gazer_params:
            gazer.enable_calibration = bool(gazer_params["enable_calibration"])
        elif hasattr(g_pool, "enable_calibration"):
            gazer.enable_calibration = bool(g_pool.enable_calibration)

        gaze_pos = gazer.map_pupil_to_gaze(pupil_list)
        ref_pos = ref_list

        try:
            correlation_result = Accuracy_Visualizer.correlate_and_coordinate_transform(
                gaze_pos, ref_pos, intrinsics
            )
            error_lines = correlation_result.norm_space.reshape(-1, 4)
            undistorted_3d = correlation_result.camera_space
        except CorrelationError:
            return AccuracyPrecisionResult.failed()

        # Accuracy is calculated as the average angular
        # offset (distance) (in degrees of visual angle)
        # between fixations locations and the corresponding
        # locations of the fixation targets.

        # Cosine distance of A and B: (A @ B) / (||A|| * ||B||)
        # No need to calculate norms, since A and B are normalized in our case.
        # np.einsum('ij,ij->i', A, B) equivalent to np.diagonal(A @ B.T) but faster.
        angular_err = np.einsum(
            "ij,ij->i", undistorted_3d[::2, :], undistorted_3d[1::2, :]
        )

        # Good values are close to 1. since cos(0) == 1.
        # Therefore we look for values greater than cos(outlier_threshold)
        selected_indices = angular_err > np.cos(np.deg2rad(outlier_threshold))
        selected_samples = angular_err[selected_indices]
        num_used, num_total = selected_samples.shape[0], angular_err.shape[0]

        error_lines = error_lines[selected_indices].reshape(
            -1, 2
        )  # shape: num_used x 2
        accuracy = np.rad2deg(np.arccos(selected_samples.clip(-1.0, 1.0).mean()))
        accuracy_result = CalculationResult(accuracy, num_used, num_total)

        # lets calculate precision:  (RMS of distance of succesive samples.)
        # This is a little rough as we do not compensate headmovements in this test.

        # Precision is calculated as the Root Mean Square (RMS)
        # of the angular distance (in degrees of visual angle)
        # between successive samples during a fixation
        undistorted_3d.shape = -1, 6  # shape: n x 6
        succesive_distances_gaze = np.einsum(
            "ij,ij->i", undistorted_3d[:-1, :3], undistorted_3d[1:, :3]
        )
        succesive_distances_ref = np.einsum(
            "ij,ij->i", undistorted_3d[:-1, 3:], undistorted_3d[1:, 3:]
        )

        # if the ref distance is to big we must have moved to a new fixation or there is
        # headmovement, if the gaze dis is to big we can assume human error
        # both times gaze data is not valid for this mesurement
        selected_indices = np.logical_and(
            succesive_distances_gaze > succession_threshold,
            succesive_distances_ref > succession_threshold,
        )
        succesive_distances = succesive_distances_gaze[selected_indices]
        num_used, num_total = (
            succesive_distances.shape[0],
            succesive_distances_gaze.shape[0],
        )
        precision = np.sqrt(
            np.mean(np.rad2deg(np.arccos(succesive_distances.clip(-1.0, 1.0))) ** 2)
        )
        precision_result = CalculationResult(precision, num_used, num_total)

        return AccuracyPrecisionResult(
            accuracy_result, precision_result, error_lines, correlation_result
        )

    @staticmethod
    def correlate_and_coordinate_transform(
        gaze_pos, ref_pos, intrinsics
    ) -> CorrelatedAndCoordinateTransformedResult:
        # reuse closest_matches_monocular to correlate one label to each prediction
        # correlated['ref']: prediction, correlated['pupil']: label location
        # NOTE the switch of the ref and pupil keys! This effects mostly hmd data.
        gaze_list = list(gaze_pos) if not isinstance(gaze_pos, list) else gaze_pos
        ref_list = list(ref_pos) if not isinstance(ref_pos, list) else ref_pos

        correlated = closest_matches_monocular(gaze_list, ref_list)
        if not correlated:
            for relaxed_dispersion in (0.2, 0.5, 1.0, 2.0):
                correlated = closest_matches_monocular(
                    gaze_list, ref_list, max_dispersion=relaxed_dispersion
                )
                if correlated:
                    break

        # [[pred.x, pred.y, label.x, label.y], ...], shape: n x 4
        if not correlated:
            raise CorrelationError("No correlation possible")

        try:
            return Accuracy_Visualizer._coordinate_transform_ref_in_norm_space(
                correlated, intrinsics
            )
        except KeyError as err:
            if "norm_pos" in err.args:
                return Accuracy_Visualizer._coordinate_transform_ref_in_camera_space(
                    correlated, intrinsics
                )
            else:
                raise

    @staticmethod
    def _coordinate_transform_ref_in_norm_space(
        correlated, intrinsics
    ) -> CorrelatedAndCoordinateTransformedResult:
        width, height = intrinsics.resolution
        locations_norm = np.array(
            [(*e["ref"]["norm_pos"], *e["pupil"]["norm_pos"]) for e in correlated]
        )
        locations_image = locations_norm.copy()  # n x 4
        locations_image[:, ::2] *= width
        locations_image[:, 1::2] = (1.0 - locations_image[:, 1::2]) * height
        locations_image.shape = -1, 2
        locations_norm.shape = -1, 2
        locations_camera = intrinsics.unprojectPoints(locations_image, normalize=True)
        return CorrelatedAndCoordinateTransformedResult(
            locations_norm, locations_image, locations_camera
        )

    @staticmethod
    def _coordinate_transform_ref_in_camera_space(
        correlated, intrinsics
    ) -> CorrelatedAndCoordinateTransformedResult:
        width, height = intrinsics.resolution
        locations_mixed = np.array(
            # NOTE: This looks incorrect, but is actually correct. The switch comes from
            # using closest_matches_monocular() above with switched arguments.
            [(*e["ref"]["norm_pos"], *e["pupil"]["mm_pos"]) for e in correlated]
        )  # n x 5
        pupil_norm = locations_mixed[:, 0:2]  # n x 2
        pupil_image = pupil_norm.copy()
        pupil_image[:, 0] *= width
        pupil_image[:, 1] = (1.0 - pupil_image[:, 1]) * height
        pupil_camera = intrinsics.unprojectPoints(pupil_image, normalize=True)  # n x 3

        ref_camera = locations_mixed[:, 2:5]  # n x 3
        ref_camera /= np.linalg.norm(ref_camera, axis=1, keepdims=True)
        ref_image = intrinsics.projectPoints(ref_camera)  # n x 2
        ref_norm = ref_image.copy()
        ref_norm[:, 0] /= width
        ref_norm[:, 1] = 1.0 - (ref_norm[:, 1] / height)

        locations_norm = np.hstack([pupil_norm, ref_norm])  # n x 4
        locations_norm.shape = -1, 2

        locations_image = np.hstack([pupil_image, ref_image])  # n x 4
        locations_image.shape = -1, 2

        locations_camera = np.hstack([pupil_camera, ref_camera])  # n x 6
        locations_camera.shape = -1, 3

        return CorrelatedAndCoordinateTransformedResult(
            locations_norm, locations_image, locations_camera
        )

    def gl_display(self):
        import OpenGL.GL as gl
        from pyglui.cygl.utils import RGBA, draw_points_norm, draw_polyline_norm

        if self.vis_mapping_error and self.error_lines is not None:
            draw_polyline_norm(
                self.error_lines, color=RGBA(1.0, 0.5, 0.0, 0.5), line_type=gl.GL_LINES
            )
            draw_points_norm(
                self.error_lines[1::2], size=3, color=RGBA(0.0, 0.5, 0.5, 0.5)
            )
            draw_points_norm(
                self.error_lines[0::2], size=3, color=RGBA(0.5, 0.0, 0.0, 0.5)
            )
        if self.vis_calibration_area and self.calibration_area is not None:
            draw_polyline_norm(
                self.calibration_area,
                thickness=2.0,
                color=RGBA(0.663, 0.863, 0.463, 0.8),
                line_type=gl.GL_LINE_LOOP,
            )

    def get_init_dict(self):
        return {
            "outlier_threshold": self.outlier_threshold,
            "vis_mapping_error": self.vis_mapping_error,
            "vis_calibration_area": self.vis_calibration_area,
            "enable_5stack_summary": self.enable_5stack_summary,
        }
