"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import errno
import glob
import logging
import multiprocessing
import os
import uuid
from queue import Empty
from shutil import copy2
from time import gmtime, localtime, monotonic, strftime, time

import csv_utils
import psutil
from av_writer import JPEG_Writer, MPEG_Writer, NonMonotonicTimestampError
from file_methods import PLData_Writer, load_object
from gaze_mapping.notifications import (
    CalibrationResultNotification,
    CalibrationSetupNotification,
)
from hotkey import Hotkey
from methods import get_system_info, timer
from ndsi import H264Writer

# from scipy.interpolate import UnivariateSpline
from plugin import System_Plugin_Base
from pupil_recording.info import RecordingInfoFile
from pyglui import ui
from video_capture.ndsi_backend import NDSI_Source

logger = logging.getLogger(__name__)


ENABLE_POINT_SEQUENCE = True
# POINT_SEQUENCE_MODE = "smooth_rieul"  # "smooth_rieul" or "discrete_keypad"
POINT_SEQUENCE_MODE = "discrete_keypad"  # "smooth_rieul" or "discrete_keypad"
POINT_DISPLAY_INTERVAL_SEC = 1.0
SMOOTH_RIEUL_DURATION_SEC = 10.0
POINT_ANIMATION_FPS = 60.0
POINT_RADIUS = 25
TARGET_SCREEN_WIDTH = 2560
TARGET_SCREEN_HEIGHT = 1440
DISPLAY_POINTS = [
    (154, 122),
    (1280, 122),
    (2406, 122),
    (154, 720),
    (1280, 720),
    (2406, 720),
    (154, 1318),
    (1280, 1318),
    (2406, 1318),
]
RIEUL_DISPLAY_POINTS = [
    (154, 122),  # 7
    (1280, 122),  # 8
    (2406, 122),  # 9
    (2406, 720),  # 6
    (1280, 720),  # 5
    (154, 720),  # 4
    (154, 1318),  # 1
    (1280, 1318),  # 2
    (2406, 1318),  # 3
]


def run_point_overlay(
    points,
    mode,
    interval_sec,
    smooth_duration_sec,
    animation_fps,
    radius,
    stop_event,
    log_queue,
):
    """Display a sequence of targets in a dedicated process."""

    def overlay_log(level, message):
        try:
            log_queue.put((level, message))
        except Exception:
            getattr(logger, level)(message)

    window = None
    glfw_initialized = False
    try:
        import math

        import glfw
        from OpenGL.GL import (
            GL_BLEND,
            GL_COLOR_BUFFER_BIT,
            GL_LINES,
            GL_MODELVIEW,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_PROJECTION,
            GL_SRC_ALPHA,
            GL_TRIANGLE_FAN,
            glBegin,
            glBlendFunc,
            glClear,
            glClearColor,
            glColor4f,
            glEnable,
            glEnd,
            glLineWidth,
            glLoadIdentity,
            glMatrixMode,
            glOrtho,
            glVertex2f,
            glViewport,
        )

        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        glfw_initialized = True

        monitor = glfw.get_primary_monitor()
        video_mode = glfw.get_video_mode(monitor) if monitor else None
        if monitor is None or video_mode is None:
            raise RuntimeError("Could not determine the primary screen resolution")

        screen_width = video_mode.size.width
        screen_height = video_mode.size.height
        monitor_x, monitor_y = glfw.get_monitor_pos(monitor)
        if (screen_width, screen_height) != (
            TARGET_SCREEN_WIDTH,
            TARGET_SCREEN_HEIGHT,
        ):
            overlay_log(
                "warning",
                "Screen resolution is %dx%d; expected %dx%d. Display scaling and "
                "multi-monitor layouts may also affect global screen coordinates."
                % (
                    screen_width,
                    screen_height,
                    TARGET_SCREEN_WIDTH,
                    TARGET_SCREEN_HEIGHT,
                ),
            )

        # GLFW window positions use global screen coordinates. Display scaling and
        # multi-monitor layouts can make these differ from physical pixel coordinates.
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        if hasattr(glfw, "MOUSE_PASSTHROUGH"):
            glfw.window_hint(glfw.MOUSE_PASSTHROUGH, glfw.TRUE)

        window = glfw.create_window(
            screen_width, screen_height, "Point overlay", None, None
        )
        if window is None:
            raise RuntimeError("Could not create the point overlay window")

        if hasattr(glfw, "MOUSE_PASSTHROUGH"):
            glfw.set_window_attrib(window, glfw.MOUSE_PASSTHROUGH, glfw.TRUE)
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        glfw.set_window_pos(window, monitor_x, monitor_y)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
        scale_x = framebuffer_width / screen_width
        scale_y = framebuffer_height / screen_height
        draw_radius = radius * min(scale_x, scale_y)

        def draw_target(point_x, point_y):
            center_x = point_x * scale_x
            # OpenGL's origin is at the bottom left, while display points use the
            # top-left corner of the primary monitor as (0, 0).
            center_y = framebuffer_height - point_y * scale_y
            glViewport(0, 0, framebuffer_width, framebuffer_height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, framebuffer_width, 0, framebuffer_height, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            glColor4f(1.0, 0.1, 0.1, 0.95)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(center_x, center_y)
            for segment in range(65):
                angle = 2.0 * math.pi * segment / 64
                glVertex2f(
                    center_x + math.cos(angle) * draw_radius,
                    center_y + math.sin(angle) * draw_radius,
                )
            glEnd()

            cross_radius = max(5.0, draw_radius * 0.4)
            glColor4f(0.0, 0.0, 0.0, 1.0)
            glLineWidth(max(2.0, 3.0 * min(scale_x, scale_y)))
            glBegin(GL_LINES)
            glVertex2f(center_x - cross_radius, center_y)
            glVertex2f(center_x + cross_radius, center_y)
            glVertex2f(center_x, center_y - cross_radius)
            glVertex2f(center_x, center_y + cross_radius)
            glEnd()

        def wait_until(deadline):
            while not stop_event.is_set():
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return True
                glfw.wait_events_timeout(min(remaining, 0.05))
                if glfw.window_should_close(window):
                    return False
            return False

        def present_target(point_x, point_y):
            nonlocal window_shown
            draw_target(point_x, point_y)
            glfw.swap_buffers(window)
            if not window_shown:
                glfw.show_window(window)
                window_shown = True

        overlay_log("info", "Point sequence started")
        point_count = len(points)
        valid_points = []
        for point_index, point in enumerate(points, start=1):
            point_x, point_y = point
            if not (0 <= point_x < screen_width and 0 <= point_y < screen_height):
                overlay_log(
                    "warning",
                    "Point is outside the screen bounds: (%d, %d)" % (point_x, point_y),
                )
                continue
            valid_points.append((point_index, point))

        window_shown = False
        if mode == "discrete_keypad":
            for point_index, (point_x, point_y) in valid_points:
                if stop_event.is_set():
                    return
                present_target(point_x, point_y)
                overlay_log(
                    "info",
                    "Displaying point %d/%d: (%d, %d)"
                    % (point_index, point_count, point_x, point_y),
                )
                if not wait_until(monotonic() + interval_sec):
                    return
        elif mode == "smooth_rieul":
            if len(valid_points) < 2:
                raise RuntimeError(
                    "Smooth rieul mode requires at least two points on the screen"
                )

            path_points = [point for _, point in valid_points]
            cumulative_distances = [0.0]
            for start_point, end_point in zip(path_points, path_points[1:]):
                segment_length = math.hypot(
                    end_point[0] - start_point[0],
                    end_point[1] - start_point[1],
                )
                cumulative_distances.append(
                    cumulative_distances[-1] + segment_length
                )
            total_distance = cumulative_distances[-1]
            if total_distance <= 0:
                raise RuntimeError("Smooth rieul path has no length")

            overlay_log(
                "info",
                "Following smooth rieul path for %.1f seconds" % smooth_duration_sec,
            )
            animation_start = monotonic()
            segment_index = 0
            next_waypoint_log_index = 0
            frame_index = 0
            while not stop_event.is_set():
                elapsed = min(monotonic() - animation_start, smooth_duration_sec)
                distance = total_distance * elapsed / smooth_duration_sec

                while (
                    segment_index < len(path_points) - 2
                    and distance >= cumulative_distances[segment_index + 1]
                ):
                    segment_index += 1

                segment_start_distance = cumulative_distances[segment_index]
                segment_length = (
                    cumulative_distances[segment_index + 1]
                    - segment_start_distance
                )
                segment_progress = (
                    distance - segment_start_distance
                ) / segment_length
                start_x, start_y = path_points[segment_index]
                end_x, end_y = path_points[segment_index + 1]
                point_x = start_x + (end_x - start_x) * segment_progress
                point_y = start_y + (end_y - start_y) * segment_progress
                present_target(point_x, point_y)

                while (
                    next_waypoint_log_index < len(valid_points)
                    and distance
                    >= cumulative_distances[next_waypoint_log_index]
                ):
                    original_index, waypoint = valid_points[next_waypoint_log_index]
                    overlay_log(
                        "info",
                        "Displaying point %d/%d: (%d, %d)"
                        % (original_index, point_count, waypoint[0], waypoint[1]),
                    )
                    next_waypoint_log_index += 1

                if elapsed >= smooth_duration_sec:
                    break
                frame_index += 1
                next_frame_time = animation_start + min(
                    frame_index / animation_fps, smooth_duration_sec
                )
                if not wait_until(next_frame_time):
                    return
        else:
            raise RuntimeError("Unknown point sequence mode: %s" % mode)

        if not stop_event.is_set():
            overlay_log("info", "Point sequence completed")
            log_queue.put(("sequence_completed", None))
    except Exception as error:
        overlay_log("error", "Failed to start point overlay: %s" % error)
        raise
    finally:
        if window is not None:
            glfw.destroy_window(window)
        if glfw_initialized:
            glfw.terminate()
        try:
            log_queue.close()
            log_queue.join_thread()
        except Exception:
            pass


def get_auto_name():
    return strftime("%Y_%m_%d", localtime())


def available_gb(path):
    num_avail_gb = psutil.disk_usage(path).free / 1e9
    # logger.debug('{} has {:.2f} GB available'.format(path, num_avail_gb))
    return num_avail_gb


class Recorder(System_Plugin_Base):
    """Capture Recorder"""

    icon_chr = chr(0xE04B)
    icon_font = "pupil_icons"
    warning_low_disk_space_th = 5.0  # threshold in GB
    stop_rec_low_disk_space_th = 1.0  # threshold in GB

    def __init__(
        self,
        g_pool,
        session_name=get_auto_name(),
        rec_root_dir=None,
        user_info={"name": "", "additional_field": "change_me"},
        info_menu_conf={},
        show_info_menu=False,
        record_world=True,
        record_eye=True,
        raw_jpeg=True,
    ):
        super().__init__(g_pool)
        # update name if it was autogenerated.
        if session_name.startswith("20") and len(session_name) == 10:
            session_name = get_auto_name()

        base_dir = self.g_pool.user_dir.rsplit(os.path.sep, 1)[0]
        default_rec_root_dir = os.path.join(base_dir, "recordings")

        logger.warning("g_pool.user_dir = %s", self.g_pool.user_dir)
        logger.warning("default recording dir = %s", default_rec_root_dir)

        if (
            rec_root_dir
            and rec_root_dir != default_rec_root_dir
            and self.verify_path(rec_root_dir)
        ):
            self.rec_root_dir = rec_root_dir
        else:
            try:
                os.makedirs(default_rec_root_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    logger.error("Could not create Rec dir")
                    raise e
            else:
                logger.info(f'Created standard Rec dir at "{default_rec_root_dir}"')
            self.rec_root_dir = default_rec_root_dir

        self.raw_jpeg = raw_jpeg
        self.order = 0.9
        self.record_world = record_world
        self.record_eye = record_eye
        self.session_name = session_name
        self.running = False
        self.menu = None
        self.button = None
        self.point_sequence_process = None
        self.point_sequence_stop_event = None
        self.point_sequence_log_queue = None
        self.point_sequence_should_stop_recording = False

        self.user_info = user_info
        self.show_info_menu = show_info_menu
        self.info_menu = None
        self.info_menu_conf = info_menu_conf

        self.low_disk_space_thumb = None
        check_timer = timer(1.0)
        self.check_space = lambda: next(check_timer)

    def get_init_dict(self):
        return {
            "record_world": self.record_world,
            "record_eye": self.record_eye,
            "session_name": self.session_name,
            "user_info": self.user_info,
            "info_menu_conf": self.info_menu_conf,
            "show_info_menu": self.show_info_menu,
            "rec_root_dir": self.rec_root_dir,
            "raw_jpeg": self.raw_jpeg,
        }

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Recorder"
        self.menu_icon.order = 0.29

        self.menu.append(
            ui.Info_Text(
                'Pupil recordings are saved like this: "path_to_recordings/recording_session_name/nnn" where "nnn" is an increasing number to avoid overwrites. You can use "/" in your session name to create subdirectories.'
            )
        )
        self.menu.append(
            ui.Info_Text(
                'Recordings are saved to "~/pupil_recordings". You can change the path here but note that invalid input will be ignored.'
            )
        )
        self.menu.append(
            ui.Text_Input(
                "rec_root_dir",
                self,
                setter=self.set_rec_root_dir,
                label="Path to recordings",
            )
        )
        self.menu.append(
            ui.Text_Input(
                "session_name",
                self,
                setter=self.set_session_name,
                label="Recording session name",
            )
        )
        self.menu.append(
            ui.Switch(
                "show_info_menu",
                self,
                on_val=True,
                off_val=False,
                label="Request additional user info",
            )
        )
        self.menu.append(
            ui.Selector(
                "raw_jpeg",
                self,
                selection=[True, False],
                labels=["bigger file, less CPU", "smaller file, more CPU"],
                label="Compression",
            )
        )
        self.menu.append(
            ui.Info_Text(
                "Enable/disable recording of eye and world video with these toggles"
            )
        )
        self.menu.append(
            ui.Switch(
                "record_world",
                self,
                on_val=True,
                off_val=False,
                label="Record world video",
            )
        )
        self.menu.append(
            ui.Switch(
                "record_eye",
                self,
                on_val=True,
                off_val=False,
                label="Record eye videos",
            )
        )
        self.button = ui.Thumb(
            "running",
            self,
            setter=self.toggle,
            label="R",
            hotkey=Hotkey.RECORDER_RUNNING_TOGGLE_CAPTURE_HOTKEY(),
        )
        self.button.on_color[:] = (1, 0.0, 0.0, 0.8)
        self.g_pool.quickbar.insert(2, self.button)

        self.low_disk_space_thumb = ui.Thumb(
            "low_disk_warn", label="!", getter=lambda: True, setter=lambda x: None
        )
        self.low_disk_space_thumb.on_color[:] = (1, 0.0, 0.0, 0.8)
        self.low_disk_space_thumb.status_text = "Low disk space"

    def deinit_ui(self):
        if self.low_disk_space_thumb in self.g_pool.quickbar:
            self.g_pool.quickbar.remove(self.low_disk_space_thumb)
        self.g_pool.quickbar.remove(self.button)
        self.button = None
        self.remove_menu()

    def toggle(self, _=None):
        if self.running:
            self.notify_all({"subject": "recording.should_stop"})
            self.notify_all(
                {"subject": "recording.should_stop", "remote_notify": "all"}
            )
        else:
            self.notify_all(
                {"subject": "recording.should_start", "session_name": self.session_name}
            )
            self.notify_all(
                {
                    "subject": "recording.should_start",
                    "session_name": self.session_name,
                    "remote_notify": "all",
                }
            )

    def on_notify(self, notification):
        """Handles recorder notifications

        Reacts to notifications:
            ``recording.should_start``: Starts a new recording session.
                fields:
                - 'session_name' change session name
                    start with `/` to ingore the rec base dir and start from root instead.
                - `record_eye` boolean that indicates recording of the eyes, defaults to current setting
            ``recording.should_stop``: Stops current recording session

        Emits notifications:
            ``recording.started``: New recording session started
            ``recording.stopped``: Current recording session stopped

        Args:
            notification (dictionary): Notification dictionary
        """
        # notification wants to be recorded
        if notification.get("record", False) and self.running:
            if "timestamp" not in notification:
                logger.error("Notification without timestamp will not be saved.")
                notification["timestamp"] = self.g_pool.get_timestamp()
            # else:
            notification["topic"] = "notify." + notification["subject"]
            try:
                writer = self.pldata_writers["notify"]
            except KeyError:
                writer = PLData_Writer(self.rec_path, "notify")
                self.pldata_writers["notify"] = writer
            writer.append(notification)

        elif notification["subject"] == "recording.should_start":
            if self.running:
                logger.info("Recording already running!")
            else:
                self.record_eye = notification.get("record_eye", self.record_eye)
                if notification.get("session_name", ""):
                    self.set_session_name(notification["session_name"])
                self.start()

        elif notification["subject"] == "recording.should_stop":
            if self.running:
                self.stop()
            else:
                logger.debug("Recording already stopped!")

    def get_rec_time_str(self):
        rec_time = gmtime(time() - self.start_time)
        return strftime("%H:%M:%S", rec_time)

    def start(self):
        self.start_time = time()
        start_time_synced = self.g_pool.get_timestamp()

        if isinstance(self.g_pool.capture, NDSI_Source):
            # If the user did not enable TimeSync, the timestamps will be way off and
            # the recording code will crash. We check the difference between the last
            # frame's time and the start_time_synced and if this does not match, we stop
            # the recording and show a warning instead.
            TIMESTAMP_ERROR_THRESHOLD = 5.0
            frame = self.g_pool.capture._recent_frame
            if frame is None:
                logger.error(
                    "Your connection does not seem to be stable enough for "
                    "recording Pupil Mobile via WiFi. We recommend recording "
                    "on the phone."
                )
                return
            if abs(frame.timestamp - start_time_synced) > TIMESTAMP_ERROR_THRESHOLD:
                logger.error(
                    "Pupil Mobile stream is not in sync. Aborting recording."
                    " Enable the Time Sync plugin and try again."
                )
                return

        session = os.path.join(self.rec_root_dir, self.session_name)
        try:
            os.makedirs(session, exist_ok=True)
            logger.debug(f"Created new recordings session dir {session}")
        except OSError:
            logger.error(
                "Could not start recording. Session dir {} not writable.".format(
                    session
                )
            )
            return

        self.pldata_writers = {}
        self.frame_count = 0
        self.running = True
        self.menu.read_only = True
        recording_uuid = uuid.uuid4()

        # set up self incrementing folder within session folder
        point_sequence_mode = (
            POINT_SEQUENCE_MODE if ENABLE_POINT_SEQUENCE else "disabled"
        )
        point_sequence_mode = "".join(
            character
            if character.isalnum() or character in ("-", "_")
            else "_"
            for character in point_sequence_mode
        )
        if not point_sequence_mode:
            point_sequence_mode = "unknown"

        counter = 0
        while True:
            recording_number = f"{counter:03d}"
            if glob.glob(os.path.join(session, f"{recording_number}*")):
                logger.debug(
                    "Recording number %s already exists, incrementing counter",
                    recording_number,
                )
                counter += 1
                continue

            recording_dir_name = f"{recording_number}_{point_sequence_mode}"
            self.rec_path = os.path.join(session, recording_dir_name, "")
            try:
                os.mkdir(self.rec_path)
                logger.debug(f"Created new recording dir {self.rec_path}")
                break
            except FileExistsError:
                logger.debug(
                    "We dont want to overwrite data, incrementing counter & trying to make new data folder"
                )
                counter += 1
            except PermissionError:
                logger.error(
                    "No sufficient permissions to create new recording at "
                    f"{self.rec_path}"
                )
                self.running = False
                self.menu.read_only = False

                return

        self.meta_info = RecordingInfoFile.create_empty_file(self.rec_path)
        self.meta_info.recording_software_name = (
            RecordingInfoFile.RECORDING_SOFTWARE_NAME_PUPIL_CAPTURE
        )
        self.meta_info.recording_software_version = str(self.g_pool.version)
        self.meta_info.recording_name = self.session_name
        self.meta_info.start_time_synced_s = start_time_synced
        self.meta_info.start_time_system_s = self.start_time
        self.meta_info.recording_uuid = recording_uuid
        self.meta_info.system_info = get_system_info()

        if self.record_world:
            video_path = os.path.join(self.rec_path, "world.mp4")
            if self.raw_jpeg and self.g_pool.capture.jpeg_support:
                self.writer = JPEG_Writer(video_path, start_time_synced)
            elif hasattr(self.g_pool.capture._recent_frame, "h264_buffer"):
                self.writer = H264Writer(
                    video_path,
                    self.g_pool.capture.frame_size[0],
                    self.g_pool.capture.frame_size[1],
                    int(self.g_pool.capture.frame_rate),
                )
            else:
                self.writer = MPEG_Writer(video_path, start_time_synced)

        calibration_data_notification_classes = [
            CalibrationSetupNotification,
            CalibrationResultNotification,
        ]
        writer = PLData_Writer(self.rec_path, "notify")

        for note_class in calibration_data_notification_classes:
            try:
                file_path = os.path.join(self.g_pool.user_dir, note_class.file_name())
                note = note_class.from_dict(load_object(file_path))
                note_dict = note.as_dict()

                note_dict["topic"] = "notify." + note_dict["subject"]
                writer.append(note_dict)
            except FileNotFoundError:
                continue

        self.pldata_writers["notify"] = writer

        if self.show_info_menu:
            self.open_info_menu()
        logger.info("Started Recording.")
        self.notify_all(
            {
                "subject": "recording.started",
                "rec_path": self.rec_path,
                "session_name": self.session_name,
                "record_eye": self.record_eye,
                "compression": self.raw_jpeg,
                "start_time_synced": float(start_time_synced),
            }
        )
        self.start_point_sequence()

    def _drain_point_sequence_logs(self):
        if self.point_sequence_log_queue is None:
            return
        while True:
            try:
                level, message = self.point_sequence_log_queue.get_nowait()
            except Empty:
                break
            except (EOFError, OSError, ValueError):
                break
            if level == "sequence_completed":
                self.point_sequence_should_stop_recording = True
                continue
            getattr(logger, level, logger.info)(message)

    def _clear_point_sequence_resources(self):
        self._drain_point_sequence_logs()
        if (
            self.point_sequence_process is not None
            and self.point_sequence_process.pid is not None
        ):
            self.point_sequence_process.join(timeout=0)
        if self.point_sequence_log_queue is not None:
            self.point_sequence_log_queue.close()
            self.point_sequence_log_queue.join_thread()
        self.point_sequence_process = None
        self.point_sequence_stop_event = None
        self.point_sequence_log_queue = None

    def _update_point_sequence_state(self):
        self._drain_point_sequence_logs()
        process = self.point_sequence_process
        if process is None or process.is_alive():
            return
        exitcode = process.exitcode
        self._clear_point_sequence_resources()
        if exitcode != 0:
            logger.error("Point overlay process terminated unexpectedly")

    def start_point_sequence(self):
        if not ENABLE_POINT_SEQUENCE:
            return

        self.point_sequence_should_stop_recording = False
        self._update_point_sequence_state()
        if (
            self.point_sequence_process is not None
            and self.point_sequence_process.is_alive()
        ):
            return

        try:
            context = multiprocessing.get_context()
            self.point_sequence_stop_event = context.Event()
            self.point_sequence_log_queue = context.Queue()
            self.point_sequence_process = context.Process(
                name="Point Overlay",
                target=run_point_overlay,
                args=(
                    (
                        RIEUL_DISPLAY_POINTS
                        if POINT_SEQUENCE_MODE == "smooth_rieul"
                        else DISPLAY_POINTS
                    ),
                    POINT_SEQUENCE_MODE,
                    POINT_DISPLAY_INTERVAL_SEC,
                    SMOOTH_RIEUL_DURATION_SEC,
                    POINT_ANIMATION_FPS,
                    POINT_RADIUS,
                    self.point_sequence_stop_event,
                    self.point_sequence_log_queue,
                ),
            )
            self.point_sequence_process.daemon = True
            self.point_sequence_process.start()
        except Exception as error:
            logger.error("Failed to start point overlay: %s", error)
            self._clear_point_sequence_resources()

    def stop_point_sequence(self):
        process = self.point_sequence_process
        if process is None:
            self.point_sequence_should_stop_recording = False
            return

        if self.point_sequence_stop_event is not None:
            self.point_sequence_stop_event.set()
        process.join(timeout=0.5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        self._clear_point_sequence_resources()
        self.point_sequence_should_stop_recording = False

    def open_info_menu(self):
        self.info_menu = ui.Growing_Menu(
            "additional Recording Info", size=(300, 300), pos=(300, 300)
        )
        self.info_menu.configuration = self.info_menu_conf

        def populate_info_menu():
            self.info_menu.elements[:-2] = []
            for name in self.user_info.keys():
                self.info_menu.insert(0, ui.Text_Input(name, self.user_info))

        def set_user_info(new_string):
            self.user_info = new_string
            populate_info_menu()

        populate_info_menu()
        self.info_menu.append(
            ui.Info_Text(
                'Use the *user info* field to add/remove additional fields and their values. The format must be a valid Python dictionary. For example -- {"key":"value"}. You can add as many fields as you require. Your custom fields will be saved for your next session.'
            )
        )
        self.info_menu.append(
            ui.Text_Input("user_info", self, setter=set_user_info, label="User info")
        )
        self.g_pool.gui.append(self.info_menu)

    def close_info_menu(self):
        if self.info_menu:
            self.info_menu_conf = self.info_menu.configuration
            self.g_pool.gui.remove(self.info_menu)
            self.info_menu = None

    def recent_events(self, events):
        self._update_point_sequence_state()
        if self.running and self.point_sequence_should_stop_recording:
            self.point_sequence_should_stop_recording = False
            self.toggle()

        if self.check_space():
            disk_space = available_gb(self.rec_root_dir)
            if (
                disk_space < self.warning_low_disk_space_th
                and self.low_disk_space_thumb not in self.g_pool.quickbar
            ):
                self.g_pool.quickbar.append(self.low_disk_space_thumb)
            elif (
                disk_space >= self.warning_low_disk_space_th
                and self.low_disk_space_thumb in self.g_pool.quickbar
            ):
                self.g_pool.quickbar.remove(self.low_disk_space_thumb)

            if self.running and disk_space <= self.stop_rec_low_disk_space_th:
                self.stop()
                logger.error("Recording was stopped due to low disk space!")

        if self.running:
            for key, data in events.items():
                if key not in ("dt", "depth_frame") and not key.startswith("frame"):
                    try:
                        writer = self.pldata_writers[key]
                    except KeyError:
                        writer = PLData_Writer(self.rec_path, key)
                        self.pldata_writers[key] = writer
                    writer.extend(data)
            if self.record_world and "frame" in events:
                frame = events["frame"]
                try:
                    self.writer.write_video_frame(frame)
                    self.frame_count += 1
                except NonMonotonicTimestampError as e:
                    logger.error(
                        "Recorder received non-monotonic timestamp!"
                        " Stopping the recording!"
                    )
                    logger.debug(str(e))
                    self.notify_all({"subject": "recording.should_stop"})
                    self.notify_all(
                        {"subject": "recording.should_stop", "remote_notify": "all"}
                    )
            # # cv2.putText(frame.img, "Frame %s"%self.frame_count,(200,200), cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,100))

            self.button.status_text = self.get_rec_time_str()

    def stop(self):
        self.stop_point_sequence()
        duration_s = self.g_pool.get_timestamp() - self.meta_info.start_time_synced_s

        if self.record_world:
            # explicit release of VideoWriter
            try:
                self.writer.release()
            except (RuntimeError, FileNotFoundError):
                logger.warning("No world video recorded")
            else:
                logger.debug("Closed media container")
                self.g_pool.capture.intrinsics.save(self.rec_path, custom_name="world")
            finally:
                self.writer = None

        for writer in self.pldata_writers.values():
            writer.close()

        del self.pldata_writers

        surface_definition_file_paths = glob.glob(
            os.path.join(self.g_pool.user_dir, "surface_definitions*")
        )

        if len(surface_definition_file_paths) > 0:
            for source_path in surface_definition_file_paths:
                _, filename = os.path.split(source_path)
                target_path = os.path.join(self.rec_path, filename)
                copy2(source_path, target_path)

        self.meta_info.duration_s = duration_s
        self.meta_info.save_file()

        try:
            with open(
                os.path.join(self.rec_path, "user_info.csv"), "w", newline=""
            ) as csvfile:
                csv_utils.write_key_value_file(csvfile, self.user_info)
        except OSError:
            logger.exception("Could not save userdata. Please report this bug!")

        self.close_info_menu()

        self.running = False
        if self.menu:
            self.menu.read_only = False
            self.button.status_text = ""

        logger.info("Saved Recording.")
        self.notify_all({"subject": "recording.stopped", "rec_path": self.rec_path})

    def cleanup(self):
        """gets called when the plugin get terminated.
        either volunatily or forced.
        """
        if self.running:
            self.stop()
        else:
            self.stop_point_sequence()

    def verify_path(self, val):
        try:
            n_path = os.path.expanduser(val)
            logger.debug("Expanded user path.")
        except Exception:
            n_path = val
        if not n_path:
            logger.warning("Please specify a path.")
            return False
        elif not os.path.isdir(n_path):
            logger.warning("This is not a valid path.")
            return False
        # elif not os.access(n_path, os.W_OK):
        elif not writable_dir(n_path):
            logger.warning(f"Do not have write access to '{n_path}'.")
            return False
        else:
            return n_path

    def set_rec_root_dir(self, val):
        n_path = self.verify_path(val)
        if n_path:
            self.rec_root_dir = n_path

    def set_session_name(self, val):
        if not val:
            self.session_name = get_auto_name()
        else:
            if os.path.sep in val:
                logger.warning(
                    "You session name will create one or more subdirectories"
                )
            self.session_name = val


def writable_dir(n_path):
    try:
        open(os.path.join(n_path, "dummpy_tmp"), "w")
    except OSError:
        return False
    else:
        os.remove(os.path.join(n_path, "dummpy_tmp"))
        return True
