"""Launch the local CPU demo/tests with isolation before application imports.

Only standard-library modules are imported in this bootstrap. The child gets
an explicit environment, a Linux network syscall filter, file-access guards,
and service-import guards before any third-party or project module is loaded.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import errno
import importlib.abc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parent
BLOCKED_MODULES = (
    "dotenv", "cloudinary", "src.api", "src.background_workers",
    "src.services.notification_service", "src.services.database_service",
    "src.services.camera_service", "src.services.camera_system",
    "src.services.video_service",
)
NETWORK_SYSCALLS = (
    "connect", "bind", "listen", "accept", "accept4", "sendto", "sendmsg",
    "sendmmsg", "io_uring_setup", "io_uring_enter", "io_uring_register",
)


def clean_environment(scratch):
    """Build from scratch; never copy os.environ or inherit account settings."""
    scratch = Path(scratch)
    return {
        "PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "FIRE_DEMO_ISOLATED": "1", "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "HEADLESS": "true",
        "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false",
        "YOLO_CONFIG_DIR": str(scratch / "ultralytics"),
        "XDG_CONFIG_HOME": str(scratch / "config"),
        "XDG_CACHE_HOME": str(scratch / "cache"),
        "MPLCONFIGDIR": str(scratch / "matplotlib"),
        "TORCH_HOME": str(scratch / "torch"), "TMPDIR": str(scratch),
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "MODEL_PATH": str(ROOT / "models_weights/firedetect-11s.pt"),
        "VIDEO_SOURCE": str(ROOT / "test_fire_smoke.mp4"),
        "DETECTED_FIRES_DIR": str(scratch / "disabled-output"),
        "DB_NAME": str(scratch / "disabled.sqlite"),
        "TARGET_HEIGHT": "480", "IOU_THRESHOLD": "0.2",
        "MIN_CONFIDENCE": "0.4", "SMOKE_CONFIDENCE": "0.4",
        "FLICKER_THRESHOLD": "15", "FLICKER_BUFFER_SIZE": "10",
        "ALERT_COOLDOWN": "45",
        # FFmpeg's native decoder is also restricted to local protocols.
        "OPENCV_FFMPEG_CAPTURE_OPTIONS": "protocol_whitelist;file,pipe",
    }


def block_network_syscalls():
    """Fail closed unless Linux seccomp also blocks native/child networking."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This isolated demo requires Linux and libseccomp")
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        for name in NETWORK_SYSCALLS:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0:
                result = lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0)
                if result:
                    raise OSError(-result, "Cannot block syscall " + name)
        result = lib.seccomp_load(context)
        if result:
            raise OSError(-result, "Cannot install network syscall filter")
    finally:
        lib.seccomp_release(context)
    # Invalid descriptors prove denial without attempting any connection.
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    if libc.connect(-1, None, 0) != -1 or ctypes.get_errno() != errno.EPERM:
        raise RuntimeError("Network syscall filter verification failed")


class BlockServices(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in BLOCKED_MODULES):
            raise ImportError("Service/configuration import disabled in isolated demo: " + fullname)
        return None


def install_isolation(scratch, output_paths=()):
    """Install and verify protections before adding the project to sys.path."""
    scratch = Path(scratch).resolve()
    if dict(os.environ) != clean_environment(scratch):
        raise RuntimeError("Clean subprocess environment verification failed")
    if any(name in sys.modules for name in BLOCKED_MODULES):
        raise RuntimeError("A disabled service was imported before isolation")
    allowed_outputs = set()
    for value in output_paths:
        path = Path(value).absolute()
        if (ROOT / "demo-output").is_symlink() or path.parent != ROOT / "demo-output" or (
                path.suffix not in {".mp4", ".json"} or path.is_symlink()):
            raise RuntimeError("Output paths must be new local demo-output MP4/JSON files")
        allowed_outputs.add(path)
    block_network_syscalls()

    def check_path(value, write=False):
        if isinstance(value, int) or value is None:
            return
        path = Path(os.fsdecode(value))
        if path.name == ".env" or path.resolve() == ROOT / ".env":
            raise PermissionError("Isolated demo blocks .env file access")
        if write and not path.resolve().is_relative_to(scratch) and path.absolute() not in allowed_outputs:
            raise PermissionError("Isolated demo blocks writes outside its temporary directory and explicit output files")

    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.sendmsg"}:
            raise PermissionError("Networking disabled in isolated demo")
        if event == "open":
            mode, flags = args[1], args[2]
            write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            check_path(args[0], write=write)
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.truncate"}:
            check_path(args[0], write=True)
        elif event in {"os.rename", "os.link", "os.symlink"}:
            check_path(args[0], write=True)
            check_path(args[1], write=True)

    sys.addaudithook(audit)
    sys.meta_path.insert(0, BlockServices())
    # Verify guards against a nonexistent sentinel, never the original file.
    try:
        with (scratch / ".env").open("rb"):
            pass
    except PermissionError:
        pass
    else:
        raise RuntimeError("Dotenv file-access guard verification failed")
    return {"clean_environment": True, "dotenv_loading": False,
            "dotenv_reads_blocked": True, "services_disabled": True,
            "kernel_network_block": True, "explicit_output_files": len(allowed_outputs)}


def prepare_output(prefix="annotated"):
    """Check both media/report ignores before creating the output directory."""
    probes = ["demo-output/.ignore-check.mp4", "demo-output/.ignore-check.json"]
    command = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false",
               "check-ignore", "-v", "--", *probes]
    check = subprocess.run(command, cwd=ROOT, env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                           capture_output=True, text=True, timeout=10)
    matches = {line.split("\t", 1)[1] for line in check.stdout.splitlines() if "\t" in line
               and not line.split("\t", 1)[0].split(":", 2)[-1].startswith("!")}
    if check.returncode or matches != set(probes):
        raise RuntimeError("demo-output media and JSON reports must be Git-ignored before generation")
    print("OUTPUT_IGNORE " + check.stdout.strip().replace("\n", " | "), flush=True)
    directory = ROOT / "demo-output"
    if directory.is_symlink():
        raise RuntimeError("demo-output must not be a symlink")
    directory.mkdir(mode=0o700, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    if prefix not in {"annotated", "diagnostic-before", "diagnostic-after", "diagnostic-current"}:
        raise ValueError("Unsupported output prefix")
    output = directory / (prefix + "-" + stamp + ".mp4")
    report = output.with_suffix(".json")
    if os.path.lexists(output) or os.path.lexists(report):
        raise FileExistsError("Refusing to overwrite an existing demo output")
    return output, report


def configure_libraries(scratch):
    from ultralytics import settings
    settings.update({"sync": False, "runs_dir": str(scratch / "runs"),
                     "weights_dir": str(scratch / "weights"),
                     "datasets_dir": str(scratch / "datasets")})
    if settings["sync"] is not False:
        raise RuntimeError("Ultralytics telemetry could not be disabled")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Isolated, headless local CPU video demo")
    parser.add_argument("--video", default=str(ROOT / "test_fire_smoke.mp4"))
    parser.add_argument("--model", default=str(ROOT / "models_weights/firedetect-11s.pt"))
    parser.add_argument("--screen-model", default=str(ROOT / "yolov8n.pt"))
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-frames", type=int)
    limits.add_argument("--full-video", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--verify-output")
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--diagnose-start", type=float)
    parser.add_argument("--diagnose-end", type=float)
    parser.add_argument("--diagnostic-tag", choices=["before", "after", "current"], default="current")
    parser.add_argument("--isolated-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scratch", help=argparse.SUPPRESS)
    parser.add_argument("--output-path", help=argparse.SUPPRESS)
    parser.add_argument("--report-path", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.diagnose_start is None) != (args.diagnose_end is None):
        parser.error("--diagnose-start and --diagnose-end must be supplied together")
    if args.diagnose_start is not None:
        import math
        if not all(math.isfinite(v) for v in (args.diagnose_start, args.diagnose_end)) or not 0 <= args.diagnose_start < args.diagnose_end:
            parser.error("diagnostic interval must satisfy 0 <= start < end")
        if args.run_tests or args.full_video or args.verify_output:
            parser.error("diagnostic mode cannot be combined with other modes")
        if not args.isolated_worker and not args.save_video:
            parser.error("diagnostic mode requires --save-video")
    if args.max_frames is None and not args.full_video:
        args.max_frames = 10
    if (args.max_frames is not None and args.max_frames < 1) or not 1 <= args.timeout <= 900:
        parser.error("max-frames must be positive and timeout must be between 1 and 900 seconds")
    if args.verify_output and (args.expected_frames is None or args.expected_frames < 1):
        parser.error("--verify-output requires a positive --expected-frames")
    if args.run_tests and (args.save_video or args.full_video or args.verify_output):
        parser.error("--run-tests cannot be combined with video-processing options")
    return args


def main(argv=None):
    args = arguments(argv)
    if not args.isolated_worker:
        output = report = None
        if args.save_video:
            try:
                prefix = "diagnostic-" + args.diagnostic_tag if args.diagnose_start is not None else "annotated"
                output, report = prepare_output(prefix)
            except Exception as error:
                print(json.dumps({"error_type": type(error).__name__, "error": str(error)}), flush=True)
                return 1
        with tempfile.TemporaryDirectory(prefix="fire-local-video-", dir="/tmp") as directory:
            hard_deadline = time.monotonic() + args.timeout
            grace = min(5.0, args.timeout / 10)
            command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                       "--isolated-worker", "--scratch", directory,
                       "--video", str(Path(args.video).absolute()),
                       "--model", str(Path(args.model).absolute()),
                       "--screen-model", str(Path(args.screen_model).absolute()),
                       "--timeout", str(args.timeout),
                       "--deadline", str(hard_deadline - grace)]
            if args.full_video:
                command.append("--full-video")
            else:
                command.extend(["--max-frames", str(args.max_frames)])
            if output is not None:
                command.extend(["--output-path", str(output), "--report-path", str(report)])
            if args.run_tests:
                command.append("--run-tests")
            if args.verify_output:
                command.extend(["--verify-output", str(Path(args.verify_output).absolute()),
                                "--expected-frames", str(args.expected_frames)])
            if args.diagnose_start is not None:
                command.extend(["--diagnose-start", str(args.diagnose_start),
                                "--diagnose-end", str(args.diagnose_end),
                                "--diagnostic-tag", args.diagnostic_tag])
            process = subprocess.Popen(command, cwd=directory, env=clean_environment(directory), close_fds=True)
            try:
                return process.wait(timeout=max(0.001, hard_deadline - grace - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()
                killed = False
                try:
                    process.wait(timeout=max(0.001, hard_deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    killed = True
                print(json.dumps({"error": "Demo/test subprocess timed out", "completed": False,
                                  "timeout_seconds": args.timeout, "forced_kill": killed,
                                  "output_path": str(output) if output else None,
                                  "partial_report_path": str(report) if report else None,
                                  "warning": "Any saved video is partial; forced termination can leave it unreadable"}), flush=True)
                return 124
    try:
        if not args.scratch:
            raise RuntimeError("Isolated worker needs a temporary directory")
        scratch = Path(args.scratch).resolve()
        outputs = [value for value in (args.output_path, args.report_path) if value]
        isolation = install_isolation(scratch, outputs)
        stopped = {"value": False}
        signal.signal(signal.SIGTERM, lambda signum, frame: stopped.update(value=True))
        print("ISOLATION " + json.dumps(isolation, sort_keys=True), flush=True)
        configure_libraries(scratch)
        # Only now may the child import project code or tests.
        sys.path.insert(0, str(ROOT))
        if args.run_tests:
            import unittest
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"),
                                                       pattern="test_local_video_demo.py")
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if result.wasSuccessful() else 1
        from src.local_video_demo import local_asset, run_video, verify_video
        if args.diagnose_start is not None:
            if not args.output_path or not args.report_path:
                raise RuntimeError("Diagnostics require new ignored output/report files")
            from src.demo_diagnostics import diagnose_video
            result = diagnose_video(args.video, args.model, args.screen_model,
                                    args.diagnose_start, args.diagnose_end,
                                    args.output_path, args.report_path,
                                    deadline=args.deadline, should_stop=lambda: stopped["value"])
            print("DIAGNOSTIC_RESULT " + json.dumps(result, sort_keys=True), flush=True)
            return 0
        if args.verify_output:
            path = local_asset(args.verify_output, {".mp4"})
            result = verify_video(path, args.expected_frames)
            print("OUTPUT_VERIFICATION " + json.dumps(result, sort_keys=True), flush=True)
            return 0
        result = run_video(args.video, args.model, args.screen_model, args.max_frames,
                           output_path=args.output_path, report_path=args.report_path,
                           deadline=args.deadline, should_stop=lambda: stopped["value"])
        if args.output_path and result["frames_written"]:
            try:
                result["output_verification"] = verify_video(args.output_path, result["frames_written"])
            except Exception as error:
                result.update(completed=False, stop_reason="verification_error",
                              error=type(error).__name__ + ": " + str(error))
            Path(args.report_path).write_text(json.dumps(result, indent=2) + "\n")
        print("DEMO_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 124 if result["stop_reason"] == "timeout" else (0 if result["completed"] else 1)
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__, "error": str(error)}), flush=True)
        return 124 if isinstance(error, TimeoutError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
