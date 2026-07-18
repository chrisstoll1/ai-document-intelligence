from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
BACKEND_SOURCE = ROOT / "backend" / "src"
FRONTEND = ROOT / "frontend"
VITE_SCRIPT = FRONTEND / "node_modules" / "vite" / "bin" / "vite.js"
APP_URL = "http://127.0.0.1:5173"
HEALTH_URL = "http://127.0.0.1:8000/api/health"


class ProcessJob(Protocol):
    def assign(self, process: subprocess.Popen[bytes]) -> None: ...

    def close(self) -> None: ...


class _NullProcessJob:
    def assign(self, process: subprocess.Popen[bytes]) -> None:
        pass

    def close(self) -> None:
        pass


def _create_process_job() -> ProcessJob:
    if os.name != "nt":
        return _NullProcessJob()

    import ctypes
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time_limit", ctypes.c_longlong),
            ("per_job_user_time_limit", ctypes.c_longlong),
            ("limit_flags", wintypes.DWORD),
            ("minimum_working_set_size", ctypes.c_size_t),
            ("maximum_working_set_size", ctypes.c_size_t),
            ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("read_operation_count", ctypes.c_ulonglong),
            ("write_operation_count", ctypes.c_ulonglong),
            ("other_operation_count", ctypes.c_ulonglong),
            ("read_transfer_count", ctypes.c_ulonglong),
            ("write_transfer_count", ctypes.c_ulonglong),
            ("other_transfer_count", ctypes.c_ulonglong),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("basic_limit_information", BasicLimitInformation),
            ("io_info", IoCounters),
            ("process_memory_limit", ctypes.c_size_t),
            ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory_used", ctypes.c_size_t),
            ("peak_job_memory_used", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    class _WindowsProcessJob:
        def __init__(self) -> None:
            self.handle = kernel32.CreateJobObjectW(None, None)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            information = ExtendedLimitInformation()
            information.basic_limit_information.limit_flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                self.handle, 9, ctypes.byref(information), ctypes.sizeof(information)
            ):
                error = ctypes.WinError(ctypes.get_last_error())
                kernel32.CloseHandle(self.handle)
                raise error

        def assign(self, process: subprocess.Popen[bytes]) -> None:
            if not kernel32.AssignProcessToJobObject(self.handle, wintypes.HANDLE(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())

        def close(self) -> None:
            if self.handle:
                kernel32.CloseHandle(self.handle)
                self.handle = None

    return _WindowsProcessJob()


def _node_command() -> str:
    executable = shutil.which("node.exe" if os.name == "nt" else "node")
    if executable is None:
        raise RuntimeError("Node.js is required. Install Node.js 20 or newer before running the app.")
    return executable


def _require_frontend_dependencies() -> None:
    if not VITE_SCRIPT.is_file():
        raise RuntimeError("Frontend dependencies are missing. Run `npm install` in the frontend directory once.")


def _require_available_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use. Stop the existing app run and try again.")


def _wait_for(
    url: str,
    processes: Sequence[tuple[str, subprocess.Popen[bytes]]],
    timeout_seconds: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for name, process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"{name.capitalize()} stopped during startup with exit code {process.returncode}.")
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url}")


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=5)
            return
        except (OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    else:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _backend_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "docintel.api:app",
        "--app-dir",
        str(BACKEND_SOURCE),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--timeout-graceful-shutdown",
        "3",
    ]


def _frontend_command() -> list[str]:
    return [_node_command(), str(VITE_SCRIPT), "--clearScreen", "false"]


def _source_signature(source: Path = BACKEND_SOURCE) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for path in source.rglob("*.py"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        signature.append((str(path.relative_to(source)), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(signature))


def _interrupt_on_break(*_: object) -> None:
    raise KeyboardInterrupt


def _start_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    creation_flags: int,
    process_job: ProcessJob,
) -> subprocess.Popen[bytes]:
    actual_command = command
    stdin = None
    if os.name == "nt":
        actual_command = [sys.executable, str(Path(__file__).resolve()), "--child", *command]
        stdin = subprocess.PIPE
    process = subprocess.Popen(
        actual_command,
        cwd=cwd,
        env=env,
        creationflags=creation_flags,
        stdin=stdin,
    )
    try:
        process_job.assign(process)
        if process.stdin is not None:
            process.stdin.write(b"1")
            process.stdin.close()
    except Exception:
        _stop(process)
        raise
    return process


def _cleanup(
    frontend: subprocess.Popen[bytes] | None,
    backend: subprocess.Popen[bytes] | None,
    process_job: ProcessJob,
) -> None:
    try:
        if frontend is not None:
            _stop(frontend)
    finally:
        try:
            if backend is not None:
                _stop(backend)
        finally:
            process_job.close()


def _run_child(command: list[str]) -> int:
    if not command:
        raise RuntimeError("Child command is missing.")
    if not sys.stdin.buffer.read(1):
        return 1
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, _interrupt_on_break)
    process = subprocess.Popen(command, creationflags=creation_flags)
    try:
        return process.wait()
    except KeyboardInterrupt:
        _stop(process)
        return 0


def main(open_browser: bool = True, exit_after: float | None = None) -> int:
    _require_frontend_dependencies()
    _require_available_port(8000)
    _require_available_port(5173)
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(BACKEND_SOURCE), current_pythonpath]))

    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process_job = _create_process_job()
    backend = None
    frontend = None
    try:
        backend = _start_process(
            _backend_command(),
            cwd=ROOT,
            env=env,
            creation_flags=creation_flags,
            process_job=process_job,
        )
        frontend = _start_process(
            _frontend_command(),
            cwd=FRONTEND,
            env=env,
            creation_flags=creation_flags,
            process_job=process_job,
        )
    except Exception:
        _cleanup(frontend, backend, process_job)
        raise

    try:
        processes = [("backend", backend), ("frontend", frontend)]
        _wait_for(HEALTH_URL, processes)
        _wait_for(APP_URL, processes)
        print(f"\nApplication ready: {APP_URL}")
        print("API documentation: http://127.0.0.1:8000/docs")
        print("Stop this run configuration to stop both services.\n")
        if open_browser:
            webbrowser.open(APP_URL)

        ready_at = time.monotonic()
        source_signature = _source_signature()
        while backend.poll() is None and frontend.poll() is None:
            if exit_after is not None and time.monotonic() - ready_at >= exit_after:
                return 0
            updated_signature = _source_signature()
            if updated_signature != source_signature:
                time.sleep(0.2)
                updated_signature = _source_signature()
                print("Backend source changed; restarting API server...")
                _stop(backend)
                backend = _start_process(
                    _backend_command(),
                    cwd=ROOT,
                    env=env,
                    creation_flags=creation_flags,
                    process_job=process_job,
                )
                _wait_for(HEALTH_URL, [("backend", backend), ("frontend", frontend)])
                source_signature = updated_signature
                print("Backend reload complete.")
            time.sleep(0.5)

        if backend.poll() is not None:
            print(f"Backend stopped unexpectedly with exit code {backend.returncode}.", file=sys.stderr)
        if frontend.poll() is not None:
            print(f"Frontend stopped unexpectedly with exit code {frontend.returncode}.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        _cleanup(frontend, backend, process_job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the backend and frontend development servers.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the application in a browser.")
    parser.add_argument("--exit-after", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--child", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.child is not None:
            raise SystemExit(_run_child(args.child))
        raise SystemExit(main(open_browser=not args.no_browser, exit_after=args.exit_after))
    except RuntimeError as error:
        print(f"Cannot start application: {error}", file=sys.stderr)
        raise SystemExit(1) from error
