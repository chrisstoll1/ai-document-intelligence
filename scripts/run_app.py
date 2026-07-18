from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_SOURCE = ROOT / "backend" / "src"
FRONTEND = ROOT / "frontend"
APP_URL = "http://127.0.0.1:5173"
HEALTH_URL = "http://127.0.0.1:8000/api/health"


def _npm_command() -> str:
    executable = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if executable is None:
        raise RuntimeError("Node.js and npm are required. Install Node.js 20 or newer before running the app.")
    return executable


def _require_frontend_dependencies() -> None:
    if not (FRONTEND / "node_modules").is_dir():
        raise RuntimeError("Frontend dependencies are missing. Run `npm install` in the frontend directory once.")


def _require_available_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use. Stop the existing app run and try again.")


def _wait_for(url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
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


def main(open_browser: bool = True, exit_after: float | None = None) -> int:
    _require_frontend_dependencies()
    _require_available_port(8000)
    _require_available_port(5173)
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(BACKEND_SOURCE), current_pythonpath]))

    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    backend = subprocess.Popen(
        [
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
            "--reload",
        ],
        cwd=ROOT,
        env=env,
        creationflags=creation_flags,
    )
    frontend = subprocess.Popen(
        [_npm_command(), "run", "dev", "--", "--clearScreen", "false"],
        cwd=FRONTEND,
        env=env,
        creationflags=creation_flags,
    )

    try:
        _wait_for(HEALTH_URL)
        _wait_for(APP_URL)
        print(f"\nApplication ready: {APP_URL}")
        print("API documentation: http://127.0.0.1:8000/docs")
        print("Stop this run configuration to stop both services.\n")
        if open_browser:
            webbrowser.open(APP_URL)

        ready_at = time.monotonic()
        while backend.poll() is None and frontend.poll() is None:
            if exit_after is not None and time.monotonic() - ready_at >= exit_after:
                return 0
            time.sleep(0.5)

        if backend.poll() is not None:
            print(f"Backend stopped unexpectedly with exit code {backend.returncode}.", file=sys.stderr)
        if frontend.poll() is not None:
            print(f"Frontend stopped unexpectedly with exit code {frontend.returncode}.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        _stop(frontend)
        _stop(backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the backend and frontend development servers.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the application in a browser.")
    parser.add_argument("--exit-after", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        raise SystemExit(main(open_browser=not args.no_browser, exit_after=args.exit_after))
    except RuntimeError as error:
        print(f"Cannot start application: {error}", file=sys.stderr)
        raise SystemExit(1) from error
