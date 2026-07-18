import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import scripts.run_app as run_app


def test_backend_runs_without_uvicorn_reload_supervisor() -> None:
    command = run_app._backend_command()

    assert command[:4] == [sys.executable, "-m", "uvicorn", "docintel.api:app"]
    assert "--reload" not in command
    assert command[command.index("--timeout-graceful-shutdown") + 1] == "3"


def test_frontend_runs_vite_directly(monkeypatch) -> None:
    monkeypatch.setattr(run_app, "_node_command", lambda: "node-test")

    command = run_app._frontend_command()

    assert command == ["node-test", str(run_app.VITE_SCRIPT), "--clearScreen", "false"]


def test_source_signature_tracks_only_python_files(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    python_file = package / "module.py"
    python_file.write_text("VALUE = 1\n", encoding="utf-8")
    (package / "ignored.txt").write_text("ignored", encoding="utf-8")

    first = run_app._source_signature(tmp_path)
    python_file.write_text("VALUE = 200\n", encoding="utf-8")
    second = run_app._source_signature(tmp_path)

    assert len(first) == len(second) == 1
    assert first != second


def test_windows_stop_sends_break_before_force_kill(monkeypatch) -> None:
    process = Mock(spec=subprocess.Popen)
    process.pid = 1234
    process.poll.return_value = None
    taskkill = Mock()
    monkeypatch.setattr(run_app.os, "name", "nt")
    monkeypatch.setattr(run_app.subprocess, "run", taskkill)

    run_app._stop(process)

    process.send_signal.assert_called_once_with(run_app.signal.CTRL_BREAK_EVENT)
    process.wait.assert_called_once_with(timeout=5)
    taskkill.assert_not_called()


def test_windows_stop_force_kills_unresponsive_tree(monkeypatch) -> None:
    process = Mock(spec=subprocess.Popen)
    process.pid = 1234
    process.poll.return_value = None
    process.wait.side_effect = subprocess.TimeoutExpired("test", 5)
    taskkill = Mock()
    monkeypatch.setattr(run_app.os, "name", "nt")
    monkeypatch.setattr(run_app.subprocess, "run", taskkill)

    run_app._stop(process)

    assert taskkill.call_args.args[0] == ["taskkill", "/PID", "1234", "/T", "/F"]


def test_windows_child_forwards_break_to_process(monkeypatch) -> None:
    process = Mock(spec=subprocess.Popen)
    process.wait.side_effect = KeyboardInterrupt
    stop = Mock()
    register_signal = Mock()
    monkeypatch.setattr(run_app.os, "name", "nt")
    monkeypatch.setattr(run_app.sys, "stdin", SimpleNamespace(buffer=SimpleNamespace(read=lambda _: b"1")))
    monkeypatch.setattr(run_app.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(run_app.signal, "signal", register_signal)
    monkeypatch.setattr(run_app, "_stop", stop)

    assert run_app._run_child(["test-command"]) == 0

    register_signal.assert_called_once_with(run_app.signal.SIGBREAK, run_app._interrupt_on_break)
    stop.assert_called_once_with(process)


def test_wait_for_fails_immediately_when_child_exits() -> None:
    process = Mock(spec=subprocess.Popen)
    process.poll.return_value = 1
    process.returncode = 1

    with pytest.raises(RuntimeError, match="Backend stopped during startup with exit code 1"):
        run_app._wait_for("http://127.0.0.1:1", [("backend", process)])


def test_cleanup_closes_job_even_when_frontend_stop_fails(monkeypatch) -> None:
    frontend = Mock(spec=subprocess.Popen)
    backend = Mock(spec=subprocess.Popen)
    process_job = Mock()
    stopped = []

    def stop(process) -> None:
        stopped.append(process)
        if process is frontend:
            raise OSError("stop failed")

    monkeypatch.setattr(run_app, "_stop", stop)

    with pytest.raises(OSError, match="stop failed"):
        run_app._cleanup(frontend, backend, process_job)

    assert stopped == [frontend, backend]
    process_job.close.assert_called_once()


def test_null_job_accepts_process() -> None:
    job = run_app._NullProcessJob()

    job.assign(Mock(spec=subprocess.Popen))
    job.close()
