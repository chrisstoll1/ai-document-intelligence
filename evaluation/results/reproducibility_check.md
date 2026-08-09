# Reproducibility Check

Date: 9 August 2026

The documented Windows setup was executed in a new Python 3.11 virtual environment under the approved temporary directory:

```powershell
py -3.11 -m venv <temporary-venv>
<temporary-venv>\Scripts\python.exe -m pip install --upgrade pip
<temporary-venv>\Scripts\python.exe -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
<temporary-venv>\Scripts\python.exe -m pip install -e ".[dev]"
<temporary-venv>\Scripts\python.exe -m pip check
<temporary-venv>\Scripts\python.exe -m pytest
```

Results:

- Python 3.11.2.
- PyTorch 2.12.0+cu130.
- CUDA available on NVIDIA GeForce RTX 4090.
- `pip check`: no broken requirements.
- Backend: 101 tests passed from fresh dependency resolution.
- One upstream Starlette warning reported that `httpx` support in `TestClient` is deprecated in favor of `httpx2`; no test failed.

The frontend was then reinstalled from `package-lock.json` using `npm ci`. The initial audit reported four transitive toolchain vulnerabilities. `npm audit fix` updated five locked packages without requiring `--force`; the final audit reported zero vulnerabilities. ESLint and the Vite production build passed after the update.

The unqualified `python` command resolved to obsolete Python 3.7 on the evaluation machine. Setup documentation now uses `py -3.11` explicitly and upgrades `pip` before installing the CUDA wheel.
