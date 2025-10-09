# MSDS601 Final Project — Causal Inference Interactive Blog (Dash)

This repository contains a Plotly Dash application that presents a blog-style writeup on causal inference alongside an interactive section. It was cloned from `ahklaus/msds601_final_project` and configured to run locally on Windows.

## Quickstart (Windows / PowerShell)

```bash
# 1) Create virtual environment
python -m venv .venv

# 2) Activate it
source .venv/bin/activate 
OR
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4) Run the app
python app.py
```

After starting, the Dash development server prints a local URL (typically `http://127.0.0.1:8050`). Open it in your browser.

## Deploying on Render.com

This app is ready for Render. Key points:

- The Dash app now exposes `server` in `app.py` and reads the `PORT` env var.
- `gunicorn` is included in `requirements.txt` and used to serve the app in production.
- Create a new Web Service on Render pointing to this repo with:
  - Runtime: Python
  - Build command: `python -m pip install -r requirements.txt`
  - Start command: `python -m gunicorn app:server --workers 2 --threads 4 --timeout 60`
  - Environment: `PYTHON_VERSION` (e.g., 3.11), optional `DASH_DEBUG=False`

Alternatively, add a `render.yaml` at the repo root:

```yaml
services:
  - type: web
    name: msds601-final-project
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:server --workers 2 --threads 4 --timeout 60
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
```

## Project Structure

- `app.py`: Dash application with blog content and an interactive example using Plotly Express.
- `assets/`: Static assets used by the app (e.g., images). Dash automatically serves files in this folder.
- `requirements.txt`: Python dependencies pinned for reproducibility.
- `README.md`: Setup and usage instructions.

## Notes

- The app uses Bootstrap theme `LUX` via `dash-bootstrap-components`.
- Replace the example interactive section with your own simulation, callbacks, and figures as needed.

## Troubleshooting

- If PowerShell blocks script execution when activating the venv, you may need to allow local scripts:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then re-run activation: `.\.venv\Scripts\Activate.ps1`.

- If ports are busy, set a custom port:
```powershell
$env:PORT=8060
python app.py
```
