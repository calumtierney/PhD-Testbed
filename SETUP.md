# SETUP.md — running and viewing the testbed

Written for VS Code on Windows. macOS and Linux differences are noted where they matter.

---

## Part 1 — one-time setup

You do this once. After that, Part 2 is all you need.

### 1. Open the project

VS Code → **File → Open Folder** → select the repository root (the folder containing `src/`).

Always open the *folder*, not individual files. VS Code needs to know where the project root is.

### 2. Install the Python extension

Extensions panel (`Ctrl+Shift+X`) → search "Python" → install the one published by Microsoft. It brings the debugger and test integration with it.

### 3. Create a virtual environment

A virtual environment is a private copy of Python for this project. Without one, packages installed for this project can break other projects on your machine, and vice versa. It is a five-second step that prevents a category of problem that is miserable to debug.

Open the terminal in VS Code with `` Ctrl+` `` (backtick, top-left of the keyboard), then:

```bash
python -m venv .venv
```

A `.venv` folder appears. It is disposable — if it ever misbehaves, delete it and repeat.

### 4. Activate it

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

macOS / Linux:
```bash
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. **If it doesn't, nothing below will work properly.**

*If PowerShell refuses with an execution policy error*, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 5. Point VS Code at it

`Ctrl+Shift+P` → type "Python: Select Interpreter" → choose the one showing `.venv`.

This is what makes VS Code's Run button, test runner and import hints use the right Python. Skipping it causes the confusing situation where the terminal works but the editor shows import errors.

### 6. Install dependencies

```bash
pip install -r requirements.txt
```

### 7. Install the project itself

```bash
pip install -e .
```

This one matters and is worth understanding. It tells Python "this folder is a package — when something asks for `src.instruments`, look here." Without it you get `ModuleNotFoundError: No module named 'src'` the moment you run a script from anywhere other than the exact root folder. The `-e` means editable: your code changes take effect immediately, with no reinstalling.

If there is no `pyproject.toml` yet, see the prompt at the end of this file.

---

## Part 2 — daily use

### Generate the figures

```bash
python -m viz.make_figures
```

PNGs appear in `results/`. Click one in VS Code's file explorer and it opens in a tab. Re-run the command and the tab refreshes.

### Run the interactive explorer

```bash
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Move a slider, the figure updates.

Stop it with `Ctrl+C` in the terminal. **Do this before closing VS Code** — otherwise the port stays occupied and next time you get "Port 8501 is already in use." If that happens:

```bash
streamlit run app.py --server.port 8502
```

### Run the tests

Either:
```bash
pytest
```

or use the flask-shaped Testing icon in VS Code's left sidebar, which gives you a tree of tests with green ticks and lets you run one at a time.

---

## Part 3 — when something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Project not installed as a package | `pip install -e .` |
| `ModuleNotFoundError: No module named 'numpy'` | Virtual environment not active | Re-run the activate command from step 4 |
| Imports underlined red in the editor but code runs fine | VS Code using the wrong interpreter | Redo step 5 |
| `Port 8501 is already in use` | A Streamlit session is still running | `Ctrl+C` in its terminal, or use `--server.port 8502` |
| Figure window opens then closes instantly | Script ends before display | Save with `savefig` instead of `show`, and open the PNG |
| Blank or empty 3D figure | Usually a real bug, not a plotting one | Check the covariance values before touching the renderer |
| Changes to code have no effect | Streamlit sometimes caches | Press `R` in the browser, or restart it |

**The general rule:** if a *figure* looks wrong, suspect the physics before the plotting. A rotated ellipsoid means a rotated covariance.

---

## Part 4 — a comfortable workflow

Once running, the loop that works well:

1. Split the VS Code window — code on the left, PNG preview on the right (`Ctrl+\` splits the editor).
2. Keep one terminal for `pytest`, a second for figures or Streamlit. The `+` icon in the terminal panel adds one.
3. Edit → save → re-run → look. Seconds per cycle.

For the Streamlit app, put it on a second monitor if you have one and leave it running. It reloads when you save a file.

---

## Prompt: hand the setup to Claude Code

If any of the above does not exist yet, paste this:

```
Read CLAUDE.md and SETUP.md.

Make this project trivial to run from a fresh clone. Specifically:

1. A pyproject.toml so that `pip install -e .` works and `from src...`
   imports resolve from anywhere in the project.
2. A requirements.txt pinned to versions that work together.
3. A viz/make_figures.py entry point that regenerates every figure in
   results/ in one command, printing what it wrote.
4. A .vscode/settings.json selecting the .venv interpreter and enabling
   pytest discovery.
5. A .gitignore covering .venv, __pycache__, and results/*.png — figures are
   generated artefacts, not source.
6. A short "Running this" section at the top of README.md: three commands,
   no explanation, for when I have forgotten.

Then verify it yourself: describe exactly what a person would type from a
fresh clone to see a figure, and confirm each command works.

If any dependency needs a system library that pip cannot install on Windows,
say so now rather than letting me discover it later.
```
