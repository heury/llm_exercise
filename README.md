# llm-exercise

## Environment setup

This project uses `uv` to manage Python dependencies from `pyproject.toml` and
`uv.lock`.

### Install uv

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, open a new terminal and check that `uv` is available:

```powershell
uv --version
```

### Sync the project environment

Run the following command from the project root:

```powershell
uv sync
```

`uv sync` creates or updates the local `.venv` environment so that it matches
the dependencies declared in `pyproject.toml` and locked in `uv.lock`.

If the existing `.venv` points to a missing location or is otherwise broken,
remove it first and then run `uv sync` again:

```powershell
Remove-Item .venv -Recurse -Force
uv sync
```

### Download the dataset

Clone the dataset from Hugging Face into `minimind_dataset`:

```powershell
git clone https://huggingface.co/datasets/caspar/llm_exercise_dataset minimind_dataset
```

### Use the environment

To run commands inside the synced environment:

```powershell
uv run python main.py
```

Or activate the virtual environment directly:

```powershell
.\.venv\Scripts\Activate.ps1
```
