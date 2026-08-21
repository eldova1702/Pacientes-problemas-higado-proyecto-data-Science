# Repository Instructions

## Current project state

- This is an ILPD binary-classification proof of concept. `src/` and most of `tests/` are still scaffolding; the notebooks are the current executable workflow.
- Notebook order is `notebooks/1-data` (raw ingestion), `notebooks/2-exploration` (cleaning and typing), then `notebooks/3-analysis` (current EDA/analysis).
- `notebooks/3-analysis/03.Analisis_Datos-DCY-2026-08-20.ipynb` is currently the work-in-progress notebook. The preceding notebook writes `data/02_intermediate/pacientes_higado_exploracion.parquet`.
- Existing notebook paths use `Path("../..")`; run a notebook with its stage directory as the working directory, or adjust paths deliberately.

## Data conventions

- The raw source is `data/01_raw/Pacientes_porblemas_higado_india.csv`. Do not modify it.
- The raw CSV contains many trailing empty columns. Load only the 11 valid columns with `usecols`, not the full malformed-looking schema.
- The raw target is `Dataset`: `1` means liver disease and `2` means no liver disease. The exploration notebook renames it to `Diagnosis`.
- The exploration stage converts `Gender` and `Diagnosis` to `category`, keeps nullable integer `Age` as pandas `Int64`, and writes intermediate data as Parquet with `pyarrow`.
- Preserve clinical outliers during EDA; handle class imbalance during model training, not during initial exploration.
- New files under `data/` are ignored by `.gitignore`; generated Parquet files are local unless intentionally forced into version control. The existing raw CSV is already tracked.

## Environment and checks

- Use Python 3.12, managed with `uv`. Set up with `uv sync --all-groups`; add dependencies with `uv add <package>`.
- `make install_data_libs` adds the data-science stack, including NumPy, SciPy, scikit-learn, Jupyter, Matplotlib, and Seaborn.
- Run focused tests with `uv run pytest tests/test_mock.py -q`, the full suite with `uv run pytest --cov`, and all repository checks with `uv run pre-commit run --all-files`.
- Ruff is configured in `.code_quality/ruff.toml`, uses a 100-character line length, and includes `*.ipynb`; its pre-commit hook runs with auto-fix. Inspect the diff after running it.
- Mypy uses `.code_quality/mypy.ini`; pytest discovers tests from `tests/` with `src` on `PYTHONPATH`.

## Git workflow

- Follow Gitflow: create feature branches from `main`, using names such as `feature/descriptive-name`.
- Use Conventional Commit prefixes such as `feat:`, `fix:`, `style:`, and `refactor:`. The pre-commit configuration also checks commit messages and branch names.
