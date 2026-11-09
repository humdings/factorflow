# Contributing

Thanks for looking at factorflow. Bug reports, tests, and small fixes are welcome.
The library is Apache-2.0. By submitting a patch you agree that it is licensed
under the same terms (see `LICENSE` and `NOTICE`).

## Setup

Nix is the preferred environment when you have it:

```bash
nix develop
uv sync
```

`nix develop` puts `uv` and a Python on `PATH` and points `UV_PYTHON` at that interpreter.
`nix build` and `nix flake check` build the package and run the test suite, skipping tests
marked `slow`.

Without Nix, any Python 3.11+ works:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies are declared in `pyproject.toml` (`numpy`, `polars`, `pandas`,
`exchange-calendars`), so a plain `pip install .` from a checkout is enough to use the library.
The `[dev]` extra adds pytest, hypothesis, ruff, and pyright. The same packages are also listed
as a uv dependency group, which is what `uv sync` installs.

## Checks

```bash
pytest
ruff check .
ruff format --check .
pyright
python examples/quickstart.py
```

With uv: `uv run pytest`, and the same for ruff and pyright.
`pytest -m "not slow"` skips the longer simulation CLI test. `nix flake check` does that too.

Docstrings use [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).
`interrogate` requires a docstring on every public module, class, function, and method,
including `__init__` and other dunder methods. Single-underscore helpers are not part of
that check. `mkdocs build --strict` rebuilds the API reference from those docstrings
(`pip install -e ".[docs]"`).

Please add a test when you change behavior. The property that matters most is that a sealed
live row matches `BatchEngine` on the same inputs. Hypothesis tests in `tests/` are the pattern.

## Examples

`examples/quickstart.py` is synthetic and has to keep running. If you change the README
quickstart, change that file to match: the test suite checks that the README embeds it.

`examples/demo.ipynb` walks through `factorflow.sim` and draws plots. Re-execute it when you
change the simulator or the notebook:

```bash
pip install -e ".[notebook]"
python -m jupyter nbconvert --execute --to notebook --inplace examples/demo.ipynb
```

With uv that extra is the `notebook` dependency group: `uv run --group notebook python -m jupyter nbconvert --execute --to notebook --inplace examples/demo.ipynb`.

## Scope

factorflow builds bars and evaluates factor pipelines. It does not load market data, talk to
brokers, or store results. Those belong in the application that calls this library. A new
feature should work from a `Panel`, a trade stream, or another input the caller already holds.

Do not commit credentials, host names, or data taken from a private system.
