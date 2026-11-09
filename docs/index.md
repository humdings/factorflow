# factorflow

factorflow evaluates factor, filter, and classifier expressions on bars. The same
pipeline runs as a batch over a history and incrementally on the still-open bar,
and the sealed rows agree.

Install and a worked example are in the repository [README](https://github.com/humdings/factorflow/blob/main/README.md).
This site is the API reference generated from the NumPy-style docstrings.

- [Bars and time](BARS.md) covers clock bars, event bars, and as-of sampling.
- [API reference](api.md) is generated from the source.

Build it locally with the `docs` extra:

```bash
pip install -e ".[docs]"
mkdocs build --strict
```
