# ActionRankWM

Evaluate whether a language world model ranks candidate actions in the same order as real execution. The real and predicted branches use the same starting state, continuation, and utility.

## Recompute results

From the repository root, run the following with Python 3.10 or newer. No installation, GPU, API key, or network access is needed.

```sh
python scripts/reproduce_paper_tables.py
```

This prints decision metrics from the bundled [evaluation records](data/README.md). For example, on the matched panel, token F1 increases from 0.671 (4B) to 0.763 (9B), while pairwise order preservation drops from 0.836 to 0.753. Use `--json results.json` for all computed values.

## Evaluate another model

Install the package (`pip install -e .`) and [Apple ToolSandbox](https://github.com/apple/ToolSandbox) at commit `165848b9a78cead7ca7fe7c89c688b58e6501219` under `third_party/ToolSandbox`. With an OpenAI-compatible endpoint already running:

```sh
WORLD_MODEL_BASE_URL=http://localhost:8000/v1 SERVED_NAME=my-model bash examples/run_e4b_pipeline.sh
```

The example builds tasks, predicts each branch, and scores the resulting choices. For development, install `pip install -e '.[dev]'` and run `pytest -q`.

Code: [Apache-2.0](LICENSE). Bundled records: [CC BY 4.0](data/LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md).
