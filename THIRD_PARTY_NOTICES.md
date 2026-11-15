# Third-party notices

This repository contains no third-party source code, model weights, or
adapter weights. It depends on, or contains records produced with, the
following third-party materials. Each is governed by its own terms. The
project license (`LICENSE`, Apache-2.0) and the data license (`data/LICENSE`,
CC BY 4.0) cover only the authors' own contributions.

## Apple ToolSandbox

* Upstream: https://github.com/apple/ToolSandbox, commit
  `165848b9a78cead7ca7fe7c89c688b58e6501219` (package `tool_sandbox` 0.0.1).
* Copyright (C) 2024 Apple Inc. Distributed under Apple's license. A verbatim copy of the upstream
  `LICENSE` at that commit is in `third_party_licenses/ToolSandbox-LICENSE.txt`
  (SHA-256 `2283d5566b38210fbfb9fcb06619f1d0a8101bff9df974ea8afc3393508e9015`). ToolSandbox's own bundled
  subcomponents are listed in its upstream `ACKNOWLEDGEMENTS` file, which is not reproduced here because no ToolSandbox
  code is redistributed.
* Use in this project: ToolSandbox is **not** included. It is an optional runtime dependency that users install
  themselves (README, "Evaluate another model"). `src/actionrankwm/envs/toolsandbox.py` imports the
  unmodified upstream tool functions at run time. The authored tasks call ToolSandbox tools by their public
  names and signatures. The tool descriptions shown to world models (`src/actionrankwm/toolsandbox_tasks.py`)
  are our own paraphrases, not copied upstream docstrings.
* The records in `data/` (initial states, candidate actions, real branch outcomes) were produced by executing
  these authored tasks with ToolSandbox. They contain no ToolSandbox source code or upstream scenario text.
  `ToolSandbox` and `Apple` are trademarks of Apple Inc. Apple does not endorse this project.

## Models whose outputs or scores appear in `data/`

No weights are distributed. Raw model outputs (`predictions.jsonl.gz`) are included only for the two Qwen
checkpoints. For the other models only derived scores are included.

| Model | Revision | License of the model |
|---|---|---|
| Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Apache-2.0 |
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Apache-2.0 |
| meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | Llama 3.1 Community License and Acceptable Use Policy |
| google/gemma-4-12B-it | `0e2b1058541244490925fbacf8972041435691ac` | Apache-2.0 (per the model card at that revision) |
| gpt-5.6-sol (OpenAI API, E4a development panel only) | n/a | OpenAI terms of use |

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.
The Llama-derived records in `data/` are evaluation scores; this repository distributes no Llama weights,
adapters, or Llama model outputs. If you train or release a Llama adapter with `scripts/train_mr_sft_lora.py`,
the Llama 3.1 license (including its attribution and naming requirements) applies to that adapter.

## Python dependencies

The core package uses only the Python standard library. Optional extras (`pytest`, `openai`, `torch`,
`transformers`, `safetensors`) are installed by the user from their upstream distributions under
their own licenses.
