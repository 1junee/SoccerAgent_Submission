# SoccerAgent Reproduction Package

This folder contains the SoccerAgent main inference flow only. Experiment notebooks, experiment launchers, logs, old `*_original.py` entrypoints, and `legacy_tool/` code were intentionally removed.

## Included

- Main entrypoint: `platform_full_version.py`
- Agent/tool orchestration: `multiagent_platform.py`
- Tool implementations: `pipeline/`
- Challenge/test JSON manifests: `challenge/test`, `challenge/challenge`
- Improved replay grounding: `pipeline/toolbox/replay_grounding.py`
- Qwen3-VL-Embedding source: `third_party/Qwen3-VL-Embedding`
- Model path documentation under `models/`

Large challenge materials, database media/images, and model checkpoints are not
intended to be committed to git. Place them at the expected paths before running.

## Environment

Recommended:

- Ubuntu/Linux
- Python 3.10
- CUDA 12.x
- NVIDIA GPUs with enough memory for the selected local VLM
- `ffmpeg` available on `PATH`

Create the environment:

```bash
cd /data1/heodnjswns/SoccerAgent_Submission
conda create -n socceragent-repro python=3.10 -y
conda activate socceragent-repro
pip install -r requirements.txt
pip install -e pipeline/toolbox/utils/GroundingDINO
```

If `flash_attn` fails to build, install a wheel matching your CUDA/PyTorch stack, then rerun the remaining requirements.

## Models

Place or download the required models under `models/`:

- `models/Qwen2.5-VL-7B-Instruct`
- `models/Qwen3-VL-8B-Instruct`
- `models/Qwen3-VL-32B-Instruct`
- `models/Qwen3-VL-Embedding-8B`

For example:

```bash
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir models/Qwen3-VL-32B-Instruct
huggingface-cli download Qwen/Qwen3-VL-Embedding-8B --local-dir models/Qwen3-VL-Embedding-8B
```

Use 7B or 8B VLM instead by setting `VLM_MODEL_NAME` in `.env`.

## Data and Materials

The JSON manifests are tracked in git, but large media/database folders should be
provided separately:

- `challenge/test/materials/`
- `challenge/challenge/materials/`
- `database/Game_dataset/`
- `database/SoccerBench/materials/`
- `database/SoccerWiki/pic/`

Keep `database/Game_dataset_csv/game_database.csv`,
`database/SoccerBench/qa`, `database/SoccerBench/subqa`, and
`database/SoccerWiki/data` with the code unless you choose to distribute all
database assets separately.

## Configuration

```bash
cp .env.example .env
```

Fill `OPENAI_API_KEY` or configure another OpenAI-compatible agent provider in `.env`.

Important defaults:

- `VISION_BACKEND=qwen`
- `VLM_MODEL_NAME=./models/Qwen3-VL-32B-Instruct`
- `REPLAY_GROUNDING_EMBED_BACKEND=qwen`
- `QWEN3_VL_EMBED_MODEL=./models/Qwen3-VL-Embedding-8B`
- `VLM_MAX_NEW_TOKENS=8000`

## Run

Run the full SoccerAgent flow on the default test set:

```bash
cd /data1/heodnjswns/SoccerAgent_Submission
conda activate socceragent-repro
GPUS=0,1,2,3 bash run_socceragent.sh challenge/test/test.json outputs/test/result.json
```

Run a grouped subset:

```bash
GPUS=0,1,2,3 bash run_socceragent.sh challenge/challenge/grouped/q9q10q11.json outputs/q9q10q11/result.json
```

The output JSON contains each item with `answer`, `planner_output`, `openA_process`, and related diagnostics.

## Notes

- This package does not include experiment folders.
- Training/validation splits are not included; this package is for inference reproduction.
- The replay-grounding API backend is disabled in the submitted code path; local Qwen3-VL-Embedding is used.
- Qwen3-VL-Embedding is imported from `third_party/Qwen3-VL-Embedding` through `QWEN3_VL_EMBED_ROOT`; it does not need a separate editable install.
- `.env` with real keys is not included.
