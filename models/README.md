# Model Directory

The local reproduction folder contains symlinks to model snapshots on this machine.

Expected paths:

- `Qwen2.5-VL-7B-Instruct`
- `Qwen3-VL-8B-Instruct`
- `Qwen3-VL-32B-Instruct`
- `Qwen3-VL-Embedding-8B`

For a clean environment, download the models into these paths:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir models/Qwen3-VL-32B-Instruct
huggingface-cli download Qwen/Qwen3-VL-Embedding-8B --local-dir models/Qwen3-VL-Embedding-8B
```
