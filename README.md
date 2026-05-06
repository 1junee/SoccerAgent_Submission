# SoccerAgent Submission

This is the cleaned SoccerAgent inference package.

Use `README_REPRO.md` for setup and reproduction instructions.

Main command:

```bash
cp .env.example .env
bash run_socceragent.sh challenge/test/test.json outputs/test/result.json
```

This package intentionally excludes experiment folders, train/valid splits, old `*_original.py` entrypoints, `legacy_tool/`, logs, and previous outputs.
