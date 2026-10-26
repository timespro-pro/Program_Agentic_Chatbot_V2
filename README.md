# Program Performance Analysis Agent — Version 2

This package is based on the latest Version 2 files supplied in the conversation. Version 1 is not included or modified. Existing UI styling is retained; this update adds a shared GitHub-backed cache for **Overall program analysis**.

## What the shared cache does

- First request for a program calculates evidence and generates the report, then attempts to store the report as JSON in the configured GitHub repository under `analysis_cache/overall_program_analysis/`.
- Later requests from other users or after a Streamlit Community Cloud restart can retrieve the report from GitHub and skip the OpenAI call.
- Cache keys include analysis/prompt schema version, selected program code, and SHA-256 fingerprints of both `data/Program Batch Data.csv` and `data/Advisor Mix.csv`. Changing either dataset invalidates the old key automatically.
- GitHub cache failures are non-fatal: the app still generates the report and tells the user if the report could not be shared.
- A per-process lock reduces duplicate same-key work in one running worker. It is **not a distributed lock** across separate app workers, so two simultaneous first-time requests on separate workers can still race. The GitHub API write/read-back handling is best-effort.

## Streamlit Community Cloud setup

In your deployed app, open **App settings → Secrets** and add the following TOML. Do not commit secrets to Git or put a token in Python code.

```toml
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_MODEL = "gpt-5.6"

[github_cache]
token = "github_pat_your_fine_grained_token"
repository = "YOUR_ORG_OR_USERNAME/YOUR_REPOSITORY"
branch = "main"
```

Create a fine-grained GitHub personal access token restricted to the repository containing this app, with **Contents: Read and write** permission. The repository must permit the token to create/update files on the configured branch. Keep the token private and rotate it if it is exposed.

If the `[github_cache]` block is omitted or the token/repository is invalid, the app continues to work without shared caching; reports will not persist for other users or restarts.

## Deployment contents

- `app.py` and the analysis modules are at the repository root.
- CSV data files are under `data/`.
- Logo and background assets are included at root and in `assets/` where the app expects them.
- `analysis_cache/` is the shared report-storage folder. It may start empty; the app creates report files through GitHub's Contents API.

## Validation performed

The Python files are syntax-compiled and the cache key/JSON serialization logic is smoke-tested. This environment does not have your GitHub token, OpenAI key, or deployed Streamlit Cloud runtime, so it cannot verify a live GitHub write, OpenAI generation, or browser/deployment behavior. After deployment, generate one report, confirm a JSON file appears under `analysis_cache/overall_program_analysis/`, then run the same analysis again and confirm it reports a cache hit.
