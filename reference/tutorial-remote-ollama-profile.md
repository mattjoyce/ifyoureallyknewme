# Running `knowme profile` against a remote Ollama host

**Audience:** future us, who has forgotten.
**Goal:** generate a profile using a model hosted on a LAN ollama box instead of OpenAI.
**Written:** 2026-04-13, first known-good run on ollama at `192.168.20.8:11434` with `gemma4:latest`.

---

## Why this exists

`knowme` uses Simon Willison's [`llm`](https://llm.datasette.io/) library for all generative calls
(`call_llm()` in `knowme.py`). `llm` supports ollama via the `llm-ollama` plugin, and the plugin
honours the `OLLAMA_HOST` env var for pointing at a non-local host. So we can drive a remote box
with no code changes — only a swap of the `llm.generative_model` value in config and an env var.

Embeddings are **not** routed through `llm`. `get_embedding()` talks to the OpenAI client directly
(`knowme.py:347-358`). So any command that computes embeddings (`analyze`, `merge`, `interviewer`,
`regen-embeddings`) still needs `OPENAI_API_KEY` and still hits OpenAI even when generative calls
go to ollama. The `profile` command does **not** compute embeddings, so it can run purely against
ollama — that's what this tutorial covers.

## Files involved

- `config_ollama.yaml` — sibling of `config.yaml`, differs only in `llm.generative_model`.
- `~/Environments/ifyoureallyknewme/` — the project venv (matt keeps venvs outside the repo; never
  create `.venv/` inside the project dir).

## One-time setup

```bash
# 1. Create the venv outside the project and sync deps
export UV_PROJECT_ENVIRONMENT=~/Environments/ifyoureallyknewme
uv sync

# 2. Install the ollama plugin into the same env
uv pip install llm-ollama

# 3. Sanity-check the plugin is registered
uv run llm plugins | grep -i ollama
```

## Verify the remote host before you run

```bash
curl -s http://192.168.20.8:11434/api/tags | python -m json.tool | grep '"name"'
```

You should see `gemma4:latest` (or whichever model you plan to use) in the list. If the host is
unreachable, everything below will fail with a cryptic `llm` error — check the host first.

## Running profile generation

```bash
export UV_PROJECT_ENVIRONMENT=~/Environments/ifyoureallyknewme
export OLLAMA_HOST=http://192.168.20.8:11434

uv run knowme --config config_ollama.yaml profile \
  --mode short \
  --format md \
  --output results/profile-gemma4-$(date +%Y%m%d).md
```

Flags recap (from `knowme.py:1867`):
- `--mode short|long` — length of the generated profile
- `--format md|json|raw`
- `--output <path>` — write to file instead of printing to terminal
- `--dump` — dump raw observations instead of generating; useful for inspecting the input

## Switching models

Edit `llm.generative_model` in `config_ollama.yaml`. Any tag listed by `/api/tags` on the remote
host works — e.g. `qwen3.5:27b`, `gpt-oss:latest`. Just make sure the model is actually pulled on
the remote box; `llm-ollama` will not pull for you.

## Why a second config file instead of a `--model` flag

The `analyze` and `merge` commands have `--model` flags; `profile` does not. We chose not to add
one because:

1. Profile runs are usually paired with a specific config (model + output format + mode).
2. `--config config_ollama.yaml` captures the full intent in one place, so future-us can re-run
   verbatim without remembering a set of flags.
3. The env var `OLLAMA_HOST` still has to be set regardless, so a flag wouldn't remove all the
   out-of-band setup anyway.

If we ever want ad-hoc overrides, add `--model` to `profile` mirroring the pattern at
`knowme.py:1606-1620` (parse flag → write into `cfg["llm"]["generative_model"]`).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Unknown model: gemma4:latest` | `llm-ollama` not installed, or installed into the wrong venv |
| Connection timeout / refused | `OLLAMA_HOST` not exported, or wrong host, or the box is off |
| `Unknown model` but plugin is installed | `OLLAMA_HOST` points at a box that doesn't have that tag pulled |
| Works for `profile` but `analyze` still hits OpenAI | Expected — embeddings are hardcoded to OpenAI |
| `ModuleNotFoundError` after `uv sync` | `UV_PROJECT_ENVIRONMENT` not exported in this shell |
