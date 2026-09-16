# Contributing to CiteCraft RAG

Thanks for helping improve CiteCraft. Focused issues and pull requests are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
docker compose up -d
```

Set `OPENAI_API_KEY` in `.env` only when exercising ingestion or answer generation. The unit tests
do not make live model calls.

## Before opening a pull request

```bash
pytest
ruff check .
```

Keep changes small, add tests for behavioral changes, and update the README when configuration or
user-facing behavior changes. Never commit API keys, uploaded documents, database dumps, or local
`.env` files.

## Pull requests

Describe the problem, the approach, and how you verified the change. Screenshots are helpful for
Streamlit UI changes. By contributing, you agree that your work will be licensed under the MIT
License.
