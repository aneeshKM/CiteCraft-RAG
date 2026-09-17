# CiteCraft benchmark

This benchmark compares CiteCraft's parent/summary retrieval with a raw-element embedding
baseline on the human-annotated FinanceBench open-source sample.

## Data source

The manifest pins the official FinanceBench revision and identifies 32 questions from 10 public
financial reports. FinanceBench supplies the source PDFs, human-written answers, and annotated
evidence pages. Source PDFs and copied benchmark data remain outside version control.

```bash
git clone https://github.com/patronus-ai/financebench.git /tmp/citecraft-financebench
git -C /tmp/citecraft-financebench checkout cc39aeb4afdf33909ee1412188bf89035950c2eb
```

The pinned revision is recorded in `financebench_manifest.json`. Check out that revision before a
reproducible run.

## Run

Start the local services, activate the project environment, and run:

```bash
docker compose up -d

python benchmarks/run_financebench.py \
  --financebench-dir /tmp/citecraft-financebench \
  --reset-index
```

Use `--limit 10` for a pilot. The benchmark uses Redis databases 1 and 2 and the PostgreSQL tables
`citecraft_benchmark_summary` and `citecraft_benchmark_raw`; application data in Redis database 0
and `citecraft_documents` is not modified.

## Measurements

- Evidence-page Hit@5, Recall@5, MRR, and nDCG@5
- Gold-answer correctness
- Groundedness, citation correctness, and citation completeness
- Source-withheld abstention accuracy
- Ingestion, retrieval, generation, and end-to-end latency
- Model token usage when returned by the API

The result JSON contains per-question measurements, aggregate values, configuration, errors, and
bootstrap confidence intervals. Reports, raw results, and improvement notes are written to the
gitignored `tmp/financebench/` workspace by default so benchmark iterations remain local until
they are ready to publish.
