"""Run CiteCraft's reproducible FinanceBench evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
for import_root in (ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from benchmarks.metrics import (  # noqa: E402
    bootstrap_mean_interval,
    mean,
    percentile,
    ranking_metrics,
)
from citecraft.config import Settings  # noqa: E402
from citecraft.hashing import sha256_file  # noqa: E402
from citecraft.models import RetrievedElement  # noqa: E402
from citecraft.pdf_processor import extract_pdf_elements  # noqa: E402
from citecraft.prompts import ANSWER_SYSTEM_PROMPT, build_context  # noqa: E402
from citecraft.storage import DocumentRepository  # noqa: E402
from citecraft.summarizer import ElementSummarizer  # noqa: E402

LOGGER = logging.getLogger("citecraft.benchmark")
DEFAULT_MANIFEST = ROOT / "benchmarks" / "financebench_manifest.json"
DEFAULT_OUTPUT = ROOT / "tmp" / "financebench" / "financebench_results.json"
DEFAULT_REPORT = ROOT / "tmp" / "financebench" / "BENCHMARK.md"


class JudgeGrade(BaseModel):
    """Structured semantic grading result on a three-point rubric."""

    correctness: int = Field(ge=0, le=2)
    groundedness: int = Field(ge=0, le=2)
    citation_correctness: int = Field(ge=0, le=2)
    citation_completeness: int = Field(ge=0, le=2)
    abstained: bool
    explanation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--financebench-dir",
        type=Path,
        required=True,
        help="Path to a checkout of the official FinanceBench repository.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N manifest questions.")
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Clear the isolated benchmark Redis databases and PostgreSQL tables first.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Measure ingestion and retrieval only.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Generate answers without semantic judge scores.",
    )
    parser.add_argument(
        "--abstention-limit",
        type=int,
        default=10,
        help="Maximum source-withheld cases, one per selected document.",
    )
    parser.add_argument(
        "--allow-revision-mismatch",
        action="store_true",
        help="Run even when the FinanceBench checkout differs from the pinned revision.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            questions[str(item["financebench_id"])] = item
    return questions


def checkout_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def redis_url_with_database(url: str, database: int) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database}", parsed.query, parsed.fragment))


def find_pdf(pdf_directory: Path, doc_name: str) -> Path:
    matches = {
        path.stem.casefold(): path for path in pdf_directory.glob("*.pdf") if path.is_file()
    }
    try:
        return matches[doc_name.casefold()]
    except KeyError as exc:
        raise FileNotFoundError(f"FinanceBench PDF not found for {doc_name!r}") from exc


def reset_repository(repository: DocumentRepository, settings: Settings) -> None:
    import psycopg
    import redis
    from psycopg import sql

    redis.Redis.from_url(settings.redis_url).flushdb()
    with psycopg.connect(
        settings.postgres_url.replace("postgresql+psycopg://", "postgresql://", 1),
        connect_timeout=5,
    ) as connection, connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(settings.collection_name))
        )
    LOGGER.info("Reset isolated index %s", settings.collection_name)


def usage_metadata(message: Any) -> dict[str, int]:
    raw = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(raw.get("input_tokens", 0) or 0),
        "output_tokens": int(raw.get("output_tokens", 0) or 0),
        "total_tokens": int(raw.get("total_tokens", 0) or 0),
    }


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part)
    return str(content)


def index_documents(
    questions: list[dict[str, Any]],
    *,
    pdf_directory: Path,
    summary_repository: DocumentRepository,
    raw_repository: DocumentRepository,
    settings: Settings,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    selected_docs = list(dict.fromkeys(str(item["doc_name"]) for item in questions))
    file_hashes: dict[str, str] = {}
    ingestion: list[dict[str, Any]] = []
    summarizer = ElementSummarizer(
        model_name=settings.chat_model,
        api_key=settings.openai_api_key,
        max_concurrency=settings.summary_concurrency,
    )

    for position, doc_name in enumerate(selected_docs, start=1):
        pdf_path = find_pdf(pdf_directory, doc_name)
        filename = pdf_path.name
        file_hash = sha256_file(pdf_path)
        page_count = len(PdfReader(pdf_path).pages)
        file_hashes[doc_name] = file_hash
        summary_cached = summary_repository.indexed_metadata(file_hash)
        raw_cached = raw_repository.indexed_metadata(file_hash)
        LOGGER.info("[%s/%s] Indexing %s", position, len(selected_docs), filename)

        if summary_cached and raw_cached:
            element_counts = summary_repository.indexed_element_counts(file_hash)
            ingestion.append(
                {
                    "doc_name": doc_name,
                    "filename": filename,
                    "file_hash": file_hash,
                    "element_count": int(summary_cached.get("element_count", 0)),
                    "page_count": page_count,
                    "text_elements": element_counts.get("text", 0),
                    "table_elements": element_counts.get("table", 0),
                    "extraction_seconds": 0.0,
                    "summary_seconds": 0.0,
                    "summary_index_seconds": 0.0,
                    "raw_index_seconds": 0.0,
                    "cached": True,
                }
            )
            continue

        extraction_start = time.perf_counter()
        elements = extract_pdf_elements(
            pdf_path,
            filename=filename,
            file_hash=file_hash,
            strategy=settings.partition_strategy,
        )
        extraction_seconds = time.perf_counter() - extraction_start

        summary_seconds = 0.0
        summary_index_seconds = 0.0
        if not summary_cached:
            summary_start = time.perf_counter()
            summaries = summarizer.summarize(elements)
            summary_seconds = time.perf_counter() - summary_start
            index_start = time.perf_counter()
            summary_repository.add(elements, summaries)
            summary_repository.mark_indexed(
                file_hash=file_hash,
                filename=filename,
                element_count=len(elements),
            )
            summary_index_seconds = time.perf_counter() - index_start

        raw_index_seconds = 0.0
        if not raw_cached:
            raw_start = time.perf_counter()
            raw_repository.add(elements, [element.text for element in elements])
            raw_repository.mark_indexed(
                file_hash=file_hash,
                filename=filename,
                element_count=len(elements),
            )
            raw_index_seconds = time.perf_counter() - raw_start

        ingestion.append(
            {
                "doc_name": doc_name,
                "filename": filename,
                "file_hash": file_hash,
                "element_count": len(elements),
                "page_count": page_count,
                "text_elements": sum(element.kind == "text" for element in elements),
                "table_elements": sum(element.kind == "table" for element in elements),
                "extraction_seconds": extraction_seconds,
                "summary_seconds": summary_seconds,
                "summary_index_seconds": summary_index_seconds,
                "raw_index_seconds": raw_index_seconds,
                "cached": False,
            }
        )
        LOGGER.info(
            "Indexed %s elements from %s in %.1fs",
            len(elements),
            filename,
            extraction_seconds + summary_seconds + summary_index_seconds + raw_index_seconds,
        )

    return file_hashes, ingestion


def relevant_pages(question: dict[str, Any], page_offset: int) -> set[tuple[str, int]]:
    pages: set[tuple[str, int]] = set()
    for evidence in question.get("evidence", []):
        doc_name = evidence.get("evidence_doc_name") or evidence.get("doc_name")
        if not doc_name:
            raise ValueError(f"Evidence is missing its document name: {evidence!r}")
        pages.add(
            (
                f"{doc_name}.pdf",
                int(evidence["evidence_page_num"]) + page_offset,
            )
        )
    return pages


def retrieved_pages(retrieved: list[RetrievedElement]) -> list[tuple[str, int | None]]:
    return [(item.element.source, item.element.page) for item in retrieved]


def answer_from_retrieved(
    question: str,
    retrieved: list[RetrievedElement],
    *,
    model: ChatOpenAI,
    max_characters: int,
) -> dict[str, Any]:
    context, citations = build_context(retrieved, max_characters=max_characters)
    if not context:
        return {
            "answer": "I could not find relevant content in the selected documents.",
            "context": "",
            "sources": [],
            "generation_seconds": 0.0,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    prompt = f"Document context:\n{context}\n\nRecent conversation:\n(none)\n\nQuestion: {question}"
    started = time.perf_counter()
    response = model.invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    generation_seconds = time.perf_counter() - started
    return {
        "answer": content_text(response.content),
        "context": context,
        "sources": [citation.label for citation in citations],
        "generation_seconds": generation_seconds,
        "usage": usage_metadata(response),
    }


def build_judge(model_name: str, api_key: str) -> Any:
    model = ChatOpenAI(model=model_name, api_key=api_key)
    return model.with_structured_output(JudgeGrade, include_raw=True)


def judge_answer(
    judge: Any,
    *,
    question: str,
    gold_answer: str,
    justification: str,
    candidate_answer: str,
    retrieved_context: str,
    expected_answerable: bool,
) -> dict[str, Any]:
    rubric = """Grade a retrieval-augmented answer using integer scores from 0 to 2.

Correctness: 2 = fully agrees with the reference; 1 = partly correct or materially incomplete;
0 = incorrect. Groundedness: 2 = every factual claim is supported by the retrieved context;
1 = mostly supported with a minor unsupported claim; 0 = materially unsupported. Citation
correctness: 2 = bracketed citation numbers exist and support the claims they follow; 1 = mixed;
0 = missing, invalid, or unsupported. Citation completeness: 2 = all material factual claims are
cited; 1 = some are cited; 0 = none are cited. Set abstained=true only when the candidate clearly
says the available documents do not contain enough evidence. When expected_answerable is false,
abstention is the correct behavior and correctness should be 2 for a clear abstention. Do not use
outside knowledge."""
    payload = (
        f"Expected answerable: {expected_answerable}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{gold_answer}\n\n"
        f"Reference justification:\n{justification or '(none)'}\n\n"
        f"Retrieved context:\n{retrieved_context or '(none)'}\n\n"
        f"Candidate answer:\n{candidate_answer}"
    )
    started = time.perf_counter()
    result = judge.invoke([SystemMessage(content=rubric), HumanMessage(content=payload)])
    judge_seconds = time.perf_counter() - started
    parsed = result.get("parsed")
    if parsed is None:
        raise ValueError(f"Judge did not return a parsed grade: {result.get('parsing_error')}")
    raw = result.get("raw")
    return {
        **parsed.model_dump(),
        "judge_seconds": judge_seconds,
        "usage": usage_metadata(raw),
    }


def evaluate_question(
    question: dict[str, Any],
    *,
    repositories: dict[str, DocumentRepository],
    answer_models: dict[str, ChatOpenAI],
    judge: Any | None,
    allowed_hashes: list[str],
    relevant: set[tuple[str, int]],
    settings: Settings,
    generate: bool,
    expected_answerable: bool,
) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system_name, repository in repositories.items():
        started = time.perf_counter()
        retrieved = repository.search(
            str(question["question"]),
            top_k=settings.top_k,
            allowed_hashes=allowed_hashes,
        )
        retrieval_seconds = time.perf_counter() - started
        ranking = ranking_metrics(retrieved_pages(retrieved), relevant, k=settings.top_k)
        system_result: dict[str, Any] = {
            "ranking": ranking,
            "retrieval_seconds": retrieval_seconds,
            "retrieved": [
                {
                    "rank": item.rank,
                    "source": item.element.source,
                    "page": item.element.page,
                    "kind": item.element.kind,
                    "parent_id": item.element.parent_id,
                }
                for item in retrieved
            ],
        }
        if generate:
            generated = answer_from_retrieved(
                str(question["question"]),
                retrieved,
                model=answer_models[system_name],
                max_characters=settings.max_context_characters,
            )
            system_result.update(
                {
                    "answer": generated["answer"],
                    "sources": generated["sources"],
                    "generation_seconds": generated["generation_seconds"],
                    "end_to_end_seconds": retrieval_seconds + generated["generation_seconds"],
                    "answer_usage": generated["usage"],
                }
            )
            if judge is not None:
                system_result["judge"] = judge_answer(
                    judge,
                    question=str(question["question"]),
                    gold_answer=str(question["answer"]),
                    justification=str(question.get("justification") or ""),
                    candidate_answer=str(generated["answer"]),
                    retrieved_context=str(generated["context"]),
                    expected_answerable=expected_answerable,
                )
        systems[system_name] = system_result
    return systems


def aggregate_system(
    cases: list[dict[str, Any]],
    system_name: str,
    *,
    include_ranking: bool = True,
) -> dict[str, Any]:
    results = [case["systems"][system_name] for case in cases if system_name in case["systems"]]
    aggregate: dict[str, Any] = {"case_count": len(results)}

    if include_ranking:
        for metric in ("hit", "recall", "mrr", "ndcg"):
            values = [float(result["ranking"][metric]) for result in results]
            low, high = bootstrap_mean_interval(values)
            aggregate[f"{metric}_at_5"] = mean(values)
            aggregate[f"{metric}_at_5_ci95"] = [low, high]

    retrieval_times = [float(result["retrieval_seconds"]) for result in results]
    aggregate["retrieval_latency_seconds"] = {
        "p50": percentile(retrieval_times, 0.50),
        "p95": percentile(retrieval_times, 0.95),
    }

    generated = [result for result in results if "answer" in result]
    if generated:
        generation_times = [float(result["generation_seconds"]) for result in generated]
        end_to_end_times = [float(result["end_to_end_seconds"]) for result in generated]
        aggregate["generation_latency_seconds"] = {
            "p50": percentile(generation_times, 0.50),
            "p95": percentile(generation_times, 0.95),
        }
        aggregate["end_to_end_latency_seconds"] = {
            "p50": percentile(end_to_end_times, 0.50),
            "p95": percentile(end_to_end_times, 0.95),
        }
        aggregate["answer_tokens"] = {
            key: sum(int(result["answer_usage"][key]) for result in generated)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }

    judged = [result["judge"] for result in generated if "judge" in result]
    if judged:
        for metric in (
            "correctness",
            "groundedness",
            "citation_correctness",
            "citation_completeness",
        ):
            values = [float(grade[metric]) / 2.0 for grade in judged]
            low, high = bootstrap_mean_interval(values)
            aggregate[metric] = mean(values)
            aggregate[f"{metric}_ci95"] = [low, high]
        aggregate["abstention_rate"] = mean([float(grade["abstained"]) for grade in judged])
        aggregate["judge_tokens"] = {
            key: sum(int(grade["usage"][key]) for grade in judged)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
    return aggregate


def create_abstention_questions(
    questions: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for question in questions:
        doc_name = str(question["doc_name"])
        if doc_name not in seen_docs:
            selected.append(question)
            seen_docs.add(doc_name)
        if len(selected) >= limit:
            break
    return selected


def report_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def report_seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}s"


def write_report(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    raw = summary["systems"]["raw"]
    parent_summary = summary["systems"]["summary"]
    abstention = summary.get("abstention", {})
    corpus = result["corpus"]
    config = result["configuration"]

    def row(label: str, key: str, *, percent: bool = True) -> str:
        formatter = report_percent if percent else report_seconds
        raw_value = raw.get(key)
        summary_value = parent_summary.get(key)
        if raw_value is None or summary_value is None:
            difference = "—"
        elif percent:
            difference = f"{(summary_value - raw_value) * 100:+.1f} pp"
        else:
            difference = f"{summary_value - raw_value:+.2f}s"
        return (
            f"| {label} | {formatter(raw_value)} | {formatter(summary_value)} | {difference} |"
        )

    hit_delta = (parent_summary["hit_at_5"] - raw["hit_at_5"]) * 100
    correctness_delta = (parent_summary["correctness"] - raw["correctness"]) * 100

    lines = [
        "# CiteCraft FinanceBench results — local snapshot",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "## Scope",
        "",
        (
            f"This run evaluated **{summary['question_count']} human-annotated questions** across "
            f"**{corpus['document_count']} public financial reports** from the "
            f"[FinanceBench open-source sample]({result['dataset']['source_url']}). The raw-text "
            "baseline and CiteCraft used the same PDFs, question set, embedding model, answer "
            "model, and retrieval depth."
        ),
        "",
        "## Measured outcome",
        "",
        (
            "Compared with raw-element embeddings, CiteCraft's summary-to-parent retrieval "
            f"changed evidence-page Hit@5 by **{hit_delta:+.1f} percentage points** and "
            f"model-graded answer correctness by **{correctness_delta:+.1f} percentage points** "
            "on this run."
        ),
        "",
        "## Results",
        "",
        "| Metric | Raw-element embeddings | CiteCraft summary-to-parent | Difference |",
        "| --- | ---: | ---: | ---: |",
        row("Evidence-page Hit@5", "hit_at_5"),
        row("Evidence-page Recall@5", "recall_at_5"),
        row("MRR@5", "mrr_at_5"),
        row("nDCG@5", "ndcg_at_5"),
        row("Answer correctness", "correctness"),
        row("Groundedness", "groundedness"),
        row("Citation correctness", "citation_correctness"),
        row("Citation completeness", "citation_completeness"),
        row(
            "Retrieval latency P95",
            "_retrieval_p95",
            percent=False,
        ),
        row(
            "End-to-end latency P95",
            "_end_to_end_p95",
            percent=False,
        ),
        "",
    ]
    if abstention:
        lines.extend(
            [
                "## Source-withheld abstention",
                "",
                (
                    f"One question from each of **{abstention['case_count']} documents** was rerun "
                    "with every PDF from the target company excluded from retrieval."
                ),
                "",
                "| System | Correct abstention rate |",
                "| --- | ---: |",
                (
                    "| Raw-element embeddings | "
                    f"{report_percent(abstention['raw']['abstention_rate'])} |"
                ),
                (
                    "| CiteCraft summary-to-parent | "
                    f"{report_percent(abstention['summary']['abstention_rate'])} |"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Corpus and configuration",
            "",
            f"- Documents: **{corpus['document_count']}**",
            f"- PDF pages: **{corpus['page_count']}**",
            f"- Extracted elements: **{corpus['element_count']}**",
            f"- Text elements: **{corpus['text_elements']}**",
            f"- Table elements: **{corpus['table_elements']}**",
            f"- Parser: `{config['partition_strategy']}`",
            (
                f"- Embedding model: `{config['embedding_model']}` "
                f"({config['embedding_dimensions']} dimensions)"
            ),
            f"- Answer and judge model: `{config['chat_model']}`",
            f"- Retrieval depth: `{config['top_k']}`",
            "",
        ]
    )
    if corpus.get("resumed_document_count"):
        lines.extend(
            [
                (
                    f"This result resumed **{corpus['resumed_document_count']} already-indexed "
                    "documents** after a workstation interruption. Corpus and query measurements "
                    "are complete; aggregate ingestion time is intentionally unset rather than "
                    "reporting a partial duration."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Evaluation protocol",
            "",
            "1. The manifest deterministically selects all 32 annotated questions attached to "
            "10 high-density FinanceBench documents and pins the source repository revision.",
            "2. Unstructured extracts citation-ready text and table parents. The baseline embeds "
            "raw parents; CiteCraft embeds one compact summary per parent and returns the complete "
            "parent after vector search.",
            "3. Both systems use the same PDFs, 1,536-dimensional embedding model, answer model, "
            "prompt, five-result cutoff, and 24,000-character context budget.",
            "4. Retrieval is scored by matching ranked filename/page pairs against FinanceBench's "
            "human-annotated evidence pages. The zero-indexed labels are converted to the "
            "one-indexed page metadata produced by Unstructured.",
            "5. The recorded answer model also applies a structured 0–2 rubric for correctness, "
            "groundedness, citation correctness, and citation completeness; scores in the table "
            "are normalized to 0–100%.",
            "6. One question per document is rerun after excluding every selected PDF from the "
            "target company, measuring whether the system explicitly abstains without its source.",
            "",
            "Retrieval metrics are deterministic for the recorded index. Answer and judge metrics "
            "are model-graded observations from one run at the model's default temperature. The "
            "JSON artifact contains per-question records, token usage, errors, and deterministic "
            "95% bootstrap intervals.",
            "",
            "The result applies to this pinned dataset subset and recorded configuration; it is "
            "not a general accuracy claim for arbitrary PDFs.",
            "",
            "## Artifacts",
            "",
            "- Selection and pinned revision: `benchmarks/financebench_manifest.json`",
            "- Machine-readable result: `tmp/financebench/financebench_results.json`",
            "- Benchmark runner: `benchmarks/run_financebench.py`",
            "",
            "## Reproduce",
            "",
            "```bash",
            "git clone https://github.com/patronus-ai/financebench.git /tmp/citecraft-financebench",
            (
                "git -C /tmp/citecraft-financebench checkout "
                f"{result['dataset']['source_revision']}"
            ),
            "docker compose up -d",
            "python benchmarks/run_financebench.py \\",
            "  --financebench-dir /tmp/citecraft-financebench \\",
            "  --reset-index",
            "```",
            "",
            f"FinanceBench revision: `{result['dataset']['source_revision']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = load_json(args.manifest)
    dataset_path = args.financebench_dir / "data" / "financebench_open_source.jsonl"
    pdf_directory = args.financebench_dir / "pdfs"
    if not dataset_path.is_file() or not pdf_directory.is_dir():
        raise FileNotFoundError("The supplied path is not a FinanceBench repository checkout.")

    actual_revision = checkout_revision(args.financebench_dir)
    expected_revision = str(manifest["source_revision"])
    if actual_revision != expected_revision and not args.allow_revision_mismatch:
        raise ValueError(
            f"FinanceBench revision mismatch: expected {expected_revision}, got {actual_revision}. "
            "Check out the pinned revision or pass --allow-revision-mismatch."
        )

    all_questions = load_questions(dataset_path)
    selected_ids = list(manifest["question_ids"])
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be greater than zero")
        selected_ids = selected_ids[: args.limit]
    questions = [all_questions[question_id] for question_id in selected_ids]
    LOGGER.info(
        "Selected %s questions across %s documents",
        len(questions),
        len({question["doc_name"] for question in questions}),
    )

    base_settings = Settings.from_env()
    base_settings.validate()
    summary_settings = replace(
        base_settings,
        redis_url=redis_url_with_database(base_settings.redis_url, 1),
        collection_name="citecraft_benchmark_summary",
    )
    raw_settings = replace(
        base_settings,
        redis_url=redis_url_with_database(base_settings.redis_url, 2),
        collection_name="citecraft_benchmark_raw",
    )
    summary_repository = DocumentRepository(summary_settings)
    raw_repository = DocumentRepository(raw_settings)
    if args.reset_index:
        reset_repository(summary_repository, summary_settings)
        reset_repository(raw_repository, raw_settings)

    file_hashes, ingestion = index_documents(
        questions,
        pdf_directory=pdf_directory,
        summary_repository=summary_repository,
        raw_repository=raw_repository,
        settings=base_settings,
    )
    allowed_hashes = [file_hashes[doc_name] for doc_name in file_hashes]
    repositories = {"raw": raw_repository, "summary": summary_repository}
    answer_models = {
        name: ChatOpenAI(
            model=base_settings.chat_model,
            api_key=base_settings.openai_api_key,
        )
        for name in repositories
    }
    judge = None
    if not args.skip_generation and not args.skip_judge:
        judge = build_judge(base_settings.chat_model, base_settings.openai_api_key)

    page_offset = int(manifest["evidence_page_offset"])
    cases: list[dict[str, Any]] = []
    for position, question in enumerate(questions, start=1):
        LOGGER.info("[%s/%s] Evaluating %s", position, len(questions), question["financebench_id"])
        relevant = relevant_pages(question, page_offset)
        try:
            systems = evaluate_question(
                question,
                repositories=repositories,
                answer_models=answer_models,
                judge=judge,
                allowed_hashes=allowed_hashes,
                relevant=relevant,
                settings=base_settings,
                generate=not args.skip_generation,
                expected_answerable=True,
            )
            error = None
        except Exception as exc:
            LOGGER.exception("Evaluation failed for %s", question["financebench_id"])
            systems = {}
            error = f"{type(exc).__name__}: {exc}"
        cases.append(
            {
                "question_id": question["financebench_id"],
                "question": question["question"],
                "reference_answer": question["answer"],
                "reference_justification": question.get("justification"),
                "doc_name": question["doc_name"],
                "company": question["company"],
                "question_type": question["question_type"],
                "question_reasoning": question.get("question_reasoning"),
                "relevant_pages": [
                    {"source": source, "page": page} for source, page in sorted(relevant)
                ],
                "systems": systems,
                "error": error,
            }
        )

    abstention_cases: list[dict[str, Any]] = []
    if not args.skip_generation and args.abstention_limit > 0:
        company_documents: dict[str, set[str]] = defaultdict(set)
        for item in questions:
            company_documents[str(item["company"])].add(str(item["doc_name"]))
        abstention_questions = create_abstention_questions(questions, args.abstention_limit)
        for position, question in enumerate(abstention_questions, start=1):
            LOGGER.info(
                "[%s/%s] Evaluating source-withheld abstention %s",
                position,
                len(abstention_questions),
                question["financebench_id"],
            )
            excluded_docs = company_documents[str(question["company"])]
            withheld_hashes = [
                file_hash
                for doc_name, file_hash in file_hashes.items()
                if doc_name not in excluded_docs
            ]
            try:
                systems = evaluate_question(
                    question,
                    repositories=repositories,
                    answer_models=answer_models,
                    judge=judge,
                    allowed_hashes=withheld_hashes,
                    relevant=set(),
                    settings=base_settings,
                    generate=True,
                    expected_answerable=False,
                )
                error = None
            except Exception as exc:
                LOGGER.exception("Abstention evaluation failed for %s", question["financebench_id"])
                systems = {}
                error = f"{type(exc).__name__}: {exc}"
            abstention_cases.append(
                {
                    "question_id": question["financebench_id"],
                    "question": question["question"],
                    "reference_answer": question["answer"],
                    "doc_name": question["doc_name"],
                    "company": question["company"],
                    "excluded_documents": sorted(excluded_docs),
                    "systems": systems,
                    "error": error,
                }
            )

    successful_cases = [case for case in cases if not case["error"] and case["systems"]]
    system_summary = {
        name: aggregate_system(successful_cases, name) for name in repositories
    }
    for aggregate in system_summary.values():
        aggregate["_retrieval_p95"] = aggregate["retrieval_latency_seconds"]["p95"]
        aggregate["_end_to_end_p95"] = aggregate.get("end_to_end_latency_seconds", {}).get("p95")

    abstention_summary: dict[str, Any] = {}
    successful_abstention = [
        case for case in abstention_cases if not case["error"] and case["systems"]
    ]
    if successful_abstention:
        abstention_summary = {
            "case_count": len(successful_abstention),
            **{
                name: aggregate_system(
                    successful_abstention,
                    name,
                    include_ranking=False,
                )
                for name in repositories
            },
        }

    uncached_ingestion = [item for item in ingestion if not item["cached"]]
    resumed_document_count = len(ingestion) - len(uncached_ingestion)
    corpus = {
        "document_count": len(file_hashes),
        "page_count": sum(int(item["page_count"]) for item in ingestion),
        "element_count": sum(int(item["element_count"]) for item in ingestion),
        "text_elements": sum(int(item["text_elements"] or 0) for item in ingestion),
        "table_elements": sum(int(item["table_elements"] or 0) for item in ingestion),
        "ingestion_seconds": (
            None
            if resumed_document_count
            else sum(
                float(item[key])
                for item in uncached_ingestion
                for key in (
                    "extraction_seconds",
                    "summary_seconds",
                    "summary_index_seconds",
                    "raw_index_seconds",
                )
            )
        ),
        "measured_ingestion_document_count": len(uncached_ingestion),
        "resumed_document_count": resumed_document_count,
        "documents": ingestion,
    }
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": manifest["benchmark"],
            "source_url": manifest["source_url"],
            "source_revision": actual_revision,
            "selection_method": manifest["selection_method"],
        },
        "configuration": {
            "chat_model": base_settings.chat_model,
            "embedding_model": base_settings.embedding_model,
            "embedding_dimensions": base_settings.embedding_dimensions,
            "partition_strategy": base_settings.partition_strategy,
            "top_k": base_settings.top_k,
            "max_context_characters": base_settings.max_context_characters,
            "summary_concurrency": base_settings.summary_concurrency,
            "temperature": "model_default",
            "question_limit": args.limit,
            "generation_enabled": not args.skip_generation,
            "judge_enabled": judge is not None,
        },
        "corpus": corpus,
        "summary": {
            "question_count": len(successful_cases),
            "failed_question_count": len(cases) - len(successful_cases),
            "systems": system_summary,
            "abstention": abstention_summary,
        },
        "cases": cases,
        "abstention_cases": abstention_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, args.report)
    LOGGER.info("Wrote %s and %s", args.output, args.report)


if __name__ == "__main__":
    main()
