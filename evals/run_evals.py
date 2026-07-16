"""
RAGAS Evaluation Orchestrator for Agentic RAG using Phoenix Experiments.

Usage:
    python -m evals.run_evals --collection <name> --dataset <path>
    python -m evals.run_evals --collection <name> --dataset <path> --samples 20
"""

import argparse
import asyncio
import os
import re
import requests
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://127.0.0.1:4317"
os.environ["PHOENIX_BASE_URL"] = "http://127.0.0.1:6006"

from phoenix.otel import register as phoenix_register  # noqa: E402
from phoenix.client import Client as PhoenixClient, AsyncClient  # noqa: E402
from openinference.instrumentation.langchain import LangChainInstrumentor  # noqa: E402
from opentelemetry.propagate import inject  # noqa: E402

from src.core.logger import setup_logging, get_logger  # noqa: E402

from evals.utils.io import load_gold_dataset, sample_varied_dataset, format_contexts, save_results  # noqa: E402
from evals.metrics.ragas_scorers import RagasEvaluators  # noqa: E402

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
API_BASE_URL = "http://127.0.0.1:8000"
API_TIMEOUT_SECONDS = 120
JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


def _safe_session_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-")
    return cleaned or "sample"


def _eval_session_id(run_name: str, example) -> str:
    example_id = getattr(example, "id", None) or getattr(example, "example_id", None)
    if not example_id:
        example_id = example.input.get("user_input", "")[:80]
    return f"eval-{_safe_session_id(run_name)}-{_safe_session_id(str(example_id))}"


def _check_api_ready(base_url: str, timeout: int) -> None:
    response = requests.get(f"{base_url.rstrip('/')}/ready", timeout=timeout)
    response.raise_for_status()


async def run_api_pipeline(
    question: str,
    collection_name: str,
    base_url: str,
    timeout: int,
    session_id: str,
) -> dict:
    def _request_chat() -> dict:
        headers = {}
        inject(headers)
        response = requests.post(
            f"{base_url.rstrip('/')}/collections/{collection_name}/chat",
            json={"message": question, "session_id": session_id},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    data = await asyncio.to_thread(_request_chat)
    retrieved_documents = data.get("retrieved_documents", [])
    return {
        "response": data.get("response", ""),
        "retrieved_contexts": format_contexts(retrieved_documents),
    }


async def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation via Phoenix Experiments")
    parser.add_argument("--collection", required=True, help="Weaviate collection name to query")
    parser.add_argument("--dataset", required=True, help="Path to the JSONL gold dataset file")
    parser.add_argument("--run-name", type=str, default="baseline", help="Name of this experiment")
    parser.add_argument("--samples", type=int, default=None, help="Run on N varied samples only")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of samples to evaluate concurrently")
    args = parser.parse_args()

    setup_logging(log_file="logs/run_evals.log")

    phoenix_register()
    LangChainInstrumentor().instrument()
    px_client = PhoenixClient(base_url="http://127.0.0.1:6006")
    px_async_client = AsyncClient(base_url="http://127.0.0.1:6006")

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    gold_data = load_gold_dataset(dataset_path)
    logger.info(f"Loaded {len(gold_data)} samples from {dataset_path.name}")

    dataset_name = dataset_path.stem
    experiment_data = gold_data
    phoenix_dry_run = False
    if args.samples is not None:
        experiment_data = sample_varied_dataset(gold_data, args.samples)
        dataset_name = f"{dataset_name}_sample{len(experiment_data)}_varied"
        distribution = Counter(item.get("question_type", "unknown") for item in experiment_data)
        logger.info(f"Using varied sample of {len(experiment_data)}: {dict(distribution)}")

    try:
        px_dataset = px_client.datasets.create_dataset(
            name=dataset_name,
            dataframe=pd.DataFrame(experiment_data),
            input_keys=["user_input"],
            output_keys=["reference_answer"],
            metadata_keys=["question_type"],
        )
        logger.info(f"Dataset synced to Phoenix: {px_dataset.id}")
    except Exception as e:
        logger.warning(f"Dataset sync note: {e}")
        px_dataset = px_client.datasets.get_dataset(dataset=dataset_name)
        logger.info(f"Using existing dataset: {px_dataset.id}")

    try:
        _check_api_ready(API_BASE_URL, API_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.error("API is not ready at %s: %s", API_BASE_URL, exc)
        sys.exit(1)

    task_results = []
    evaluators = RagasEvaluators(judge_model=JUDGE_MODEL, embedding_model=EMBEDDING_MODEL)

    async def rag_task(example):
        question = example.input["user_input"]
        max_retries = 5
        query_started = time.perf_counter()
        for attempt in range(max_retries):
            try:
                attempt_started = time.perf_counter()
                session_id = _eval_session_id(args.run_name, example)
                result = await run_api_pipeline(
                    question=question,
                    collection_name=args.collection,
                    base_url=API_BASE_URL,
                    timeout=API_TIMEOUT_SECONDS,
                    session_id=session_id,
                )

                pipeline_latency_seconds = time.perf_counter() - attempt_started
                query_latency_seconds = time.perf_counter() - query_started

                question_type = ""
                if hasattr(example, "metadata") and example.metadata:
                    question_type = example.metadata.get("question_type", "")

                task_results.append({
                    "user_input": question,
                    "question_type": question_type,
                    "reference_answer": example.output.get("reference_answer", ""),
                    "latency_seconds": round(query_latency_seconds, 3),
                    "latency_ms": round(query_latency_seconds * 1000, 2),
                    "pipeline_latency_seconds": round(pipeline_latency_seconds, 3),
                    "attempts": attempt + 1,
                    "session_id": session_id,
                    **result,
                })
                return result

            except Exception as e:
                err_str = str(e)
                if any(k in err_str for k in ('429', 'Rate limit', '502', 'rate_limit_exceeded')):
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 15.0
                        match = re.search(r'try again in (?:(\d+)m)?([\d\.]+)s', err_str)
                        if match:
                            minutes = float(match.group(1)) if match.group(1) else 0.0
                            seconds = float(match.group(2))
                            wait_time = max(wait_time, minutes * 60 + seconds + 1.0)
                        logger.warning(f"[Rate limit] Waiting {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                raise

    logger.info(f"Running experiment: {args.run_name}")
    logger.info("Mode:        api")
    logger.info(f"API URL:     {API_BASE_URL}")
    logger.info("Pipeline:    Docker API full agent graph")
    logger.info(f"Collection:  {args.collection}")
    logger.info(f"LLM judge:   {JUDGE_MODEL}")
    logger.info(f"Embeddings:  {EMBEDDING_MODEL}")
    logger.info(f"Concurrency: {args.concurrency}")
    start = time.time()

    await px_async_client.experiments.run_experiment(
        dataset=px_dataset,
        task=rag_task,
        evaluators=[
            evaluators.context_precision,
            evaluators.context_recall,
            evaluators.faithfulness,
            evaluators.answer_relevancy,
        ],
        experiment_name=args.run_name,
        concurrency=args.concurrency,
        dry_run=phoenix_dry_run,
    )

    elapsed = time.time() - start
    logger.info(f"Experiment completed in {elapsed:.1f}s")
    logger.info(f"Samples processed: {len(task_results)}/{len(experiment_data)}")

    metadata = {
        "experiment_name": args.run_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "collection": args.collection,
        "mode": "api",
        "base_url": API_BASE_URL,
        "pipeline": "docker_api_full_agent_graph",
        "total_samples": len(gold_data),
        "dataset_samples": len(experiment_data),
        "sample_strategy": "varied_by_question_type_and_length" if args.samples is not None else "full_dataset",
        "concurrency": args.concurrency,
        "evaluated_samples": len(task_results),
        "elapsed_seconds": round(elapsed, 1),
        "judge_llm": JUDGE_MODEL,
        "embeddings": EMBEDDING_MODEL,
    }

    json_path, csv_path, aggregate, per_type_scores = save_results(
        task_results, evaluators.eval_scores, metadata, RESULTS_DIR
    )

    logger.info("========== RAGAS Evaluation Results ==========")
    for metric_name, score in aggregate.items():
        display = f"{score:.4f}" if score is not None else "N/A"
        logger.info(f"{metric_name:25s}: {display}")

    logger.info("---------- Scores by Question Type ----------")
    for qtype, scores in per_type_scores.items():
        logger.info(f"{qtype} (n={scores['count']})")
        for col in ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]:
            val = scores.get(col)
            display = f"{val:.4f}" if val is not None else "N/A"
            logger.info(f"  {col:25s}: {display}")

    logger.info("==============================================")
    logger.info(f"Detailed results â†’ {json_path}")
    logger.info(f"Summary CSV      â†’ {csv_path}")
    logger.info(f"Phoenix UI       â†’ Datasets & Experiments â†’ {args.run_name}")


if __name__ == "__main__":
    asyncio.run(main())
