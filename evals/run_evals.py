"""
RAGAS Evaluation Orchestrator for Agentic RAG using Phoenix Experiments.

Usage:
    python -m evals.run_evals --collection <name> --dataset <path>
    python -m evals.run_evals --mode api --collection <name> --dataset <path>
    python -m evals.run_evals --collection <name> --dataset <path> --eval-model gpt-4o --emb-model text-embedding-3-small
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

from src.agentic_rag.agent_workflow import AgenticRAG  # noqa: E402
from src.components.reranker import get_reranker  # noqa: E402
from src.components.retriever import resolve_reranker_mode  # noqa: E402
from src.core.config import get_settings  # noqa: E402
from src.core.weaviate_client import init_weaviate, close_weaviate, get_weaviate_client  # noqa: E402
from src.core.logger import setup_logging, get_logger  # noqa: E402

from evals.utils.io import load_gold_dataset, sample_varied_dataset, format_contexts, save_results  # noqa: E402
from evals.metrics.ragas_scorers import RagasEvaluators  # noqa: E402

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _check_api_ready(base_url: str, timeout: int) -> None:
    response = requests.get(f"{base_url.rstrip('/')}/ready", timeout=timeout)
    response.raise_for_status()


async def run_rag_pipeline(rag: AgenticRAG, question: str, collection_name: str) -> dict:
    state = {"query": question, "collection_name": collection_name, "messages": []}
    
    state.update(await rag.query_resolver(state))
    state.update(await rag.query_decomposer(state))
    state.update(await rag.retriever(state))
    state.update(await rag.rag_generator(state))

    response_text = ""
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        response_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    return {
        "response": response_text,
        "retrieved_contexts": format_contexts(state.get("retrieved_documents", [])),
        "resolved_query": state.get("resolved_query", question),
        "decomposed_queries": state.get("decomposed_queries", []),
    }


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
    parser.add_argument("--mode", choices=["api", "internal"], default="api", help="Evaluate Docker/API deployment or internal pipeline")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL when --mode api")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds when --mode api")
    parser.add_argument("--run-name", type=str, default="baseline", help="Name of this experiment")
    parser.add_argument("--dry-run", type=int, nargs="?", const=1, default=False, help="Run on N samples only")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of samples to evaluate concurrently")
    parser.add_argument("--eval-model", type=str, default="gpt-4o-mini", help="Judge LLM (default: gpt-4o-mini)")
    parser.add_argument("--emb-model", type=str, default="text-embedding-3-small", help="Judge Embedding (default: text-embedding-3-small)")
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
    if args.dry_run is not False:
        experiment_data = sample_varied_dataset(gold_data, args.dry_run)
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

    if args.mode == "api":
        try:
            _check_api_ready(args.base_url, args.timeout)
        except Exception as exc:
            logger.error("API is not ready at %s: %s", args.base_url, exc)
            sys.exit(1)
    else:
        init_weaviate()
        client = get_weaviate_client()
        if not client.collections.exists(args.collection):
            logger.error(f"Weaviate collection '{args.collection}' does not exist!")
            close_weaviate()
            sys.exit(1)

    try:
        rag = AgenticRAG() if args.mode == "internal" else None
        task_results = []
        evaluators = RagasEvaluators(judge_model=args.eval_model, embedding_model=args.emb_model)

        if args.mode == "internal":
            settings = get_settings()

        if (
            args.mode == "internal"
            and settings.warmup_reranker
            and resolve_reranker_mode(settings.reranker_mode) == "app"
        ):
            logger.info("Warming up app reranker before timed eval run")
            await asyncio.to_thread(get_reranker(settings).warmup)
            logger.info("App reranker warmup complete")

        async def rag_task(example):
            question = example.input["user_input"]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if args.mode == "api":
                        result = await run_api_pipeline(
                            question=question,
                            collection_name=args.collection,
                            base_url=args.base_url,
                            timeout=args.timeout,
                            session_id=f"eval-{args.run_name}",
                        )
                    else:
                        result = await run_rag_pipeline(rag, question, args.collection)
                    
                    question_type = ""
                    if hasattr(example, "metadata") and example.metadata:
                        question_type = example.metadata.get("question_type", "")

                    task_results.append({
                        "user_input": question,
                        "question_type": question_type,
                        "reference_answer": example.output.get("reference_answer", ""),
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
        logger.info(f"Mode:        {args.mode}")
        if args.mode == "api":
            logger.info(f"API URL:     {args.base_url}")
        logger.info(f"Collection:  {args.collection}")
        logger.info(f"LLM judge:   {args.eval_model}")
        logger.info(f"Embeddings:  {args.emb_model}")
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
        logger.info(f"Samples processed: {len(task_results)}/{len(gold_data)}")
        
        metadata = {
            "experiment_name": args.run_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "collection": args.collection,
            "mode": args.mode,
            "base_url": args.base_url if args.mode == "api" else None,
            "total_samples": len(gold_data),
            "dataset_samples": len(experiment_data),
            "sample_strategy": "varied_by_question_type_and_length" if args.dry_run is not False else "full_dataset",
            "concurrency": args.concurrency,
            "evaluated_samples": len(task_results),
            "elapsed_seconds": round(elapsed, 1),
            "judge_llm": args.eval_model,
            "embeddings": args.emb_model,
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
        logger.info(f"Detailed results → {json_path}")
        logger.info(f"Summary CSV      → {csv_path}")
        logger.info(f"Phoenix UI       → Datasets & Experiments → {args.run_name}")

    finally:
        if args.mode == "internal":
            close_weaviate()

if __name__ == "__main__":
    asyncio.run(main())
