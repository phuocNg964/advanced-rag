"""
RAGAS Evaluation Orchestrator for Agentic RAG using Phoenix Experiments.

Usage:
    python -m evals.run_evals --collection <name> --dataset <path>
    python -m evals.run_evals --collection <name> --dataset <path> --eval-model gpt-4o --emb-model text-embedding-3-small
"""

import argparse
import asyncio
import warnings
warnings.filterwarnings("ignore")

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://127.0.0.1:4317"
os.environ["PHOENIX_BASE_URL"] = "http://127.0.0.1:6006"

from phoenix.otel import register as phoenix_register
from phoenix.client import Client as PhoenixClient, AsyncClient
from openinference.instrumentation.langchain import LangChainInstrumentor

from src.agentic_rag.agent_workflow import AgenticRAG
from src.core.weaviate_client import init_weaviate, close_weaviate, get_weaviate_client
from src.core.logger import setup_logging, get_logger

from evals.utils.io import load_gold_dataset, format_contexts, save_results
from evals.metrics.ragas_scorers import RagasEvaluators

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

async def run_rag_pipeline(rag: AgenticRAG, question: str, collection_name: str) -> dict:
    state = {"query": question, "collection_name": collection_name, "messages": []}
    
    state.update(rag.query_resolver(state))
    state.update(rag.query_decomposer(state))
    state.update(rag.retriever(state))
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

async def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation via Phoenix Experiments")
    parser.add_argument("--collection", required=True, help="Weaviate collection name to query")
    parser.add_argument("--dataset", required=True, help="Path to the JSONL gold dataset file")
    parser.add_argument("--run-name", type=str, default="baseline", help="Name of this experiment")
    parser.add_argument("--dry-run", type=int, nargs="?", const=1, default=False, help="Run on N samples only")
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
    try:
        px_dataset = px_client.datasets.create_dataset(
            name=dataset_name,
            dataframe=pd.DataFrame(gold_data),
            input_keys=["user_input"],
            output_keys=["reference_answer"],
            metadata_keys=["question_type"],
        )
        logger.info(f"Dataset synced to Phoenix: {px_dataset.id}")
    except Exception as e:
        logger.warning(f"Dataset sync note: {e}")
        px_dataset = px_client.datasets.get_dataset(dataset=dataset_name)
        logger.info(f"Using existing dataset: {px_dataset.id}")

    init_weaviate()
    client = get_weaviate_client()
    if not client.collections.exists(args.collection):
        logger.error(f"Weaviate collection '{args.collection}' does not exist!")
        close_weaviate()
        sys.exit(1)

    try:
        rag = AgenticRAG()
        task_results = []
        evaluators = RagasEvaluators(judge_model=args.eval_model, embedding_model=args.emb_model)

        async def rag_task(example):
            question = example.input["user_input"]
            max_retries = 5
            for attempt in range(max_retries):
                try:
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
        logger.info(f"Collection:  {args.collection}")
        logger.info(f"LLM judge:   {args.eval_model}")
        logger.info(f"Embeddings:  {args.emb_model}")
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
            concurrency=2,
            dry_run=args.dry_run if args.dry_run is not False else False,
        )

        elapsed = time.time() - start
        logger.info(f"Experiment completed in {elapsed:.1f}s")
        logger.info(f"Samples processed: {len(task_results)}/{len(gold_data)}")
        
        metadata = {
            "experiment_name": args.run_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "collection": args.collection,
            "total_samples": len(gold_data),
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
        close_weaviate()

if __name__ == "__main__":
    asyncio.run(main())
