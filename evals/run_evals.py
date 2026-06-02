"""
RAGAS Evaluation Script for Agentic RAG using Phoenix Experiments.

Runs the RAG pipeline against a gold dataset as a Phoenix Experiment,
with RAGAS metrics as evaluators. Results are tracked in Phoenix UI
under Datasets & Experiments, and also saved locally as JSON/CSV.

Usage:
    python -m evals.run_evals --collection <name>
    python -m evals.run_evals --collection <name> --run-name "my-experiment"
"""

import argparse
import asyncio
import warnings

# Suppress noisy third-party deprecation warnings
warnings.filterwarnings("ignore")
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd

from src.agentic_rag.agent_workflow import AgenticRAG
from src.core.weaviate_client import init_weaviate, close_weaviate, get_weaviate_client
from src.core.logger import setup_logging

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://127.0.0.1:4317"
os.environ["PHOENIX_BASE_URL"] = "http://127.0.0.1:6006"

from phoenix.otel import register as phoenix_register
from phoenix.client import Client as PhoenixClient, AsyncClient
from openinference.instrumentation.langchain import LangChainInstrumentor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Gold dataset loading
# ---------------------------------------------------------------------------

def load_gold_dataset(path: Path) -> list[dict]:
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


# ---------------------------------------------------------------------------
# Context extraction from retrieved documents
# ---------------------------------------------------------------------------

def format_contexts(retrieved_documents: list) -> list[str]:
    """Extract text representations from Weaviate document objects.

    - Text chunks  → ``text`` field
    - Table chunks → ``text`` field (HTML is fine for the LLM judge)
    - Image chunks → ``caption`` field (the summarized description)
    """
    contexts = []
    for doc in retrieved_documents:
        props = doc.properties if hasattr(doc, "properties") else doc
        doc_type = props.get("type", "")

        if doc_type == "Image":
            text = props.get("text", "")
            caption = props.get("caption", "")
            # Prefer text (LLM summary) over caption (short label)
            if text and text != caption:
                ctx = f"[IMAGE] {caption}\n{text}" if caption else text
            else:
                ctx = f"[IMAGE] {caption}" if caption else text
        else:
            ctx = props.get("text", "")

        if ctx.strip():
            contexts.append(ctx)
    return contexts


# ---------------------------------------------------------------------------
# RAG pipeline execution (direct component calls, no intent router)
# ---------------------------------------------------------------------------

async def run_rag_pipeline(
    rag: AgenticRAG, question: str, collection_name: str
) -> dict:
    """Execute query_resolver → query_decomposer → retriever → rag_generator."""

    state = {
        "query": question,
        "collection_name": collection_name,
        "messages": [],
    }

    # 1. Resolve references in the query
    state.update(rag.query_resolver(state))

    # 2. Decompose into sub-queries
    state.update(rag.query_decomposer(state))

    # 3. Retrieve documents from Weaviate
    state.update(rag.retriever(state))

    # 4. Generate answer (async)
    state.update(await rag.rag_generator(state))

    # Extract generated answer from the last AI message
    response_text = ""
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        response_text = (
            last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        )

    return {
        "response": response_text,
        "retrieved_contexts": format_contexts(state.get("retrieved_documents", [])),
        "resolved_query": state.get("resolved_query", question),
        "decomposed_queries": state.get("decomposed_queries", []),
    }


# ---------------------------------------------------------------------------
# RAGAS evaluation retry helper
# ---------------------------------------------------------------------------

async def _eval_with_retry(func, max_retries=5, **kwargs):
    """Call a RAGAS async scorer with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            result = await func(**kwargs)
            val = getattr(result, 'value', None)
            if val is None:
                try:
                    val = float(result)
                except (ValueError, TypeError):
                    return None
            if val is not None and not math.isnan(val):
                return round(float(val), 4)
            return None
        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in ('429', 'Rate limit', '502', 'Timeout')):
                wait_time = (attempt + 1) * 10.0
                print(f"    [Eval API] Rate limit. Waiting {wait_time:.1f}s... (Attempt {attempt+1})")
                await asyncio.sleep(wait_time)
            else:
                print(f"    Eval error: {e}")
                return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation via Phoenix Experiments"
    )
    parser.add_argument(
        "--collection", required=True,
        help="Weaviate collection name to query",
    )
    parser.add_argument(
        "--run-name", type=str, default="baseline",
        help="Name of this experiment (visible in Phoenix UI)",
    )
    parser.add_argument(
        "--dry-run", type=int, nargs="?", const=1, default=False,
        help="Run on N samples only for quick testing (default: 1)",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to the JSONL gold dataset file",
    )
    args = parser.parse_args()

    setup_logging(log_file="logs/run_evals.log")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phoenix automatically handles experiment traces!

    phoenix_register()
    LangChainInstrumentor().instrument()
    px_client = PhoenixClient(base_url="http://127.0.0.1:6006")
    px_async_client = AsyncClient(base_url="http://127.0.0.1:6006")

    # ---- Load and upload gold dataset to Phoenix ---------------------------
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    
    gold_data = load_gold_dataset(dataset_path)
    print(f"Loaded {len(gold_data)} samples from {dataset_path.name}")

    dataset_name = dataset_path.stem
    try:
        px_dataset = px_client.datasets.create_dataset(
            name=dataset_name,
            dataframe=pd.DataFrame(gold_data),
            input_keys=["user_input"],
            output_keys=["reference_answer"],
            metadata_keys=["question_type"],
        )
        print(f"Dataset synced to Phoenix: {px_dataset.id}")
    except Exception as e:
        print(f"Dataset sync note: {e}")
        px_dataset = px_client.datasets.get_dataset(dataset=dataset_name)
        print(f"Using existing dataset: {px_dataset.id}")

    # ---- Initialize infrastructure ----------------------------------------
    init_weaviate()
    
    client = get_weaviate_client()
    if not client.collections.exists(args.collection):
        print(f"\n❌ ERROR: Weaviate collection '{args.collection}' does not exist!")
        print("Please ensure you have indexed your data or check the spelling.")
        close_weaviate()
        sys.exit(1)

    try:
        rag = AgenticRAG()
        collection_name = args.collection

        # Collector for local file output
        task_results = []

        # ---- Task: RAG pipeline execution ----------------------------------
        async def rag_task(example):
            """Phoenix experiment task: runs the RAG pipeline for one row."""
            question = example.input["user_input"]

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    result = await run_rag_pipeline(rag, question, collection_name)

                    # Side-effect: collect for local file output
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
                            print(f"\n  [Rate limit] Waiting {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                    raise

        # ---- Configure RAGAS scorers ---------------------------------------
        from ragas.metrics.collections import (
            ContextRecall, ContextPrecision, Faithfulness, AnswerRelevancy,
        )
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        eval_llm = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=8192)
        eval_emb = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)

        precision_scorer = ContextPrecision(llm=eval_llm)
        recall_scorer = ContextRecall(llm=eval_llm)
        faith_scorer = Faithfulness(llm=eval_llm)
        relevancy_scorer = AnswerRelevancy(llm=eval_llm, embeddings=eval_emb)

        # Collector for evaluation scores (keyed by question text)
        eval_scores = {}

        # ---- Phoenix evaluators (RAGAS wrappers) ---------------------------
        async def context_precision(input, output, expected):
            contexts = output.get("retrieved_contexts", []) if output else []
            if not contexts:
                raise ValueError("Task failed or returned no contexts")
            score = await _eval_with_retry(
                precision_scorer.ascore,
                user_input=input["user_input"],
                reference=expected["reference_answer"],
                retrieved_contexts=contexts,
            )
            eval_scores.setdefault(input["user_input"], {})["context_precision"] = score
            return score

        async def context_recall(input, output, expected):
            contexts = output.get("retrieved_contexts", []) if output else []
            if not contexts:
                raise ValueError("Task failed or returned no contexts")
            score = await _eval_with_retry(
                recall_scorer.ascore,
                user_input=input["user_input"],
                reference=expected["reference_answer"],
                retrieved_contexts=contexts,
            )
            eval_scores.setdefault(input["user_input"], {})["context_recall"] = score
            return score

        async def faithfulness(input, output, expected):
            contexts = output.get("retrieved_contexts", []) if output else []
            response = output.get("response", "") if output else ""
            if not contexts or not response:
                raise ValueError("Task failed or returned empty response/contexts")
            score = await _eval_with_retry(
                faith_scorer.ascore,
                user_input=input["user_input"],
                response=response,
                retrieved_contexts=contexts,
            )
            eval_scores.setdefault(input["user_input"], {})["faithfulness"] = score
            return score

        async def answer_relevancy(input, output, expected):
            response = output.get("response", "") if output else ""
            if not response:
                raise ValueError("Task failed or returned empty response")
            score = await _eval_with_retry(
                relevancy_scorer.ascore,
                user_input=input["user_input"],
                response=response,
            )
            eval_scores.setdefault(input["user_input"], {})["answer_relevancy"] = score
            return score

        # ---- Run Phoenix Experiment ----------------------------------------
        print(f"\nRunning experiment: {args.run_name}")
        print(f"  Collection:  {args.collection}")
        print(f"  LLM judge:   gpt-4o-mini")
        print(f"  Embeddings:  text-embedding-3-small")
        start = time.time()

        experiment = await px_async_client.experiments.run_experiment(
            dataset=px_dataset,
            task=rag_task,
            evaluators=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            experiment_name=args.run_name,
            concurrency=2, # Reduced to prevent rate limits and local GPU/CPU overload
            dry_run=args.dry_run if args.dry_run is not False else False,
        )

        elapsed = time.time() - start
        print(f"\nExperiment completed in {elapsed:.1f}s")
        print(f"  Samples processed: {len(task_results)}/{len(gold_data)}")

        # ---- Save local results --------------------------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        metric_cols = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]

        # Merge task results with evaluation scores
        for item in task_results:
            key = item["user_input"]
            scores = eval_scores.get(key, {})
            for col in metric_cols:
                item[col] = scores.get(col)

        # Aggregate means
        aggregate = {}
        for col in metric_cols:
            vals = [item.get(col) for item in task_results if item.get(col) is not None]
            aggregate[col] = round(sum(vals) / len(vals), 4) if vals else None

        # Per-question-type aggregation
        type_buckets = defaultdict(list)
        for item in task_results:
            type_buckets[item.get("question_type", "unknown")].append(item)

        per_type_scores = {}
        for qtype, items in sorted(type_buckets.items()):
            type_agg = {}
            for col in metric_cols:
                vals = [it.get(col) for it in items if it.get(col) is not None]
                type_agg[col] = round(sum(vals) / len(vals), 4) if vals else None
            type_agg["count"] = len(items)
            per_type_scores[qtype] = type_agg

        output = {
            "metadata": {
                "experiment_name": args.run_name,
                "timestamp": ts,
                "collection": args.collection,
                "total_samples": len(gold_data),
                "evaluated_samples": len(task_results),
                "elapsed_seconds": round(elapsed, 1),
                "judge_llm": "gpt-4o-mini",
                "embeddings": "text-embedding-3-small",
            },
            "aggregate_scores": aggregate,
            "per_type_scores": per_type_scores,
            "per_sample_results": task_results,
        }

        json_path = RESULTS_DIR / f"eval_results_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # CSV summary
        csv_path = RESULTS_DIR / f"eval_summary_{ts}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "score"])
            for metric_name, score in aggregate.items():
                writer.writerow([
                    metric_name,
                    f"{score:.4f}" if score is not None else "N/A",
                ])
            writer.writerow([])
            writer.writerow(["question_type", "count"] + metric_cols)
            for qtype, scores in per_type_scores.items():
                writer.writerow(
                    [qtype, scores["count"]] +
                    [f"{scores[col]:.4f}" if scores.get(col) is not None else "N/A" for col in metric_cols]
                )

        # ---- Print summary -------------------------------------------------
        print(f"\n{'=' * 50}")
        print("  RAGAS Evaluation Results")
        print(f"{'=' * 50}")
        for metric_name, score in aggregate.items():
            display = f"{score:.4f}" if score is not None else "N/A"
            print(f"  {metric_name:25s}: {display}")

        print(f"\n{'-' * 50}")
        print("  Scores by Question Type")
        print(f"{'-' * 50}")
        for qtype, scores in per_type_scores.items():
            print(f"  {qtype} (n={scores['count']})")
            for col in metric_cols:
                val = scores.get(col)
                display = f"{val:.4f}" if val is not None else "N/A"
                print(f"    {col:25s}: {display}")

        print(f"{'=' * 50}")
        print(f"  Detailed results → {json_path}")
        print(f"  Summary CSV      → {csv_path}")
        print(f"  Phoenix UI       → Datasets & Experiments → {args.run_name}")

    finally:
        close_weaviate()


if __name__ == "__main__":
    asyncio.run(main())
