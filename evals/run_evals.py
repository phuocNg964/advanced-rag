"""
RAGAS Evaluation Script for Agentic RAG.

Runs the RAG pipeline directly (bypassing intent router) against a gold dataset,
then evaluates using RAGAS metrics: context recall, context precision,
faithfulness, and answer relevancy.

Usage:
    python -m evals.run_evals --collection <name>
    python -m evals.run_evals --collection <name> --dry-run
"""

import asyncio
import argparse
import csv
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from tqdm import tqdm

from src.agentic_rag.agent_workflow import AgenticRAG
from src.core.weaviate_client import init_weaviate, close_weaviate
from src.core.logger import setup_logging

# ---------------------------------------------------------------------------
# Gold dataset loading
# ---------------------------------------------------------------------------

GOLD_DATASET_PATH = PROJECT_ROOT / "data" / "gold" / "gold_dataset_v3.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_gold_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        "resolved_query": state.get("query", question),
        "decomposed_queries": state.get("decomposed_queries", []),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the gold dataset"
    )
    parser.add_argument(
        "--collection", help="Weaviate collection name to query (required unless --evaluate-from is used)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the RAG pipeline but skip the RAGAS evaluation step",
    )
    parser.add_argument(
        "--evaluate-from",
        type=str,
        help="Path to a previous dry-run JSON file to evaluate directly, skipping generation",
    )
    args = parser.parse_args()

    if not args.collection and not args.evaluate_from:
        parser.error("Either --collection or --evaluate-from must be provided.")

    setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load gold dataset ------------------------------------------------
    dataset = load_gold_dataset(GOLD_DATASET_PATH)
    print(f"Loaded {len(dataset)} samples from {GOLD_DATASET_PATH.name}")

    # ---- Initialise infrastructure ----------------------------------------
    if not args.evaluate_from:
        init_weaviate()

    try:
        raw_results: list[dict] = []

        if args.evaluate_from:
            print(f"Loading existing generation results from {args.evaluate_from}...")
            with open(args.evaluate_from, "r", encoding="utf-8") as f:
                raw_results = json.load(f)
            successful = [r for r in raw_results if "error" not in r]
            print(f"Loaded {len(successful)} successful samples for evaluation.")
        else:
            rag = AgenticRAG()  # LLMs init here; no postgres setup needed

            # ---- Run the RAG pipeline on every question -----------------------
            for item in tqdm(dataset, desc="Running RAG pipeline"):
                question = item["user_input"]
                
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        result = await run_rag_pipeline(rag, question, args.collection)
                        raw_results.append({
                            "user_input": question,
                            "question_type": item.get("question_type", ""),
                            "reference_answer": item["reference_answer"],
                            "resolved_query": result["resolved_query"],
                            "decomposed_queries": result["decomposed_queries"],
                            "response": result["response"],
                            "retrieved_contexts": result["retrieved_contexts"],
                        })
                        break  # Success, exit retry loop
                    except Exception as e:
                        err_str = str(e)
                        if '429' in err_str or 'Rate limit' in err_str or '502' in err_str or 'rate_limit_exceeded' in err_str:
                            if attempt < max_retries - 1:
                                wait_time = (attempt + 1) * 15.0
                                
                                # Try to parse Groq's wait time (e.g., 'try again in 2m21.8688s' or 'try again in 8.4s')
                                match = re.search(r'try again in (?:(\d+)m)?([\d\.]+)s', err_str)
                                if match:
                                    minutes = float(match.group(1)) if match.group(1) else 0.0
                                    seconds = float(match.group(2))
                                    wait_time = max(wait_time, minutes * 60 + seconds + 1.0)
                                    
                                print(f"\n  [LLM API] Rate limit hit. Waiting {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                                await asyncio.sleep(wait_time)
                                continue
                        
                        # If not a rate limit, or max retries exceeded
                        print(f"\n  ERROR [{question[:60]}…]: {e}")
                        raw_results.append({
                            "user_input": question,
                            "question_type": item.get("question_type", ""),
                            "reference_answer": item["reference_answer"],
                            "resolved_query": "",
                            "decomposed_queries": [],
                            "response": "",
                            "retrieved_contexts": [],
                            "error": str(e),
                        })
                        break

            successful = [r for r in raw_results if "error" not in r]
            print(f"\nPipeline complete: {len(successful)}/{len(dataset)} samples succeeded")

        # ---- Dry-run exit -------------------------------------------------
        if args.dry_run:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = RESULTS_DIR / f"dry_run_{ts}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(raw_results, f, indent=2, ensure_ascii=False)
            print(f"Dry-run results saved → {out_path}")
            return

        if not successful:
            print("No successful samples to evaluate. Exiting.")
            return

        # Filter out samples with empty contexts (would produce NaN scores)
        evaluable = [r for r in successful if r["retrieved_contexts"]]
        if len(evaluable) < len(successful):
            print(
                f"  ⚠ Skipping {len(successful) - len(evaluable)} samples "
                f"with empty retrieved contexts"
            )
        if not evaluable:
            print("No samples with retrieved contexts. Exiting.")
            return

        # ---- Configure RAGAS models ---------------------------------------
        from ragas.metrics.collections import (
            ContextRecall,
            ContextPrecision,
            Faithfulness,
            AnswerRelevancy,
        )
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        import os
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        evaluator_llm = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=8192)
        evaluator_embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)

        context_precision_scorer = ContextPrecision(llm=evaluator_llm)
        context_recall_scorer = ContextRecall(llm=evaluator_llm)
        faithfulness_scorer = Faithfulness(llm=evaluator_llm)
        answer_relevancy_scorer = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

        max_workers = 3
        sem = asyncio.Semaphore(max_workers)

        async def evaluate_sample(item):
            user_input = item["user_input"]
            reference = item["reference_answer"]
            response = item["response"] if item["response"] else "No answer generated."
            
            retrieved_contexts = item["retrieved_contexts"]
            if not retrieved_contexts:
                retrieved_contexts = ["(no context retrieved)"]
                
            async def _eval_retry(func, max_retries=5, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return await func(**kwargs)
                    except Exception as e:
                        err_str = str(e)
                        if '429' in err_str or 'Rate limit' in err_str or '502' in err_str or 'Timeout' in err_str:
                            wait_time = (attempt + 1) * 10.0
                            print(f"    [OpenAI API] Network/RateLimit issue. Waiting {wait_time}s... (Attempt {attempt+1})")
                            await asyncio.sleep(wait_time)
                        else:
                            return e
                return Exception("Max retries exceeded for OpenAI.")
            
            async with sem:
                precision_task = _eval_retry(context_precision_scorer.ascore,
                    user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts)
                
                recall_task = _eval_retry(context_recall_scorer.ascore,
                    user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts)
                
                faithfulness_task = _eval_retry(faithfulness_scorer.ascore,
                    user_input=user_input, response=response, retrieved_contexts=retrieved_contexts)
                
                relevancy_task = _eval_retry(answer_relevancy_scorer.ascore,
                    user_input=user_input, response=response)
                
                res = await asyncio.gather(
                    precision_task, recall_task, faithfulness_task, relevancy_task,
                    return_exceptions=True
                )
            
            def get_val(r):
                if isinstance(r, Exception):
                    print(f"Error evaluating sample: {r}")
                    return None
                val = getattr(r, 'value', None)
                if val is None:
                    try:
                        val = float(r)
                    except (ValueError, TypeError):
                        return None
                if val is not None and not math.isnan(val):
                    return round(float(val), 4)
                return None

            item["context_precision"] = get_val(res[0])
            item["context_recall"] = get_val(res[1])
            item["faithfulness"] = get_val(res[2])
            item["answer_relevancy"] = get_val(res[3])
            
            errors = [str(r) for r in res if isinstance(r, Exception)]
            if errors:
                item["error_msg"] = " | ".join(errors)

        # ---- Run Custom RAGAS evaluation ----------------------------------
        print(f"\nRunning RAGAS evaluation ({len(evaluable)} samples)…")
        print(f"  LLM judge:   gpt-4o-mini")
        print(f"  Embeddings:  BAAI/bge-m3 (local)")
        print(f"  Concurrency: {max_workers} workers")
        start = time.time()

        tasks = [evaluate_sample(item) for item in evaluable]
        await asyncio.gather(*tasks)

        elapsed = time.time() - start
        print(f"Evaluation completed in {elapsed:.1f}s")

        # ---- Save results -------------------------------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        metric_cols = [
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_relevancy",
        ]

        # Per-sample detail JSON
        per_sample = []
        for item in evaluable:
            entry = item.copy()
            entry["scores"] = {
                col: item.get(col) for col in metric_cols
            }
            per_sample.append(entry)

        # Aggregate means (skip None)
        aggregate = {}
        for col in metric_cols:
            vals = [item.get(col) for item in evaluable if item.get(col) is not None]
            aggregate[col] = sum(vals) / len(vals) if vals else None

        output = {
            "metadata": {
                "timestamp": ts,
                "collection": args.collection or "N/A",
                "total_samples": len(raw_results) if args.evaluate_from else len(dataset),
                "pipeline_succeeded": len(successful),
                "evaluated_samples": len(evaluable),
                "elapsed_seconds": round(elapsed, 1),
                "judge_llm": "gpt-4o-mini",
                "embeddings": "BAAI/bge-m3",
                "max_workers": max_workers,
            },
            "aggregate_scores": aggregate,
            "per_sample_results": per_sample,
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

        # ---- Print summary ------------------------------------------------
        print(f"\n{'=' * 50}")
        print("  RAGAS Evaluation Results")
        print(f"{'=' * 50}")
        for metric_name, score in aggregate.items():
            display = f"{score:.4f}" if score is not None else "N/A"
            print(f"  {metric_name:25s}: {display}")
        print(f"{'=' * 50}")
        print(f"  Detailed results → {json_path}")
        print(f"  Summary CSV      → {csv_path}")

    finally:
        if not args.evaluate_from:
            close_weaviate()


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    asyncio.run(main())
