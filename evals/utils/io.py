import json
import csv
from collections import defaultdict
from pathlib import Path

def load_gold_dataset(path: Path) -> list[dict]:
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset

def format_contexts(retrieved_documents: list) -> list[str]:
    """Extract text representations from Weaviate document objects."""
    contexts = []
    for doc in retrieved_documents:
        props = doc.properties if hasattr(doc, "properties") else doc
        doc_type = props.get("type", "")

        if doc_type == "Image":
            text = props.get("text", "")
            caption = props.get("caption", "")
            if text and text != caption:
                ctx = f"[IMAGE] {caption}\n{text}" if caption else text
            else:
                ctx = f"[IMAGE] {caption}" if caption else text
        else:
            ctx = props.get("text", "")

        if ctx.strip():
            contexts.append(ctx)
    return contexts

def save_results(
    task_results: list, 
    eval_scores: dict, 
    metadata: dict, 
    results_dir: Path
):
    """Aggregates scores and saves the evaluation results to JSON and CSV files."""
    ts = metadata["timestamp"]
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
        "metadata": metadata,
        "aggregate_scores": aggregate,
        "per_type_scores": per_type_scores,
        "per_sample_results": task_results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / f"eval_results_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    csv_path = results_dir / f"eval_summary_{ts}.csv"
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

    return json_path, csv_path, aggregate, per_type_scores
