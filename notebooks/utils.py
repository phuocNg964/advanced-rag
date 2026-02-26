import pandas as pd
import ast
import re
from pathlib import Path
import sys
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root)) 

from src.config import get_settings
from src.retriever import retrieve

setting = get_settings()

def parse_contexts(ctx_str):
    """Parse string representation of list to actual list"""
    if pd.isna(ctx_str):
        return []
    try:
        return ast.literal_eval(ctx_str)
    except:
        return [ctx_str]

def clean_hop_markers(contexts: list) -> list:
    """
    Remove <1-hop>, <2-hop>, <3-hop> markers from contexts
    Example: "<1-hop>\n\nActual content..." -> "Actual content..."
    """
    cleaned = []
    for ctx in contexts:
        # Remove patterns like "<1-hop>\n\n", "<2-hop>\n\n", "<3-hop>\n\n"
        cleaned_ctx = re.sub(r'^<\d+-hop>\n\n', '', ctx.strip())
        cleaned.append(cleaned_ctx)
    return cleaned

def exact_precision(retrieves, references) -> float:
    # actual docs in retrieved docs / all retrieved docs
    retrieved_docs = set(r.strip() for r in retrieves)
    reference_docs = set(r.strip() for r in references)

    numerator = len(retrieved_docs & reference_docs)

    return numerator / len(retrieved_docs)

def exact_recall(retrieves, references) -> float:
    # actual docs in retrieved docs / all reference docs
    retrieved_docs = set(r.strip() for r in retrieves)
    reference_docs = set(r.strip() for r in references)

    numerator = len(retrieved_docs & reference_docs)

    return numerator / len(reference_docs)

def systematic_retrieval_eval(
    queries, 
    reference_contexts, 
    top_k, 
    top_reranker, 
    rewritten_queries=None,
    collection_name='TestCollection',
) -> pd.DataFrame:

    recall_scores = []
    precision_scores = []
    retrieved_contexts = []

    for idx, query in enumerate(queries):
        retrieved_docs = []
        seen_ids = set()

        sub_queries = rewritten_queries[idx] if rewritten_queries else [query]

        for sub_query in sub_queries:
            docs = retrieve(
                sub_query, 
                collection_name=collection_name, 
                top_k=top_k, 
                top_k_reranker=top_reranker,
            )

            for doc in docs:
                # Use UUID or chunk_id for deduplication
                doc_id = str(doc.uuid)  # Weaviate objects have .uuid
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    retrieved_docs.append(doc.properties['text'])
        
        retrieved_contexts.append(retrieved_docs)

        recall_scores.append(exact_recall(retrieved_contexts[idx], reference_contexts[idx]))

        precision_scores.append(exact_precision(retrieved_contexts[idx], reference_contexts[idx]))
    
    data = {'queries': queries, 'reference_contexts': reference_contexts, 'retrieved_contexts': retrieved_contexts, 'recall_scores': recall_scores, 'precision_scores': precision_scores}

    # Create the DataFrame
    df = pd.DataFrame(data)
    if rewritten_queries:
        df['rewritten_queries'] = rewritten_queries
    return df

def evaluate_rewrites(
    original_queries: list[str],
    rewritten_queries: list[list[str]],
    llm  # your evaluator LLM
) -> pd.DataFrame:
    """Score query rewrites using LLM-as-Judge."""
    import json
    
    eval_prompt = """You are evaluating a query rewriter for a document retrieval system.

Given the ORIGINAL query and the REWRITTEN query(ies), score each criterion from 1-5.

## Criteria:

1. **Entity Preservation** (1-5): Are ALL specific names, numbers, metrics, datasets 
   from the original query preserved in the rewrite?
   5 = all preserved, 1 = key entities dropped

2. **Semantic Equivalence** (1-5): Does the rewrite ask for the same information?
   5 = identical meaning, 1 = meaning changed significantly

3. **Typo Handling** (1-5): Were obvious typos/misspellings fixed?
   5 = all fixed, 3 = no typos to fix (N/A), 1 = typos remain

4. **Fluff Removal** (1-5): Was unnecessary persona/politeness text removed 
   while keeping the core question?
   5 = clean removal, 3 = no fluff to remove (N/A), 1 = fluff remains or core content removed

5. **Decomposition** (1-5): Was the split/no-split decision appropriate?
   Test: "Can each sub-query be fully answered without knowing the other?"
   5 = perfect decision, 3 = acceptable, 1 = harmful split or missed necessary split

## Input:
Original: {original}
Rewritten: {rewritten}

## Output format (JSON only):
{{"entity_preservation": <score>, "semantic_equivalence": <score>, "typo_handling": <score>, "fluff_removal": <score>, "decomposition": <score>, "reasoning": "<brief explanation>"}}
"""
    
    results = []
    for orig, rewrite in zip(original_queries, rewritten_queries):
        response = llm.invoke(eval_prompt.format(
            original=orig, 
            rewritten=rewrite
        ))
        # Parse JSON from response
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            scores['original'] = orig
            scores['rewritten'] = str(rewrite)
            results.append(scores)
    
    df = pd.DataFrame(results)
    return df

async def LLM_based_eval_retrieval(
    queries, 
    reference_contexts, 
    top_k, 
    llm,
    top_reranker, 
    rewritten_queries=None,
) -> pd.DataFrame:

    from ragas import SingleTurnSample
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
    
    recall_scores = []
    precision_scores = []
    retrieved_contexts = []

    for idx, query in enumerate(queries):
        retrieved_docs = []
        seen_ids = set()

        sub_queries = rewritten_queries[idx] if rewritten_queries else [query]

        for sub_query in sub_queries:
            docs = retrieve(
                sub_query, 
                collection_name='TestCollection', 
                top_k=top_k, 
                top_k_reranker=top_reranker
            )

            for doc in docs:
                # Use UUID or chunk_id for deduplication
                doc_id = str(doc.uuid)  # Weaviate objects have .uuid
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    retrieved_docs.append(doc.properties['text'])
        
        retrieved_contexts.append(retrieved_docs)

        recall_scorer = LLMContextRecall(llm=llm)
        precision_scorer = LLMContextPrecisionWithReference(llm=llm)
        

        sample = SingleTurnSample(
            user_input=query,
            reference=reference_contexts[idx],
            retrieved_contexts=retrieved_docs
        )

        recall_score = await recall_scorer.single_turn_ascore(sample)
        precision_score = await precision_scorer.single_turn_ascore(sample)

        recall_scores.append(recall_score)

        precision_scores.append(precision_score)
    
    data = {'queries': queries, 'reference_contexts': reference_contexts, 'retrieved_contexts': retrieved_contexts, 'recall_scores': recall_scores, 'precision_scores': precision_scores}

    # Create the DataFrame
    df = pd.DataFrame(data)
    if rewritten_queries:
        df['rewritten_queries'] = rewritten_queries
    return df