import os
from ragas.metrics.collections import (
    ContextRecall, ContextPrecision, Faithfulness, AnswerRelevancy,
)
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI

from evals.utils.retry import eval_with_retry

class RagasEvaluators:
    """Wrapper class for Phoenix-compatible RAGAS evaluators."""
    
    def __init__(self, judge_model="gpt-4o-mini", embedding_model="text-embedding-3-small"):
        openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        eval_llm = llm_factory(judge_model, client=openai_client, max_tokens=8192)
        eval_emb = embedding_factory("openai", model=embedding_model, client=openai_client)

        self.precision_scorer = ContextPrecision(llm=eval_llm)
        self.recall_scorer = ContextRecall(llm=eval_llm)
        self.faith_scorer = Faithfulness(llm=eval_llm)
        self.relevancy_scorer = AnswerRelevancy(llm=eval_llm, embeddings=eval_emb)
        
        # Collector for evaluation scores (keyed by question text)
        self.eval_scores = {}

    async def context_precision(self, input, output, expected):
        contexts = output.get("retrieved_contexts", []) if output else []
        if not contexts:
            raise ValueError("Task failed or returned no contexts")
        score = await eval_with_retry(
            self.precision_scorer.ascore,
            user_input=input["user_input"],
            reference=expected["reference_answer"],
            retrieved_contexts=contexts,
        )
        self.eval_scores.setdefault(input["user_input"], {})["context_precision"] = score
        return score

    async def context_recall(self, input, output, expected):
        contexts = output.get("retrieved_contexts", []) if output else []
        if not contexts:
            raise ValueError("Task failed or returned no contexts")
        score = await eval_with_retry(
            self.recall_scorer.ascore,
            user_input=input["user_input"],
            reference=expected["reference_answer"],
            retrieved_contexts=contexts,
        )
        self.eval_scores.setdefault(input["user_input"], {})["context_recall"] = score
        return score

    async def faithfulness(self, input, output, expected):
        contexts = output.get("retrieved_contexts", []) if output else []
        response = output.get("response", "") if output else ""
        if not contexts or not response:
            raise ValueError("Task failed or returned empty response/contexts")
        score = await eval_with_retry(
            self.faith_scorer.ascore,
            user_input=input["user_input"],
            response=response,
            retrieved_contexts=contexts,
        )
        self.eval_scores.setdefault(input["user_input"], {})["faithfulness"] = score
        return score

    async def answer_relevancy(self, input, output, expected):
        response = output.get("response", "") if output else ""
        if not response:
            raise ValueError("Task failed or returned empty response")
        score = await eval_with_retry(
            self.relevancy_scorer.ascore,
            user_input=input["user_input"],
            response=response,
        )
        self.eval_scores.setdefault(input["user_input"], {})["answer_relevancy"] = score
        return score
