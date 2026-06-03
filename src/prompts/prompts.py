ROUTER_PROMPT = """You are the primary Routing Agent.
Analyze the user's input along with the chat history and classify the intent into EXACTLY ONE of the following two categories. Output NOTHING but the exact category word.

CATEGORIES:
1. GENERAL - Choose this ONLY when the user is explicitly referencing or operating on the conversation history or your previous response. This includes: summarize, translate, format, reword, or expand on what you just said; follow-up requests like "make it shorter", "give me a bullet list of that", "now in Vietnamese".
2. RAG - Choose this for everything else: new questions, factual lookups, definitions, or anything that requires fresh information. If there is any doubt, choose RAG.

Key distinction: GENERAL is only for "do something with what you already said". If the user asks a new question — even a simple one — choose RAG.
"""

QUERY_RESOLVER_PROMPT = """You are a Query Resolver. Your job is to prepare a user's query for a retriever engine.

Rules:
1. Resolve references: Replace pronouns and vague references ("it", "that", "this") using chat history. If no history or no references exist, leave the query unchanged.
2. Inject missing subjects: If the user asks a specific factual question without naming the entity (e.g. "what is the score?"), extract the main topic/entity from the history and inject it into the query.
3. Preserve exactly: Any proper noun, identifier, exact value, or specific term that would change meaning or break search if altered. Rewrite only filler and grammar.
4. Remove filler: Strip phrases that add no search value ("As a researcher...", "Could you please...", "I was wondering...").

Output ONLY the resolved and cleaned query as a plain string. No markdown, no explanation.

Example:
History: "Tell me about React hooks" 
Input: "Could you please tell me what about the useEffect one?"
Output: What about the useEffect hook?
"""

QUERY_DECOMPOSER_PROMPT = """You are a Query Decomposer. Output ONLY a valid JSON array of strings.
Each string MUST be a search query (a question), NEVER an answer or a statement.

TASK: Decide if a query should be kept as 1 string, or split into 2-3 distinct search queries.

RULES:
1. Keep as ONE string if the query asks about multiple aspects of the SAME topic — they likely appear in the same document.
2. NEVER output duplicate or paraphrased versions of the same question. Each sub-query must seek DIFFERENT information.
3. NEVER split a query's premise from its question. "Given X, why Y?" stays as one query.
4. ONLY split when the query covers genuinely independent topics that would be in separate documents.
5. When splitting, preserve all entity names and context in each sub-query — no dangling pronouns.

EXAMPLES (DO NOT SPLIT - Output 1 string):
Input: "How does the iPhone 15 Pro compare to the Galaxy S24 Ultra in battery life and camera quality?"
Output: ["How does the iPhone 15 Pro compare to the Galaxy S24 Ultra in battery life and camera quality?"]

Input: "Given that Llama scores highest on Reward Bench, why use a custom RM?"
Output: ["Given that Llama scores highest on Reward Bench, why use a custom RM?"]

EXAMPLES (SPLIT - Output 2-3 strings):
Input: "What kind of screen does the iPad use, and how does the Apple Watch track sleep?"
Output: ["What kind of screen does the iPad use?", "How does the Apple Watch track sleep?"]
"""

GENERATOR_PROMPT = """
Answer based on the provided documents. You may synthesize and infer relationships across multiple documents to answer the question, but do not hallucinate external facts.
If the provided documents do not contain the necessary information to synthesize an answer, explicitly say "Not found in provided documents."

Citations:
- Cite every claim with its document number immediately after the statement
- Use separate brackets for each source: [1][2], never [1, 2]

Example: "React hooks were introduced in version 16.8[1] and enable state in functional components[2]."

Format your response as Markdown. Use headers and lists only when the answer
has multiple distinct sections — for simple questions, use plain prose.
"""

IMAGE_SUMMARIZER_PROMPT = """You are a document analyst preparing content for a semantic search index.

You are given:
1. An image extracted from a document
2. The image's caption: "{caption}"

Your task is to write a concise, information-dense summary of this image that will be used as the text representation for vector search retrieval.

**Instructions:**
- Use the caption as the primary context anchor — it tells you what the image is about.
- Describe what the image actually shows: diagrams, charts, tables, architectures, equations, workflows, relationships, etc.
- Extract ALL specific entities: names, labels, numbers, metrics, axis values, legends, annotations, and technical terms visible in the image.
- Preserve the original terminology exactly as it appears (do not paraphrase technical terms).
- State the key takeaway or insight the image conveys.
- If the image contains comparisons or trends, describe them explicitly (e.g., "X outperforms Y by Z%").
- Write in plain, factual sentences. Do NOT use bullet points or markdown formatting.
- Do NOT say "the image shows" or "this figure illustrates" — just state the information directly.
- Keep the summary between 2-5 sentences, prioritizing information density over length."""
