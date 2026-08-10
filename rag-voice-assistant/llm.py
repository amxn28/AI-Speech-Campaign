import re
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL, MAX_TOKENS

client = OpenAI(api_key=OPENAI_API_KEY)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting so text reads cleanly as speech/plain text."""
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build prompt with retrieved context."""
    if not chunks:
        context = "No relevant information found in the campaign documents."
    else:
        context = "\n\n".join([
            f"[Source: {c['metadata'].get('source', 'unknown')} | District: {c['metadata'].get('district', 'general')}]\n{c['content']}"
            for c in chunks
        ])

    prompt = f"""You are a helpful political campaign assistant.
Answer the user's question using ONLY the context provided below.
If the context doesn't contain enough information, say so honestly.
Reply fully in English, even if the user's question or the context below is in Hindi — translate any facts you use.
Write plain conversational sentences only — no markdown, no bold, no headers, no bullet points, no asterisks or hashes — since this answer may be read aloud.

Context:
{context}

User Question: {query}

Answer:"""

    return prompt


def generate_response(query: str, chunks: list[dict]) -> dict:
    """Call OpenAI and return response with citations."""
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a political campaign voice assistant. Be concise and factual. Never use markdown formatting - plain text only. Always reply in English, regardless of what language the question or source context is in."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_TOKENS
    )

    answer = strip_markdown(response.choices[0].message.content)

    sources = list(set([
        c["metadata"].get("source", "unknown") for c in chunks
    ]))

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks)
    }
