PROMPT_TEMPLATE = """Answer the question using only the information in the context below. \
You don't need an exact matching sentence — synthesize a clear answer from what the context \
describes. If the context truly contains nothing relevant to the question, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

def build_prompt(query, results):
    context = "\n\n".join(c.text for c, _ in results)
    return PROMPT_TEMPLATE.format(context=context, question=query)

import os

def generate_with_anthropic(query,results, model="claude-sonnet-5"):
    """
    results: the output of retriever.search(query, top_k=...) - a list of (chunk, score) tuples.
    """
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "Set the ANTHROPIC_API_KEY environment variable first."

    prompt = build_prompt(query, results)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model = model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text

def extractive_answer(query, results):
    """
    No LLM call - just shows what context would be sent to a generator.
    Good for verifying retrieval quality without spending any API credits.
    """
    lines = [f"Question: {query}", "", "Retrieved context:"]
    for chunk, score in results:
        lines.append(f"  ({score:.3f}) [{chunk.chunk_id}]: {chunk.text}")
    return "\n".join(lines)

import requests

def generate_with_ollama(query, results, model="llama3.2"):
    """
    Calls a local Ollama server — free, no API key, runs entirely on your machine.
    Requires `ollama pull llama3.2` and the Ollama app running in the background.
    """
    prompt = build_prompt(query, results)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"]