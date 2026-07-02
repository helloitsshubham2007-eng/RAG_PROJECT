"""
test_model_direct.py — sanity-check the Claude API call in isolation,
with no retrieval/chunking/vector-DB in the loop.

Use this to answer: "is the model itself working, or is something in
my RAG pipeline (chunking, retrieval, prompt-building) the problem?"

Run:
    python test_model_direct.py
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY not set. Add it to your .env file.")

client = Anthropic(api_key=api_key)

MODEL = "claude-sonnet-5"

test_cases = [
    "What is 2 + 2?",                                   # sanity check, no context needed
    "Say the word 'pong' and nothing else.",             # tests exact instruction-following
    "Summarize in one sentence: RAG combines retrieval and generation.",
]

for i, prompt in enumerate(test_cases, 1):
    print(f"\n--- Test {i}: {prompt}")
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    print("Response:", response.content[0].text)
    print("Stop reason:", response.stop_reason)
    print("Tokens used:", response.usage.input_tokens, "in /", response.usage.output_tokens, "out")
