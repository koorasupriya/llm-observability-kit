"""Send a stream of chat requests to the demo API so the dashboards have data.

    python -m llm_obs.loadgen --url http://localhost:8000 --requests 200
"""
from __future__ import annotations

import argparse
import random

import httpx

QUESTIONS = [
    "Summarise our leave policy",
    "How do I reset my VPN password?",
    "Draft an email to the finance team about the Q3 budget",
    "Explain the difference between RAG and fine-tuning",
    "What are the steps to raise a priority 1 incident?",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--requests", type=int, default=100)
    args = p.parse_args()
    ok = err = 0
    with httpx.Client(base_url=args.url, timeout=30) as client:
        for _ in range(args.requests):
            model = random.choices(["demo-small", "demo-large"], weights=[3, 1])[0]
            r = client.post("/chat", json={"question": random.choice(QUESTIONS), "model": model})
            ok, err = (ok + 1, err) if r.status_code == 200 else (ok, err + 1)
    print(f"done: {ok} ok, {err} errors")


if __name__ == "__main__":
    main()
