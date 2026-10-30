"""Bench llama-server : TTFT et tokens/s en streaming.

Usage:
    uv run python bench/bench_llm.py --prompt "..." --max-tokens 512 [-n 3]

À lancer sur le Framework 13 contre le vrai serveur (dense 24B vs MoE) :
c'est la mesure qui tranche le choix du modèle sur bande passante mémoire
partagée (~70-80 Go/s réels mesurés).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import tomllib
from pathlib import Path


def _load_llm_cfg() -> dict:
    p = Path("config.toml")
    if not p.is_file():
        p = Path("config.example.toml")
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    return raw.get("llm", {})


async def one_pass(client: httpx.AsyncClient, base_url: str, model: str,
                   prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    body = []
    async with client.stream("POST", f"{base_url}/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                tokens += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if len(body) < 10:
                    body.append(delta["content"])
    t_total = time.perf_counter() - t0
    return {
        "ttft_s": ttft,
        "total_s": t_total,
        "completion_tokens": tokens,
        "tok_s": tokens / t_total if t_total else 0.0,
        "preview": "".join(body)[:80],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Explique en détail le fonctionnement de BGP.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("-n", "--runs", type=int, default=3)
    args = parser.parse_args()

    cfg = _load_llm_cfg()
    base_url = cfg.get("base_url", "http://127.0.0.1:9931/v1").rstrip("/")
    model = cfg.get("model", "mistral")
    print(f"cible: {base_url}  modèle: {model}  runs: {args.runs}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
        results = []
        for i in range(args.runs):
            r = await one_pass(client, base_url, model, args.prompt, args.max_tokens)
            results.append(r)
            print(f"run {i + 1}: TTFT {r['ttft_s']:.2f}s | {r['tok_s']:.1f} tok/s "
                  f"| {r['completion_tokens']} tokens | {r['total_s']:.2f}s")

    valid = [r for r in results if r["completion_tokens"] > 0]
    if valid:
        best = max(r["tok_s"] for r in valid)
        mean = sum(r["tok_s"] for r in valid) / len(valid)
        print(f"\nmoyenne {mean:.1f} tok/s | meilleur {best:.1f} tok/s")
        print("verdict: < 5 tok/s = dense 24B non viable en boucle vocale, "
              "tester un MoE ; > 20 tok/s = confortable CLI, acceptable voix.")


if __name__ == "__main__":
    asyncio.run(main())
