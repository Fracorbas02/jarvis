"""Bench llama-server : TTFT et tokens/s en streaming.

Gère les modèles thinking (type Qwen3.6) : llama.cpp sépare le raisonnement
dans delta.reasoning_content. On chronomètre et compte les deux flux
séparément — le TTFT utile pour la boucle vocale est celui du contenu.

Usage:
    uv run python bench/bench_llm.py --prompt "..." --max-tokens 512 [-n 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import tomllib
from pathlib import Path

import httpx


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
    ttft_any = None       # premier token quel qu'il soit (raisonnement inclus)
    ttft_content = None   # premier token de contenu effectif
    n_content = 0
    n_reasoning = 0
    error = None

    async with client.stream("POST", f"{base_url}/chat/completions", json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode(errors="replace")[:500]
            error = f"HTTP {resp.status_code}: {body}"
        else:
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
                if chunk.get("error"):
                    error = f"erreur serveur: {chunk['error']}"
                    break
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    n_content += 1
                    if ttft_content is None:
                        ttft_content = time.perf_counter() - t0
                    if ttft_any is None:
                        ttft_any = ttft_content
                elif delta.get("reasoning_content"):
                    n_reasoning += 1
                    if ttft_any is None:
                        ttft_any = time.perf_counter() - t0

    t_total = time.perf_counter() - t0
    return {
        "ttft_any_s": ttft_any,
        "ttft_content_s": ttft_content,
        "total_s": t_total,
        "content_tokens": n_content,
        "reasoning_tokens": n_reasoning,
        "total_tokens": n_content + n_reasoning,
        "tok_s": (n_content + n_reasoning) / t_total if t_total else 0.0,
        "error": error,
    }


def _fmt(x: float | None, unit: str) -> str:
    return f"{x:.2f}{unit}" if x is not None else "n/a"


async def main() -> int:
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
            if r["error"]:
                print(f"run {i + 1}: ERREUR {r['error']}")
                continue
            print(f"run {i + 1}: TTFT 1er token {_fmt(r['ttft_any_s'], 's')} | "
                  f"TTFT contenu {_fmt(r['ttft_content_s'], 's')} | "
                  f"{r['tok_s']:.1f} tok/s "
                  f"({r['content_tokens']} contenu + {r['reasoning_tokens']} raisonnement) | "
                  f"{r['total_s']:.2f}s")

    valid = [r for r in results if r["total_tokens"] > 0 and not r["error"]]
    if not valid:
        print("\naucun token reçu : vérifie que llama-server tourne sur ce port "
              "et que le nom de modèle dans la config correspond au serveur.")
        return 1
    if len(valid) < len(results):
        print(f"\nattention : {len(results) - len(valid)} run(s) en erreur ignoré(s)")

    mean = sum(r["tok_s"] for r in valid) / len(valid)
    contents = [r["ttft_content_s"] for r in valid if r["ttft_content_s"] is not None]
    mean_ttft = sum(contents) / len(contents) if contents else None
    print(f"\nmoyenne {mean:.1f} tok/s | TTFT contenu moyen {_fmt(mean_ttft, 's')}")
    if valid[0]["reasoning_tokens"] > 0:
        print("modèle thinking détecté : en production, le raisonnement s'exclut de la "
              "réponse parlée ; le TTFT qui compte pour la voix est celui du contenu.")
    print("verdict: < 5 tok/s = dense non viable en boucle vocale ; "
          "> 15 tok/s = confortable CLI, acceptable voix (20 tok/s ≈ débit de lecture)")
    return 0


if __name__ == "__main__":
    sys_exit = asyncio.run(main())
    raise SystemExit(sys_exit)
