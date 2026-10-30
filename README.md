# Jarvis — app pilote

Assistant local : LLM auto-hébergé (llama.cpp/Vulkan) + outils exposés via MCP.
La voix (STT local, TTS) arrive en toute dernière phase ; ce dépôt est d'abord
le harnais agentique.

## Phases

1. **Boucle agent CLI** (ce squelette) : chat REPL + tool calling sur MCP, LLM via
   `llama-server` (API OpenAI-compatible).
2. **Outils** : MCP servers dédiés (diag réseau, CVE, exec shell restreint...).
3. **Voix** : micro permanent (VAD → STT local → agent), TTS via l'API Voxtral.
   Barge-in/full-duplex : évaluer Pipecat à ce moment-là, pas avant.

## Servir le LLM

llama.cpp build avec backend Vulkan (`-DGGML_VULKAN=ON`). Sur le Framework 13
(Ryzen AI 9 HX 370, Radeon 890M), la VRAM est de la RAM partagée : régler la
carve-out UMA dans le BIOS (32G recommandé pour un 24B Q4).

Exemple MoE (recommandé sur bande passante mémoire partagée : 3B actifs/token) :
```sh
llama-server -m Qwen3.6-35B-A3B.Q4_K_M.gguf -ngl 99 --jinja -fa --port 9931 --ctx-size 16384
```

Exemple dense (comparaison bench) :
```sh
llama-server -m Mistral-Small-3.3-24B-Instruct.Q4_K_M.gguf \
  -ngl 99 --jinja -fa --port 9931 --ctx-size 16384
```

Qwen3.6-35B-A3B Q4_K_M ≈ 20,4 Go : carve-out UMA 32G dans le BIOS, tout le
modèle tient côté GPU.

Note : depuis llama.cpp v0.7.0 le port par défaut de `llama-server` est 9931.

## Installation et lancement

```sh
uv sync
cp config.example.toml config.toml   # ajuster base_url / modèle / MCP servers
uv run jarvis
```

Commandes REPL : `/tools` (liste), `/reset` (historique), `/quit`.

## Benchmark

```sh
uv run python bench/bench_llm.py --prompt "Résous step by step 17*23+5" --max-tokens 512
```
Mesure TTFT et tokens/s en streaming sur le serveur réel : c'est la donnée qui
décide du modèle final (dense 24B vs MoE) sur ta cible.

## Tests (sans LLM réel)

```sh
uv run python tests/test_agent.py
```
Boucle agent end-to-end contre un mock OpenAI-compatible + un MCP server d'echo.

## Structure

```
src/jarvis/
  config.py   # chargement TOML
  llm.py      # client OpenAI-compatible (llama-server)
  mcp.py      # agrégation d'MCP servers stdio -> schémas outils
  agent.py    # boucle tool-calling
  cli.py      # REPL
mcp_servers/  # exemples : nettools (ping/dig), echo (test)
bench/       # mesure tok/s / TTFT
tests/       # mock LLM + test end-to-end
```

Le modèle est interchangeable : le harnais ne dépend que de l'API. Le futur
fine-tune cybersécurité (CyberLLM) se branche ici sans rien réécrire tant que
l'interface outils reste du MCP standard.
