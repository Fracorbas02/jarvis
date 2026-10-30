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
llama-server -m ~/models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 --jinja -fa on --port 9931 --ctx-size 16384
```

Exemple dense (comparaison bench) :
```sh
llama-server -m Mistral-Small-3.3-24B-Instruct.Q4_K_M.gguf \
  -ngl 99 --jinja -fa on --port 9931 --ctx-size 16384
```

Qwen3.6-35B-A3B Q4_K_M ≈ 21 Go : carve-out UMA 32G dans le BIOS, tout le
modèle tient côté GPU.

Note : depuis llama.cpp v0.7.0 le port par défaut de `llama-server` est 9931,
et `-fa` requiert une valeur explicite (`on|off|auto`).

## Installation et lancement

```sh
uv sync
cp config.example.toml config.toml   # ajuster base_url / modèle / MCP servers
uv run jarvis
```

Commandes REPL : `/tools` (liste), `/reset` (historique), `/quit`.

## Outils MCP

| Server   | Outils | Source |
|----------|--------|--------|
| nettools | ping, dig, traceroute | local (binaires système) |
| cve      | cve_info, search_cve (filtre `days` optionnel), latest_cve | API NVD 2.0 |
| echo     | ping | test du harnais uniquement |

`cve` : sans clé API NVD, limité à 5 requêtes/30 s (throttle intégré côté
server). Clé gratuite : https://nvd.nist.gov/developers/request-key puis
`export NVD_API_KEY=...` (50 req/30 s).

## Benchmark

```sh
uv run python bench/bench_llm.py --max-tokens 2048 -n 3
```
Mesure TTFT et tokens/s en streaming sur le serveur réel. Le bench distingue
le raisonnement du contenu (modèles thinking type Qwen3.6) : pour la boucle
vocale, le TTFT qui compte est celui du premier token de contenu — prévoir un
budget max-tokens suffisant pour couvrir la phase de raisonnement.

## Latence et modèles thinking

Qwen3.6 réfléchit avant de répondre : à ~24 tok/s, une réponse longue avec
raisonnement peut prendre plus d'une minute (mesuré : ~76 s pour un résumé
CVE détaillé). Pour l'usage interactif, le thinking se coupe par requête dans
la config — sans redémarrer llama-server :

```toml
[llm]
chat_template_kwargs = { enable_thinking = false }
```

llama-server transmet ces kwargs au chat template du modèle. À retirer au cas
par cas pour les tâches complexes ; en boucle vocale, le thinking sera à couper
systématiquement.

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
mcp_servers/  # nettools (ping/dig/traceroute), cve (NVD 2.0 : détail CVE,
              #  recherche par mots-clés + fenêtre de jours, veille), echo (test)
bench/       # mesure tok/s / TTFT
tests/       # mock LLM + test end-to-end
```

Le modèle est interchangeable : le harnais ne dépend que de l'API. Le futur
fine-tune cybersécurité (CyberLLM) se branche ici sans rien réécrire tant que
l'interface outils reste du MCP standard.
