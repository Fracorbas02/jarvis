# Jarvis — app pilote

Assistant local : LLM auto-hébergé (llama.cpp/Vulkan) + outils exposés via MCP
+ voix locale (STT whisper.cpp, TTS llama.cpp avec Voxtral 4B TTS).

## Phases

1. **Boucle agent CLI** : chat REPL + tool calling sur MCP, LLM via
   `llama-server` (API OpenAI-compatible).
2. **Outils** : MCP servers dédiés (diag réseau, CVE, exec shell restreint...).
3. **Voix** : STT local (whisper.cpp) + TTS local (llama-server + modèle de
   synthèse, ex. Voxtral 4B TTS) — voir « Voix locale » ci-dessous.
   Barge-in/full-duplex (micro permanent VAD → agent) : évaluer Pipecat, pas avant.

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

## Voix locale (STT + TTS)

Deux serveurs indépendants du llama-server principal, configurés dans
`config.toml` (`[voice.stt]` / `[voice.tts]`) ; s'ils ne répondent pas, le
CLI et les interfaces les lancent automatiquement avec le GGUF choisi :

```sh
# STT : whisper.cpp (ggml), sert POST /inference
whisper-server --host 127.0.0.1 --port 8178 -m ~/models/stt/Q4_K_M.gguf -ngl 99

# TTS : llama-server avec un modèle de synthèse, sert POST /v1/audio/speech
# (et POST /tts sur les builds récents)
llama-server -m ~/models/tts/voxtral-4b-tts-q4_k.gguf \
  --host 127.0.0.1 --port 9932 -ngl 99 --jinja -fa on
```

La découverte des modèles classe un GGUF en STT/TTS si son **répertoire
parent** s'appelle `stt/` ou `tts/` (ex. `~/models/stt/Q4_K_M.gguf`),
sinon par heuristique de nom (`whisper`/`ggml-` → STT,
`voxtral`/`tts`/`piper`/… → TTS, reste → LLM).

Commandes CLI :

- `/tts <texte>` : synthétise en WAV (écrit dans `~/.cache/jarvis/`) et le joue ;
- `/stt [fichier.wav]` : transcrit le fichier, ou enregistre 5 s de micro
  (arecord, fallback ffmpeg) s'il n'y a pas d'argument ;
- `/voice` : statut des deux serveurs voix.

Dans la GUI web (`jarvis-gui`) et l'app desktop (`jarvis-app`), les panneaux
TTS/STT ont leur propre statut + boutons démarrer/arrêter, la synthèse avec
lecture dans le navigateur (web) / lecture du WAV (desktop), et la
transcription par upload de fichier ou — web uniquement — enregistrement
micro via `getUserMedia` (nécessite localhost/HTTPS).

## Installation et lancement

```sh
uv sync
cp config.example.toml config.toml   # ajuster base_url / modèle / MCP servers
uv run jarvis
```

Commandes REPL : `/help` (liste complète), `/tools` (table outil/description),
`/thinking` (revoir le raisonnement du dernier tour), `/tool N` (résultat
complet du N-ième outil du dernier tour), `/history` (derniers messages),
`/undo` (retirer le dernier échange), `/copy` (dernière réponse dans le
presse-papier), `/save [fichier]` (exporter la conversation en Markdown),
`/sessions` (conversations passées), `/load N`, `/new`, `/compact` (résume
l'historique via le LLM pour libérer du contexte — le modèle peut aussi
appeler seul l'outil interne `jarvis__compact`), `/reset`, `/verbose`,
`/quit`. Tab complète les commandes `/` ; l'historique des saisies persiste
dans `~/.cache/jarvis/history`. **Multiligne** : Maj+Entrée insère un saut de
ligne (protocole clavier kitty, sur kitty/wezterm/ghostty/foot), Entrée envoie ;
fallback universel : `\` en fin de ligne puis Entrée. Ctrl+C annule la saisie,
double Ctrl+C quitte.
Une ligne de métriques (tok, tok/s, durée) suit chaque réponse streamée ; la
bannière affiche le statut du serveur (GET /v1/models) et la taille du
contexte max (GET /slots, dynamique — 32k si relancé avec `--ctx-size 32768`),
et le prompt affiche l'usage réel du contexte (`n_prompt_tokens` de /slots).
Notification desktop (OSC 777/9) en fin de tour long (> 3 s).

**Sessions persistantes** (globales, pas par dossier) : chaque conversation
est sauvegardée en JSON dans `~/.cache/jarvis/sessions/` après chaque tour ;
au démarrage la dernière est reprise automatiquement, `/sessions` liste les
passées, `/load N` en reprend une, `/new` démarre frais.

## Interface graphique

```sh
uv run jarvis-gui            # http://127.0.0.1:8080 (ouvre le navigateur)
uv run jarvis-gui --port 9000 --no-open-browser
```

UI web locale (NiceGUI, thème sombre) :

- **Sélecteur de mode** : Chat LLM (boucle agent complète avec outils MCP),
  TTS, STT — les panneaux voix listent les modèles détectés pour le mode.
- **Modèles installés** : découverts automatiquement (`~/models`, cache
  HuggingFace, `$JARVIS_MODEL_DIRS`) et classés par mode (LLM / TTS / STT)
  par heuristique de nom (`whisper`/`ggml-` → STT, `piper`/`kokoro`/`xtts`… →
  TTS, sinon LLM).
- **Serveur LLM** : statut live de llama-server (`GET /v1/models`), modèle
  servi, et si aucun serveur ne répond : bouton pour en lancer un avec le
  GGUF sélectionné (`llama-server -m … -ngl 99 --jinja -fa on`).
- **Chat** : streaming identique au CLI — raisonnement du modèle en direct
  (expansion grisée), réponse en markdown, appels d'outils en chips.

Tests GUI (sans navigateur, via le plugin pytest de NiceGUI) :

```sh
uv run pytest tests/ -v
```

## Application desktop

```sh
uv run jarvis-app             # fenêtre native customtkinter (dark mode)
uv run jarvis-app --config /chemin/config.toml
uv run jarvis-app --scale 2   # forcer le facteur d'échelle (défaut: auto)
```

La fenêtre est redimensionnable (sash déplaçable entre la sidebar et la zone
principale) et s'adapte à la résolution de l'écran (62 % x 68 %, centrée).
Le facteur d'échelle est détecté automatiquement (DPI de l'écran, hints
`GDK_SCALE`/`QT_SCREEN_SCALE_FACTORS`, gsettings GNOME, `monitors.xml`
GNOME >= 46, override `--scale`) et appliqué avant
la création des widgets : dimensions et polices suivent le scaling GNOME
(200 % → tout est x2).

Même expérience que la GUI web, en application native : sélecteur de mode
(Chat LLM / TTS / STT), modèles détectés par mode, panneau serveur
llama-server (statut, start/stop), liste des outils MCP, et chat avec
streaming (réflexion en grisé italique, outils en violet, réponse en clair).
La logique async (LLM + MCP) tourne dans une boucle asyncio sur un thread
dédié, les événements remontent au thread Tk via une queue.

Test de l'app desktop sous Xvfb (mock LLM, fenêtre pilotée) :

```sh
xvfb-run -a uv run python tests/test_desktop_app.py
```

## Outils MCP

| Server   | Outils | Source |
|----------|--------|--------|
| nettools | ping, dig, traceroute, whois, listening_ports, connections, ip_addr, tls_info, http_get | local (binaires système) / urllib |
| sysinfo  | disk_usage, memory, load, gpu_status, temperatures, processes, logs (journalctl), ports_listening, network_interfaces | local (/proc, binaires système) |
| files    | grep_files (ripgrep), read_file (paginé), file_info (empreintes), list_dir | local, lecture confinée à `JARVIS_FILES_ROOTS` (défaut: cwd + `$HOME`) |
| netutils | ip_calc, ip_convert, encode (b64/hex/url/rot13), hash_text, time_now, time_convert | stdlib uniquement, aucun appel externe |
| git      | git_status, git_log, git_diff, git_branches, git_remotes, git_show | dépôt courant ou `JARVIS_GIT_REPO`, lecture seule |
| memory   | remember, recall, forget | `~/.cache/jarvis/memory.md` (ou `JARVIS_MEMORY_FILE`) |
| cve      | cve_info, search_cve (filtre `days` optionnel), latest_cve | API NVD 2.0 |
| echo     | ping | test du harnais uniquement |

`cve` : sans clé API NVD, limité à 5 requêtes/30 s (throttle intégré côté
server). Clé gratuite : https://nvd.nist.gov/developers/request-key puis
`export NVD_API_KEY=...` (50 req/30 s).

`files` est **lecture seule et confiné** : un chemin hors de `JARVIS_FILES_ROOTS`
renvoie une erreur. `git` n'expose aucune commande d'écriture (pas de
commit/add/push). Les serveurs reçoivent l'environnement complet du processus
jarvis (variables `NVD_API_KEY`, `JARVIS_FILES_ROOTS`, `JARVIS_MEMORY_FILE`,
`JARVIS_GIT_REPO`...).

## Benchmark

```sh
uv run python bench/bench_llm.py --max-tokens 2048 -n 3
```
Mesure TTFT et tokens/s en streaming sur le serveur réel. Le bench distingue
le raisonnement du contenu (modèles thinking type Qwen3.6) : pour la boucle
vocale, le TTFT qui compte est celui du premier token de contenu.

## Latence, thinking et streaming

Qwen3.6 réfléchit avant de répondre : à ~24 tok/s, une réponse longue avec
raisonnement peut prendre plus d'une minute (mesuré : ~76 s pour un résumé
CVE détaillé). Le choix assumé est de garder le thinking et de le rendre
visible : la TUI (rich) streame `delta.reasoning_content` dans un panneau
grisé pendant que le modèle réfléchit, le replie en une ligne résumé
(durée, tokens) à la fin, puis rend la réponse en Markdown dans un panneau
en direct. Les appels d'outils s'affichent avec un spinner puis un statut
`✓/✗ + durée` et un aperçu du résultat. L'équivalent du « je réfléchis… »
de Le Chat, sauf que c'est la vraie pensée, en direct.

Pour couper le thinking au profit de la latence pure (au cas par cas) :
```toml
[llm]
chat_template_kwargs = { enable_thinking = false }
```

En boucle vocale, le raisonnement ne sera jamais parlé : il servira à décider
si la réponse vaut l'attente, et à annoncer des étapes (« je consulte NVD… »)
pendant que le TTS attend le premier token de contenu.

## Tests (sans LLM réel)

```sh
uv run python tests/test_agent.py
```
Boucle agent end-to-end contre un mock OpenAI-compatible + un MCP server d'echo.

## Structure

```
src/jarvis/
  config.py   # chargement TOML (dont [voice.stt] / [voice.tts])
  llm.py      # client OpenAI-compatible (llama-server), streaming SSE + tools
  voice.py    # voix locale : clients STT (whisper-server /inference) et TTS
              #  (llama-server /v1/audio/speech + fallback /tts), gestion des
              #  processus serveurs voix, record/play WAV (arecord/ffmpeg/ffplay)
  mcp.py      # agrégation d'MCP servers stdio -> schémas outils
  agent.py    # boucle tool-calling (streaming activable, hooks d'affichage)
  tui.py      # couche d'affichage rich : panneau thinking repliable, réponse
              #  en Markdown (live), statut des outils, /thinking et /tool N
  cli.py      # REPL (bannière, prompt, /help /tools /verbose /tts /stt /voice ...)
  gui/        # interface web locale (NiceGUI) : app, sidebar, chat_view,
              #  voice_views (TTS/STT fonctionnels), models (découverte),
              #  server (statut llama-server),
              #  backend (pont agent + streaming) -> `uv run jarvis-gui`
  desktop/    # application desktop (customtkinter) : app, sidebar, chat_panel,
              #  voice_panel (TTS/STT fonctionnels), controller (boucle asyncio
              #  + queue vers Tk, appels voix inclus) -> `uv run jarvis-app`
mcp_servers/  # nettools (ping/dig/traceroute/whois/ss/ip/TLS/HTTP),
              #  sysinfo (disques/mémoire/GPU/processus/journalctl),
              #  files (grep/read/empreintes, lecture confinée),
              #  netutils (IP/encode/hash/heure, stdlib),
              #  git (status/log/diff, lecture seule), memory (mémoire persistante),
              #  cve (NVD 2.0), echo (test)
bench/       # mesure tok/s / TTFT
tests/       # mock LLM (stream SSE + tool_calls fragmentés), test end-to-end,
              #  test_tui.py : validation visuelle du rendu sans LLM
```

Le modèle est interchangeable : le harnais ne dépend que de l'API. Le futur
fine-tune cybersécurité (CyberLLM) se branche ici sans rien réécrire tant que
l'interface outils reste du MCP standard.
