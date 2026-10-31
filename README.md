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

Dans l'app GTK4 (`jarvis-app`), la conversation se fait au micro (STT :
enregistrement local puis transcription) et les réponses peuvent être lues
à voix haute (TTS, switch dans les réglages) ; le micro utilisé est choisi dans
les réglages.

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

## Interface graphique (GTK4 / libadwaita)

```sh
uv run jarvis-app                 # fenêtre native GTK4 (PyGObject + libadwaita)
uv run jarvis-app --config /chemin/config.toml
```

UI native GTK4, qui ne touche à rien d'autre que la partie graphique
(le cœur LLM + MCP + voix + sessions est réutilisé tel quel) :

- **Accueil** : un seul gros bouton micro « Parler à Jarvis ». Si aucun
  llama-server ne répond, il en démarre un avec le GGUF choisi (sélecteur dans
  le popover statut, `--ctx-size` réglable) puis ouvre la conversation ;
  sinon il ouvre directement la conversation. Le prompt texte est dans le
  **menu (Ctrl+P)**, avec les conversations (Ctrl+O) et la lecture TTS.
- **Conversation** : centrée sur l'échange — barre du bas avec micro (STT
  local : record → transcribe → tour LLM) + champ texte + envoyer ; réponses
  en streaming avec raisonnement repliable et appels d'outils en lignes
  ✓/✗ + durée.
- **Réglages** (vrai onglet) : thème (système/sombre/clair), GPU utilisé par les
  serveurs locaux (persisté dans `~/.config/jarvis/gpu.json`), taille de
  contexte (`--ctx-size`), liste des outils MCP, micro utilisé (arecord -L,
  persisté dans `~/.config/jarvis/ui.json`), TTS, et autorisations système
  (statut de l'autorisation via le portail XDG, test du micro + ouverture
  des réglages Confidentialité GNOME). Au **premier lancement**, une boîte
  de dialogue demande l'autorisation micro (bouton Autoriser = test
  d'enregistrement réel de 0,5 s qui déclenche la demande système GNOME ;
  le dialogue ne se remontre plus ensuite, flag `mic_permission_asked`
  dans `ui.json`).
- Popover de statut serveur (icône réseau) : en ligne / géré / hors ligne,
  modèle servi, usage réel du contexte (GET /slots), start/stop du serveur géré.

Dépendance système : `python3-gobject`, `gtk4`, `libadwaita-1`
(ex. `sudo pacman -S python-gobject gtk4 libadwaita` sur Arch).

Piège uv : PyGObject vit dans les site-packages du **python système**
(`/usr/lib/python3.x/site-packages`). Le venv d'uv doit donc être créé sur
ce python avec `--system-site-packages`, sinon `import gi` échoue même avec
`python-gobject` installé (le Python standalone d'uv ne voit pas `/usr/lib/...`) :

```sh
rm -rf .venv
uv venv --python /usr/bin/python3 --system-site-packages .venv
uv sync --python /usr/bin/python3
# si uv resette le venv sans le flag : UV_SYSTEM_SITE_PACKAGES=true uv sync --python /usr/bin/python3
uv run python -c "import gi; gi.require_version('Gtk','4.0'); from gi.repository import Gtk; print('GTK', Gtk.get_major_version())"
uv run jarvis-app
```

Tests du pont GTK (sans GTK ni LLM réel, mock LLM + MCP echo) :

```sh
python3 tests/test_gtk_backend.py          # flux d'événements, sessions, réglages
python3 tests/test_gtk_app_structure.py    # structure de la fenêtre (stub gi)
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
  server.py   # statut + start/stop du llama-server (GET /v1/models, Popen)
  models.py   # découverte des GGUF locaux (~/models, cache HF, $JARVIS_MODEL_DIRS),
              #  classés par mode (LLM/TTS/STT) par heuristique de nom
  gtk/        # interface GTK4 / libadwaita (seule UI graphique) : app (fenêtre
              #  Adw : accueil micro, conversation, réglages), widgets (bulles,
              #  reasoning repliable, outils), backend (boucle asyncio + queue
              #  + sessions + statut serveur) -> `uv run jarvis-app`
  gtk/        # interface GTK4 / libadwaita (refonte UI) : app (fen\u00eatre Adw :
              #  accueil "Lancer", conversation, r\u00e9glages), widgets (bulles,
              #  reasoning repliable, outils), backend (boucle asyncio + queue
              #  + sessions + statut serveur) -> `uv run jarvis-app`
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
