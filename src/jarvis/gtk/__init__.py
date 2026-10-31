"""Interface GTK4 / libadwaita de Jarvis.

Nouvelle UI native (branche de refonte) : page d'accueil avec un seul bouton
\u00ab Lancer \u00bb, vue de conversation vocale/texte centr\u00e9e, menu avec le prompt
(diskussions LLM), et un vrai onglet de r\u00e9glages (th\u00e8me, GPU, contexte\u2026).

Ne touche \u00e0 rien d'autre : r\u00e9utilise config/llm/agent/mcp/voice/gpu/sessions
existants (le pont async est dans backend.py).
"""
