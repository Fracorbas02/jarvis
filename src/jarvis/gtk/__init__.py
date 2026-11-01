"""Interface GTK4 / libadwaita de Jarvis.

Nouvelle UI native (branche de refonte) : page d'accueil avec un seul bouton
« Lancer », vue de conversation vocale/texte centrée, menu avec le prompt
(diskussions LLM), et un vrai onglet de réglages (thème, GPU, contexte…).

Ne touche à rien d'autre : réutilise config/llm/agent/mcp/voice/gpu/sessions
existants (le pont async est dans backend.py).
"""
