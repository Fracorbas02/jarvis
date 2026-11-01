"""Tags d'émotion/pace inline pour le TTS (Voxtral 3, API Mistral).

Voxtral TTS 3 accepte des tags inline dans `input`, posés en début de ligne :
  [emotion=happy][pace=fast] Bonjour !
34 émotions + 2 pace (slow/fast). Le LLM de Jarvis peut émettre un tel tag en
début de réponse ; on l'extrait (pour le TTS uniquement) et on le retire du
texte affiché/sauvegardé, afin qu'il ne soit jamais lu à l'écran ni parlé tel quel.

Format retenu côté LLM : le tag en toute première ligne, ex. "[emotion=calm]".
"""

from __future__ import annotations

import re

# syntaxe exacte des tags Voxtral 3 : [emotion=<valeur>][pace=<valeur>]
TAG_RE = re.compile(r"\[emotion=[a-z_]+\]|\[pace=(?:slow|fast)\]", re.IGNORECASE)
# on n'accepte un tag en tête de réponse que s'il est en début de 1re ligne
LEAD_TAGS_RE = re.compile(r"^((?:\[emotion=[a-z_]+\]|\[pace=(?:slow|fast)\]\s*)+)",
                          re.IGNORECASE)

# émotions documentées par Mistral (sélection ; la doc parle de 34 tags)
EMOTIONS = [
    "neutral", "happy", "sad", "angry", "excited", "calm", "curious",
    "surprised", "confused", "sarcastic", "whisper", "laugh", "sigh",
    "frustrated", "confident", "nervous", "playful", "serious", "warm",
]
PACES = ["slow", "fast"]


def extract_lead_tags(text: str) -> tuple[list[str], str]:
    """Extrait les tags en tête de `text` (1re ligne). Retourne (tags, texte_nettoyé)."""
    if not text:
        return [], text
    m = LEAD_TAGS_RE.match(text)
    if not m:
        return [], text
    tags = TAG_RE.findall(m.group(1))
    rest = text[m.end():].lstrip()
    return tags, rest


def strip_all_tags(text: str) -> str:
    """Retire TOUS les tags [emotion=..]/[pace=..] du texte (sécurité affichage)."""
    return TAG_RE.sub("", text)


def apply_tags(text: str, tags: list[str]) -> str:
    """Préfixe `text` avec les tags (pour l'envoi TTS uniquement)."""
    if not tags:
        return text
    prefix = "".join(tags) + " "
    return prefix + text
