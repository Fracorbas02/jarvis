"""Outil interne jarvis__voice_tone : le LLM fixe le ton de la voix (TTS).

Complement des tags inline [emotion=..][pace=..] : ici le modele APPELE
explicitement l'outil interne avec l'emotion choisie ; Jarvis l'enregistre
dans trace.tts_tags (envoye au TTS uniquement, jamais affiche). Plus fiable
qu'un tag oublie dans le texte, car c'est un appel d'outil tracable.

Emotions/paces valides : voir jarvis.tts_tags (EMOTIONS, PACES).
"""

from __future__ import annotations

from .agent import InternalTool
from .tts_tags import EMOTIONS, PACES


def _norm(v: str) -> str:
    return str(v).strip().lower().replace("[", "").replace("]", "")


def make_voice_tone_tool():
    """Construit l'outil interne. fn(agent, history, args) -> (result, None).

    Le tag est pose sur agent._last_tts_tags (lu par Agent.turn a la fin du tour
    pour remplir trace.tts_tags)."""
    async def fn(agent, history, args):
        emotion = _norm(args.get("emotion", "neutral"))
        pace = _norm(args.get("pace", ""))
        if emotion not in EMOTIONS:
            return (f"ERREUR: emotion '{emotion}' inconnue. Dispo: {', '.join(EMOTIONS)}", None)
        if pace and pace not in PACES:
            return (f"ERREUR: pace '{pace}' invalide (slow|fast).", None)
        tags = [f"[emotion={emotion}]"]
        if pace:
            tags.append(f"[pace={pace}]")
        agent._last_tts_tags = tags  # repris par Agent.turn -> trace.tts_tags
        return (f"Ton de voix enregistr\u00e9 pour la lecture : {' '.join(tags)} "
                f"(envoy\u00e9 au TTS uniquement, jamais affich\u00e9).", None)

    return InternalTool(
        description=(
            "Fixe le TON DE LA VOIX pour lire ta prochaine r\u00e9ponse \u00e0 voix haute "
            "(TTS Voxtral 3, tags [emotion=..][pace=..]). Appelle-le quand tu veux "
            "adapter l\u2019\u00e9motion/pace de ta r\u00e9ponse \u00e0 son contenu "
            "(ex. emotion='excited' pour une bonne nouvelle, 'calm' pour une explication, "
            "'serious' pour un incident). args: emotion (une des 34 \u00e9motions: "
            + ", ".join(EMOTIONS[:12]) + "\u2026), pace optionnel ('slow'|'fast'). "
            "Le tag n\u2019appara\u00eetra pas dans le texte affich\u00e9."
        ),
        fn=fn,
    )


voice_tone_tool = make_voice_tone_tool()
