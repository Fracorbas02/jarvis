"""Validation visuelle du rendu TUI, sans LLM.

Pousse des fragments simulés (raisonnement, réponse Markdown) dans
jarvis.tui.Display et affiche le résultat. Usage :
    uv run python tests/test_tui.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402

from jarvis.tui import Display  # noqa: E402

MD = """## CVE récentes Arista (30 derniers jours)

| CVE | Score | Sévérité | Date |
|-----|-------|----------|------|
| **CVE-2026-102163** | **8.8** | HIGH | 2026-10-06 |
| CVE-2026-102162 | 8.8 | HIGH | 2026-10-06 |
| CVE-2026-102167 | 7.5 | HIGH | 2026-10-06 |

### Détail de CVE-2026-102163

Vecteur : `CVSS:3.1/AV:A/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H` — CWE-787.

- aucun privilège requis, proximité RF suffisante
- [Arista Security Advisory 0195](https://www.arista.com/en/support/advisories-notices)

```python
print("exemple de bloc de code")
```
"""


def main() -> None:
    console = Console(width=90, force_terminal=True, color_system="truecolor")
    d = Display(console)
    for chunk in ["Je cherche ", "les CVE récentes ", "pour Arista ", "avec search_cve…"]:
        d.reasoning(chunk)
        time.sleep(0.05)
    d.end_llm_stream()  # fin du stream LLM (panneau replié en ligne résumé)
    d.tool_call("cve__search_cve", {"keywords": "Arista", "days": 30, "limit": 10})
    time.sleep(0.1)
    d.tool_result("cve__search_cve", {"keywords": "Arista", "days": 30, "limit": 10},
                  True, 1.23,
                  "CVE-2026-102163 | 8.8 | HIGH\nCVE-2026-102162 | 8.8 | HIGH\n"
                  "CVE-2026-102167 | 7.5 | HIGH\n... 7 lignes au total")
    for chunk in ["Voici le résultat :\n\n", MD[: len(MD) // 2], MD[len(MD) // 2 :]]:
        d.content(chunk)
        time.sleep(0.05)
    d.end_turn()
    print("\n--- /tool 1 ---")
    d.show_tool(1)
    print("\n--- /thinking ---")
    d.show_reasoning()


if __name__ == "__main__":
    main()
