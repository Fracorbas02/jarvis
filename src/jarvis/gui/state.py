"""État partagé de la GUI (une instance par processus, servie à chaque client)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .backend import GuiBackend
from .models import ModelInfo
from .server import ServerStatus


@dataclass
class GuiState:
    backend: GuiBackend | None = None
    backend_error: str | None = None
    models_by_mode: dict[str, list[ModelInfo]] = field(default_factory=dict)
    mode: str = "llm"
    # mode -> ModelInfo choisi dans la liste déroulante
    selected: dict[str, ModelInfo | None] = field(default_factory=dict)
    server_status: ServerStatus | None = None
    reset_listeners: list[Callable[[], None]] = field(default_factory=list)
    # refs UI remplies par build_page (sidebar / zone principale)
    sidebar: object | None = None
    main_box: object | None = None

    def notify_reset(self) -> None:
        for listener in self.reset_listeners:
            listener()
