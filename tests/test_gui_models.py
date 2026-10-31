"""Tests de la découverte des modèles de la GUI (sans LLM, sans GUI)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jarvis.gui.models import classify_mode, discover_models, format_size


def test_classify_mode():
    assert classify_mode("Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf") == "llm"
    assert classify_mode("Mistral-Small-3.3-24B-Q4_K_M.gguf") == "llm"
    assert classify_mode("faster-whisper-large-v3-Q8.gguf") == "stt"
    assert classify_mode("ggml-large-v3-q5_0.bin.gguf") == "stt"  # convention whisper.cpp
    assert classify_mode("ggml-base.en.gguf") == "stt"
    assert classify_mode("piper-voice-fr_FR-siwis-medium.onnx.gguf") == "tts"
    assert classify_mode("kokoro-v1.0.fp16.gguf") == "tts"
    assert classify_mode("xtts-v2-Q4.gguf") == "tts"


def test_format_size():
    assert format_size(512) == "512 o"
    assert format_size(2048) == "2.0 Ko"
    assert format_size(21 * 1024**3) == "21.0 Go"


def test_discover_models_tmp(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "Qwen3.6-Q4.gguf").write_bytes(b"x" * 2048)
        sub = root / "whisper"
        sub.mkdir()
        (sub / "ggml-base.en.gguf").write_bytes(b"x" * 1024)
        (root / "kokoro.onnx.gguf").write_bytes(b"x" * 512)
        # répertoires dédiés stt/ et tts/ (noms de fichiers opaques)
        (root / "stt").mkdir()
        (root / "stt" / "Q4_K_M.gguf").write_bytes(b"x" * 256)
        (root / "tts").mkdir()
        (root / "tts" / "voxtral-4b-tts-q4_k.gguf").write_bytes(b"x" * 128)
        # un doublon via un sous-répertoire ne doit apparaître qu'une fois
        found = discover_models(extra_dirs=[root], use_defaults=False)
        assert [m.name for m in found["llm"]] == ["Qwen3.6-Q4.gguf"]
        assert found["llm"][0].size_bytes == 2048
        assert [m.name for m in found["stt"]] == ["ggml-base.en.gguf", "Q4_K_M.gguf"]
        assert [m.name for m in found["tts"]] == ["kokoro.onnx.gguf", "voxtral-4b-tts-q4_k.gguf"]


def test_discover_models_real_dirs():
    # les répertoires par défaut existent sur la machine de dev : au moins un LLM
    found = discover_models()
    assert any(m.mode == "llm" for models in found.values() for m in models)


if __name__ == "__main__":
    test_classify_mode()
    test_format_size()
    test_discover_models_tmp()
    test_discover_models_real_dirs()
    print("OK")
