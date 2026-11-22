import numpy as np
import pytest

from quillgrep.ollama import ModelMissing, OllamaClient, OllamaUnavailable, check_ready, model_name_matches


def test_model_name_matching():
    assert model_name_matches("nomic-embed-text", "nomic-embed-text:latest")
    assert model_name_matches("qwen3.5:9b", "qwen3.5:9b")
    assert not model_name_matches("qwen3.5:9b", "qwen3.5:latest")
    assert not model_name_matches("qwen3.5", "qwen3.5:9b")


def test_unreachable_ollama_says_how_to_install_and_start_it():
    client = OllamaClient("http://127.0.0.1:9", timeout=2)  # nothing listens on the discard port
    with pytest.raises(OllamaUnavailable) as e:
        client.installed_models()
    assert "https://ollama.com/download" in str(e.value) and "ollama serve" in str(e.value)


def test_check_ready_names_the_exact_pull_command(fake, client):
    fake.installed = ["nomic-embed-text:latest"]
    problems = check_ready(client, ["nomic-embed-text", "qwen3.5:9b"])
    assert len(problems) == 1 and "ollama pull qwen3.5:9b" in problems[0]
    assert check_ready(OllamaClient("http://127.0.0.1:9", 2), ["x"])[0].startswith("Could not reach Ollama")


def test_embeddings_are_unit_vectors(client):
    vecs = client.embed("nomic-embed-text", ["tomato soup", "seed swap"])
    assert vecs.shape[0] == 2
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_missing_model_raises_with_pull_hint(fake, client):
    fake.installed = []
    with pytest.raises(ModelMissing, match="ollama pull nomic-embed-text"):
        client.embed("nomic-embed-text", ["x"])


def test_chat_and_stream_hide_thinking(fake, client):
    fake.think_prefix = True
    fake.answer = lambda prompt: "Hello there friend"
    msgs = [{"role": "user", "content": "hi"}]
    assert client.chat("qwen3.5:9b", msgs) == "Hello there friend"
    assert "".join(client.chat_stream("qwen3.5:9b", msgs)) == "Hello there friend"
    assert fake.chat_calls[0]["think"] is False
