from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # lets tests import fake_ollama / pdfmaker

from fake_ollama import FakeOllama  # noqa: E402
from quillgrep.config import Config, load_config  # noqa: E402
from quillgrep.indexer import index_vault  # noqa: E402
from quillgrep.ollama import OllamaClient  # noqa: E402
from quillgrep.store import Store  # noqa: E402

SAMPLE_VAULT = Path(__file__).parent.parent / "examples" / "sample-vault"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real config, index or audit log of whoever runs the tests."""
    monkeypatch.setenv("QUILLGREP_HOME", str(tmp_path / "qg-home"))
    monkeypatch.delenv("QUILLGREP_CONFIG", raising=False)


@pytest.fixture
def fake():
    server = FakeOllama().start()
    yield server
    server.stop()


@pytest.fixture
def vault(tmp_path) -> Path:
    """A writable copy of the synthetic sample vault."""
    dest = tmp_path / "vault"
    shutil.copytree(SAMPLE_VAULT, dest)
    return dest


@pytest.fixture
def make_cfg(fake, vault):
    def make(**overrides) -> Config:
        cfg = Config(vault=str(vault))
        cfg.models.host = fake.url
        cfg.models.timeout_seconds = 30
        for dotted, value in overrides.items():
            section, _, key = dotted.partition("__")
            setattr(getattr(cfg, section), key, value)
        return cfg

    return make


@pytest.fixture
def cfg(make_cfg) -> Config:
    return make_cfg()


@pytest.fixture
def client(cfg) -> OllamaClient:
    return OllamaClient(cfg.models.host, 30)


@pytest.fixture
def store(cfg):
    s = Store(cfg.index_path)
    yield s
    s.close()


@pytest.fixture
def indexed(cfg, client, store):
    """The sample vault, indexed against the fake Ollama."""
    index_vault(cfg, client, store)
    return store


@pytest.fixture
def config_file(tmp_path, fake, vault):
    """A config file on disk pointing at the fake Ollama (for CLI and subprocess tests)."""
    from quillgrep.config import starter_config

    path = tmp_path / "config.toml"
    text = starter_config(str(vault).replace("\\", "/")).replace("http://127.0.0.1:11434", fake.url)
    path.write_text(text, encoding="utf-8")
    load_config(path)  # fails loudly if the starter config is ever invalid
    return path
