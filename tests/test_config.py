import pytest

from quillgrep.config import ConfigError, load_config, starter_config


def write(tmp_path, text):
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_when_only_vault_is_given(tmp_path):
    cfg = load_config(write(tmp_path, f'vault = "{tmp_path.as_posix()}"'))
    assert cfg.models.embedding == "nomic-embed-text"
    assert cfg.models.chat == "qwen3.5:9b"
    assert cfg.models.ask_model == "qwen3.5:4b" and cfg.models.deep_model == "qwen3.5:9b"
    assert cfg.models.planner_model == "qwen3.5:2b" and cfg.models.keep_alive == "1h"
    assert cfg.retrieval.excerpts == 6 and cfg.output.links == "auto"
    assert cfg.index.exclude == [".trash", ".obsidian"]
    assert cfg.retrieval.per_note == 3
    assert cfg.privacy.show_titles is True


def test_starter_config_is_valid_and_round_trips(tmp_path):
    vault = tmp_path / "My Vault"
    vault.mkdir()
    cfg = load_config(write(tmp_path, starter_config(str(vault).replace("\\", "/"))))
    assert cfg.vault_path == vault
    assert cfg.privacy.redact == ["email", "phone", "number"]


def test_unknown_setting_is_rejected_with_a_hint(tmp_path):
    with pytest.raises(ConfigError, match=r"Unknown setting `chatt` in \[models\]"):
        load_config(write(tmp_path, '[models]\nchatt = "x"'))


def test_wrong_types_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="must be a list of strings"):
        load_config(write(tmp_path, '[index]\nexclude = ".trash"'))
    with pytest.raises(ConfigError, match="must be true or false"):
        load_config(write(tmp_path, '[privacy]\nshow_titles = "yes"'))
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(write(tmp_path, '[retrieval]\nper_note = "three"'))


def test_windows_backslash_mistake_gets_a_tip(tmp_path):
    with pytest.raises(ConfigError, match="forward slashes"):
        load_config(write(tmp_path, 'vault = "C:\\Users\\me\\Vault"'))


def test_missing_vault_and_missing_folder_messages(tmp_path):
    cfg = load_config(write(tmp_path, ""))
    with pytest.raises(ConfigError, match="quillgrep init"):
        _ = cfg.vault_path
    cfg = load_config(write(tmp_path, 'vault = "/definitely/not/here"'))
    with pytest.raises(ConfigError, match="does not exist"):
        _ = cfg.vault_path


def test_explicit_config_that_does_not_exist_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_index_path_is_per_vault(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    ca, cb = load_config(vault=str(a)), load_config(vault=str(b))
    assert ca.index_path != cb.index_path


def test_empty_optional_models_fall_back_to_the_main_model(tmp_path):
    cfg = load_config(write(tmp_path, '[models]\nchat = "big:9b"\nask_chat = ""\nplanner = ""\n'))
    assert cfg.models.ask_model == cfg.models.deep_model == cfg.models.planner_model == "big:9b"
    cfg = load_config(write(tmp_path, '[models]\nchat = "big:9b"\ndeep_chat = "huge:30b"\nplanner = ""\n'))
    assert cfg.models.deep_model == "huge:30b" and cfg.models.planner_model == "huge:30b"


def test_output_settings_and_vault_name(tmp_path):
    vault = tmp_path / "My Notes"
    vault.mkdir()
    cfg = load_config(write(tmp_path, f'vault = "{vault.as_posix()}"'))
    assert cfg.obsidian_vault_name == "My Notes"  # defaults to the folder's name
    out = '\n[output]\nvault_name = "Brain"\nlinks = "never"\n'
    cfg = load_config(write(tmp_path, f'vault = "{vault.as_posix()}"' + out))
    assert cfg.obsidian_vault_name == "Brain" and cfg.output.links == "never"
    with pytest.raises(ConfigError, match="links"):
        load_config(write(tmp_path, '[output]\nlinks = "sometimes"\n'))
