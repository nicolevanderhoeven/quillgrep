"""Quillgrep: private, local AI search and answers over an Obsidian vault."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quillgrep")
except PackageNotFoundError:  # running from a source checkout that is not installed
    __version__ = "0.0.0+source"
