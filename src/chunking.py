

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

from chonkie import RecursiveChunker
from chonkie.types import Document, RecursiveLevel, RecursiveRules

def _level_to_dict(level: RecursiveLevel) -> dict[str, Any]:
    return {
        "delimiters": level.delimiters,
        "whitespace": level.whitespace,
        "include_delim": level.include_delim,
    }


def _dict_to_level(d: dict[str, Any]) -> RecursiveLevel:
    return RecursiveLevel(
        delimiters=d.get("delimiters"),
        whitespace=d.get("whitespace", False),
        include_delim=d.get("include_delim", "prev"),
    )


def rules_to_dict(rules: RecursiveRules) -> dict[str, Any]:
    return {"levels": [_level_to_dict(lvl) for lvl in rules.levels]}


def dict_to_rules(d: dict[str, Any]) -> RecursiveRules:
    return RecursiveRules(levels=[_dict_to_level(lvl) for lvl in d["levels"]])


# --------------------------------------------------------------------
# Full config: save
# --------------------------------------------------------------------

def save_chunker_config(
    path: Union[str, Path],
    *,
    rules: RecursiveRules,
    tokenizer: str = "character",
    chunk_size: int = 512,
    min_characters_per_chunk: int = 100,
    chunk_type: str = "recursive",
) -> None:
    """Write the chunker's type + settings + rules to a JSON config file."""
    config = {
        "chunk_type": chunk_type,          # identifies which chunker class to rebuild
        "tokenizer": tokenizer,
        "chunk_size": chunk_size,
        "min_characters_per_chunk": min_characters_per_chunk,
        "rules": rules_to_dict(rules),
    }
    Path(path).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------
# Full config: load (cached, so re-calling this is essentially free)
# --------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_chunker(path: str) -> RecursiveChunker:
    """
    Build a RecursiveChunker from a saved config file.
    Cached per path: the first call parses JSON and constructs the
    rules/chunker; every later call with the same path string returns
    the same instance instantly, with no disk I/O or rebuild.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))

    if config["chunk_type"] != "recursive":
        raise ValueError(f"Unsupported chunk_type: {config['chunk_type']!r}")

    return RecursiveChunker(
        tokenizer=config["tokenizer"],
        chunk_size=config["chunk_size"],
        min_characters_per_chunk=config["min_characters_per_chunk"],
        rules=dict_to_rules(config["rules"]),
    )


def clear_chunker_cache(path: Optional[str] = None) -> None:
    """Force a reload next time load_chunker() is called (e.g. after editing the config)."""
    load_chunker.cache_clear()


DEFAULT_CHUNKER_CONFIG_PATH = "chunker_config.json"


def ensure_chunker_config(
    path: Union[str, Path] = DEFAULT_CHUNKER_CONFIG_PATH,
    *,
    chunk_size: int = 512,
    min_characters_per_chunk: int = 100,
    tokenizer: str = "character",
) -> Path:
    """Create a default recursive chunker config if missing."""
    config_path = Path(path)
    if config_path.is_file():
        return config_path

    template = RecursiveChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        min_characters_per_chunk=min_characters_per_chunk,
    )
    save_chunker_config(
        config_path,
        rules=template.rules,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        min_characters_per_chunk=min_characters_per_chunk,
    )
    return config_path


def chunk_file(
    file_path: Union[str, Path],
    chunker: RecursiveChunker,
) -> Document:
    """Parse a file and split it into chunks with source metadata."""
    from parsing import file_to_document

    document = file_to_document(file_path)
    return chunker.chunk_document(document)


