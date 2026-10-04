from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from chonkie.types import Document
from liteparse import LiteParse

SUPPORTED_FORMATS = ("pdf", "txt", "csv", "xlsx", "xls", "docx", "doc", "md", "markdown")

# Parsed with LiteParse (structured / binary office formats).
LITEPARSE_EXTENSIONS = frozenset({"pdf", "csv", "xlsx", "xls", "docx", "doc"})

# Read as UTF-8 text (no LiteParse overhead).
TEXT_EXTENSIONS = frozenset({"txt", "md", "markdown", "rst"})


class Parser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> str:
        pass


class PlainTextParser(Parser):
    def parse(self, file_path: str) -> str:
        return Path(file_path).read_text(encoding="utf-8")


class LiteParseParser(Parser):
    """Parse PDFs and office documents via LiteParse."""

    def __init__(self) -> None:
        self.lp = LiteParse(
            output_format="markdown",
            image_mode="placeholder",
            extract_links=True,
        )

    def parse(self, file_path: str) -> str:
        return self.lp.parse(file_path).text


class PDFParser(LiteParseParser):
    """Backward-compatible name; handles all LiteParse-supported formats."""


def extension_for(path: str | Path) -> str:
    return Path(path).suffix.lower().lstrip(".")


def get_parser(file_path: str | Path) -> Parser:
    ext = extension_for(file_path)
    if ext in TEXT_EXTENSIONS:
        return PlainTextParser()
    if ext in LITEPARSE_EXTENSIONS:
        return LiteParseParser()
    if ext in SUPPORTED_FORMATS:
        return LiteParseParser()
    supported = ", ".join(sorted(SUPPORTED_FORMATS))
    raise ValueError(f"Unsupported file type {ext!r}. Supported: {supported}")


def parse_file(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    return get_parser(path).parse(str(path))


def file_to_document(file_path: str | Path) -> Document:
    path = Path(file_path).resolve()
    text = parse_file(path)
    return Document(
        content=text,
        metadata={
            "source": path.name,
            "path": str(path),
        },
    )
