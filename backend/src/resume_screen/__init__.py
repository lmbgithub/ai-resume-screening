"""Resume screening against a local Ollama model."""

from .advice import Point, Suggestions
from .documents import Chunk, chunk_document, load_document
from .extract import Extracted, ExtractionError, extract
from .fakes import FakeClient
from .judge import Verdict, parse_verdict
from .ollama import Client, OllamaClient, OllamaError
from .pipeline import ScreenResult, screen
from .report import render
from .requirements import Requirement, RequirementSet, find_requirements, parse_requirements
from .serialize import result_to_dict

__all__ = [
    "Chunk",
    "Client",
    "Extracted",
    "ExtractionError",
    "FakeClient",
    "OllamaClient",
    "OllamaError",
    "Point",
    "Requirement",
    "RequirementSet",
    "ScreenResult",
    "Suggestions",
    "Verdict",
    "chunk_document",
    "extract",
    "find_requirements",
    "load_document",
    "parse_requirements",
    "parse_verdict",
    "render",
    "result_to_dict",
    "screen",
]
