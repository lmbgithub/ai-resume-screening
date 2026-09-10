"""Command line: resume-screen --resume R.md --job J.md"""

from __future__ import annotations

import argparse
import sys

from .documents import load_document
from .fakes import FakeClient
from .ollama import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, DEFAULT_HOST, OllamaClient, OllamaError
from .pipeline import DEFAULT_TOP_K, EmptyDocumentError, screen
from .report import render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resume-screen",
        description="Match a resume against a job description using a local Ollama model.",
    )
    parser.add_argument("--resume", required=True, help="path to a plain-text or Markdown resume")
    parser.add_argument("--job", required=True, help="path to a plain-text job description")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="resume lines shown to the model per requirement",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the deterministic fake backend; no model, no network",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        print("error: --top-k must be positive", file=sys.stderr)
        return 2

    client = (
        FakeClient()
        if args.offline
        else OllamaClient(
            host=args.host,
            chat_model=args.chat_model,
            embed_model=args.embed_model,
            timeout=args.timeout,
        )
    )
    try:
        resume = load_document(args.resume)
        job = load_document(args.job)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        result = screen(client, resume, job, top_k=args.top_k)
    except EmptyDocumentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(render(result), end="")
    # A run whose requirements could not all be scored is not a clean run, and
    # a caller wiring this into a script needs to be able to tell.
    return 1 if result.summary.unscored else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
