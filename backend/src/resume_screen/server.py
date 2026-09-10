"""Entry point for the API container: `python -m resume_screen.server`.

Configuration comes from the environment because that is what a container
composes with; every value also has a flag for running it by hand.
"""

from __future__ import annotations

import argparse
import os

from .api import Config, serve
from .fakes import FakeClient
from .ollama import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, DEFAULT_HOST, OllamaClient
from .pipeline import DEFAULT_TOP_K


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resume-screen-api")
    parser.add_argument("--host", default=os.environ.get("API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("API_PORT", "8000")))
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST))
    parser.add_argument("--chat-model", default=os.environ.get("CHAT_MODEL", DEFAULT_CHAT_MODEL))
    parser.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", DEFAULT_TOP_K)))
    parser.add_argument(
        "--timeout", type=float, default=float(os.environ.get("OLLAMA_TIMEOUT", "300"))
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=_flag("OFFLINE"),
        help="serve the deterministic fake backend; no model, no network",
    )
    parser.add_argument(
        "--no-suggestions",
        action="store_true",
        default=_flag("NO_SUGGESTIONS"),
        help="skip the suggestions call; scoring is unaffected",
    )
    return parser


def _progress(message: str) -> None:
    print(f"  … {message}", flush=True)


def build_config(args: argparse.Namespace) -> Config:
    if args.offline:
        return Config(
            client_factory=FakeClient,
            top_k=args.top_k,
            with_suggestions=not args.no_suggestions,
            on_progress=_progress,
        )
    # A factory, not an instance: the client is frozen and cheap, and building
    # it per request keeps the handler free of shared mutable state.
    return Config(
        client_factory=lambda: OllamaClient(
            host=args.ollama_host,
            chat_model=args.chat_model,
            embed_model=args.embed_model,
            timeout=args.timeout,
        ),
        top_k=args.top_k,
        with_suggestions=not args.no_suggestions,
        on_progress=_progress,
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - blocks forever
    args = build_parser().parse_args(argv)
    serve(args.host, args.port, config=build_config(args))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
