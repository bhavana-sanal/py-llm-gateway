"""Command-line entry point: ``llm-gateway --config gateway.toml``."""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sequence
from pathlib import Path

import uvicorn
from pydantic import ValidationError

from llm_gateway import __version__
from llm_gateway.app import create_app
from llm_gateway.config import load_config
from llm_gateway.logs import configure_logging


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="llm-gateway", description="Self-hosted, OpenAI-compatible LLM gateway."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("gateway.toml"), help="path to the TOML config"
    )
    parser.add_argument("--host", help="override [server].host")
    parser.add_argument("--port", type=int, help="override [server].port")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        parser.exit(2, f"llm-gateway: cannot load {args.config}: {exc}\n")

    configure_logging(config.server.log_level)
    uvicorn.run(
        create_app(config),
        host=args.host or config.server.host,
        port=args.port or config.server.port,
        log_config=None,  # keep uvicorn's records flowing through our JSON handler
        access_log=False,  # the app logs each request itself, with its request ID
    )
