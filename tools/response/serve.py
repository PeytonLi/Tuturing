"""Start the response service.

    uv run python -m tools.response.serve --config configs/response/serve_stub.yaml

``--check`` validates the config, resolves the named environment variables and builds the
pipeline without binding a port or contacting any model. ``--check --probe`` additionally
sends one minimal request in the configured format (same token field, temperature,
reasoning_effort and extra_body, a synthetic two-message prompt) and reports whether the
provider accepts it; it costs a few tokens, so it runs only when asked and never in tests.
``--host`` and ``--port`` override the config.

Exit codes: 0 ok, 2 configuration error, 3 the probe was rejected or failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.parse

from apps.response_service.config import ConfigError, load_config
from apps.response_service.server import (
    GENERATE_PATH,
    LOGGER,
    build_service,
    describe_service,
    make_server,
)
from packages.response.responders import ResponderConfigError


def probe(responder, max_tokens: int) -> dict:
    """One real-format request through the configured responder; never prints the key."""
    send = getattr(responder, "probe", None)
    if not callable(send):
        return {"status": "skipped", "detail": f"{responder.kind} has no remote endpoint"}
    host = urllib.parse.urlsplit(responder.base_url).hostname
    print(
        f"probe: sending one request to {host} for model {responder.model}", file=sys.stderr
    )
    return send(max_tokens=max_tokens)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, help="service config YAML")
    parser.add_argument("--host", help="override service.host")
    parser.add_argument("--port", type=int, help="override service.port (0 picks a free port)")
    parser.add_argument(
        "--check", action="store_true", help="validate the config and exit without serving"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="with --check: send one minimal real-format request to the model endpoint",
    )
    args = parser.parse_args(argv)
    if args.probe and not args.check:
        parser.error("--probe only runs together with --check")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    overrides = {
        key: value for key, value in (("host", args.host), ("port", args.port)) if value is not None
    }
    try:
        config = load_config(args.config, overrides=overrides)
        service = build_service(config)
    except (ConfigError, ResponderConfigError, OSError, ValueError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    summary = describe_service(config, service)
    if args.check:
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not args.probe:
            return 0
        result = probe(service.pipeline.responder, config.max_output_tokens)
        print(json.dumps({"probe": result}, indent=2, sort_keys=True))
        return 0 if result["status"] in {"accepted", "skipped"} else 3
    server = make_server(config, service)
    host, port = server.server_address[:2]
    LOGGER.info(json.dumps({"event": "serving", **summary, "port": port}, sort_keys=True))
    print(f"response service listening on http://{host}:{port}{GENERATE_PATH}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
