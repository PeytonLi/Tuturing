"""Start the response service.

    uv run python -m tools.response.serve --config configs/response/serve_stub.yaml

``--check`` validates the config, resolves the named environment variables and builds the
pipeline without binding a port. ``--host`` and ``--port`` override the config.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from apps.response_service.config import ConfigError, load_config
from apps.response_service.server import (
    GENERATE_PATH,
    LOGGER,
    build_service,
    describe_service,
    make_server,
)
from packages.response.responders import ResponderConfigError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, help="service config YAML")
    parser.add_argument("--host", help="override service.host")
    parser.add_argument("--port", type=int, help="override service.port (0 picks a free port)")
    parser.add_argument(
        "--check", action="store_true", help="validate the config and exit without serving"
    )
    args = parser.parse_args(argv)
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
        return 0
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
