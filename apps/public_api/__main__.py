"""Local entry point: `python -m apps.public_api --config configs/response/public_api_local.yaml`.

Secrets come from the environment variables the config names. With `learning.mode: standin`
the process uses Person A's test stand-in for B's learning service (transaction behavior only,
fixture plans replayed in a fixed order); it is a development aid, not B's service.
"""

from __future__ import annotations

import argparse
import logging
import sys

from apps.public_api.clients import HttpLearningClient, HttpResponseClient, InProcessResponseClient
from apps.public_api.config import PublicApiConfig, load_callers
from apps.public_api.server import PublicApiServer
from apps.public_api.service import PublicApi
from apps.public_api.storage import RequestStore

log = logging.getLogger("tutor.public_api")


def build_api(config: PublicApiConfig) -> PublicApi:
    store = RequestStore(config.storage_path)
    if config.learning.mode == "http":
        learning = HttpLearningClient(
            config.learning.base_url,
            service_token_env=config.learning.service_token_env,
            timeout_s=config.learning.timeout_s,
        )
    elif config.learning.mode == "standin":
        from tests.integration.response.learning_standin import LearningStandIn

        log.warning(
            "learning.mode=standin: using Person A's transaction stand-in, not B's learning service"
        )
        learning = LearningStandIn(plan_ttl_s=config.learning.standin_plan_ttl_s)
    else:
        raise SystemExit(f"unknown learning.mode {config.learning.mode!r}")
    if config.response.mode == "in_process":
        response = InProcessResponseClient()
    elif config.response.mode == "http":
        response = HttpResponseClient(
            config.response.base_url,
            service_token_env=config.response.service_token_env,
            timeout_s=config.response.timeout_s,
        )
    else:
        raise SystemExit(f"unknown response.mode {config.response.mode!r}")
    return PublicApi(store, learning, response, config=config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tuturing public API")
    parser.add_argument(
        "--config", required=True, help="YAML config, e.g. configs/response/public_api_local.yaml"
    )
    parser.add_argument("--host", help="override server.host")
    parser.add_argument("--port", type=int, help="override server.port")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    config = PublicApiConfig.load(args.config)
    callers = load_callers(config.tokens_env)
    if not callers:
        print(
            f"error: set {config.tokens_env} to comma-separated caller=token entries before"
            " starting (never put token values in the config)",
            file=sys.stderr,
        )
        return 2
    api = build_api(config)
    server = PublicApiServer(
        (args.host or config.host, args.port or config.port),
        api,
        callers,
        max_request_bytes=config.limits.max_request_bytes,
    )
    log.info(
        "public API listening on %s callers=%d store=%s learning=%s response=%s",
        server.base_url,
        len(callers),
        config.storage_path,
        config.learning.mode,
        config.response.mode,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
