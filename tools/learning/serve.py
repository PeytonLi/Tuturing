"""Run the internal CPU learning service."""

import argparse
import os
from pathlib import Path

import yaml

from apps.learning_service.engine import LearningEngine, build_strategist
from apps.learning_service.http import LearningHTTPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/learning/rules.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--db")
    args = parser.parse_args()
    if not os.environ.get("TUTOR_LEARNING_SERVICE_TOKEN"):
        parser.error("TUTOR_LEARNING_SERVICE_TOKEN must not be empty")
    config = yaml.safe_load(Path(args.config).read_text())
    db_path = args.db or config.get("db_path", "work/learning.sqlite3")
    engine = LearningEngine(config, db_path, strategist=build_strategist(config))
    try:
        with LearningHTTPServer((args.host, args.port), engine) as server:
            host, port = server.server_address
            print(f"Learning service listening on http://{host}:{port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        engine.close()


if __name__ == "__main__":
    main()
