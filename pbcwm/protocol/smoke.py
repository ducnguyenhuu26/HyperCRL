"""CLI for the reproducible placeholder lifetime protocol."""

import argparse
import json

from .config import load_protocol_config
from .runner import PlaceholderLifetimeRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pbcwm/configs/protocol.yaml")
    parser.add_argument("--environment", default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=137)
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()
    config = load_protocol_config(args.config)
    summary = PlaceholderLifetimeRunner(config, args.environment, args.seed, episode_length=args.episode_length).run(log_path=args.log_path)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
