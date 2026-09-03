#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Pin the shared checks and vendored guidance to one audited revision.
#     "marin-style @ git+https://github.com/marin-community/marin-style@dcbc8d5a81451ebb055b196be426e87c6ca963a6",
# ]
# ///
"""Run the pinned Marin-style checks configured for Slime."""

from marin_style.precommit import main

if __name__ == "__main__":
    raise SystemExit(main())
