"""Allow ``python -m hscode`` invocation."""

from hscode.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
