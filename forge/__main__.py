"""Main CLI dispatcher supporting both legacy invocations and modern subcommands."""
import json
from pathlib import Path
import sys
from forge.duct import Duct, write_candidate
from forge.cli.main import app


def legacy_main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="forge", description="Generate an experimental hollow rectangular duct STL and geometry report.")
    parser.add_argument("spec", type=Path, help="JSON specification with dimensions in mm")
    parser.add_argument("--out", required=True, type=Path, help="Output directory; files may be overwritten")
    args = parser.parse_args(argv)
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        duct = Duct(**spec)
        report = write_candidate(duct, args.out)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"forge: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.out), "geometry_sha256": report["geometry_sha256"], "evaluation_type": report["evaluation_type"]}))
    return 0


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    # If invoked with subcommands (or empty or help flag)
    subcommands = {"generate", "simulate", "dfm", "openfoam", "su2", "optimize", "lab", "serve"}
    if args and args[0] in subcommands:
        return app(args)
    elif args and ("--out" in args or len(args) == 2 or (len(args) > 0 and args[0].endswith(".json"))):
        return legacy_main(args)
    elif not args:
        return app([])
    else:
        return app(args)


if __name__ == "__main__":
    raise SystemExit(main())
