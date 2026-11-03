import argparse
import json
from pathlib import Path
import sys
from .duct import Duct, write_candidate


def main(argv=None):
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


if __name__ == "__main__":
    raise SystemExit(main())
