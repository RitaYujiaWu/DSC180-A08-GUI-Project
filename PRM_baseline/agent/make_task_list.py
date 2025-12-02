import argparse
import json
from pathlib import Path


def build_meta_from_examples_root(examples_root: Path) -> dict[str, list[str]]:
    """
    Scan examples_root and return {domain: [id, ...]} where
    each domain is a subfolder and each id is a JSON filename stem.
    """
    meta: dict[str, list[str]] = {}

    for domain_dir in sorted(examples_root.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name
        ids: list[str] = []

        for f in sorted(domain_dir.glob("*.json")):
            ids.append(f.stem)

        if ids:
            meta[domain] = ids

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a meta JSON mapping domain -> list of example IDs "
        "from a folder tree like evaluation_examples/generated_examples."
    )
    parser.add_argument(
        "--examples_root",
        type=Path,
        default=Path("evaluation_examples/generated_examples"),
        help="Root folder containing per-domain example JSONs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output meta JSON file to write (e.g. evaluation_examples/test_generated_auto.json).",
    )
    args = parser.parse_args()

    meta = build_meta_from_examples_root(args.examples_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in meta.values())
    print(f"Wrote meta for {len(meta)} domains, {total} examples to {args.output}")


if __name__ == "__main__":
    main()