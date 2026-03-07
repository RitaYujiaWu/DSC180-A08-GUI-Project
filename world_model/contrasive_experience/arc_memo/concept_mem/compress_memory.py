#!/usr/bin/env python3
"""
Compress ConceptMemory JSON into a fixed text string for retrieval prompts (GUI agent).
"""

import argparse
import logging
from pathlib import Path

from concept import ConceptMemory

DEFAULT_MEMORY_PATH = Path("/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/arc_memo/output/memory.json")
DEFAULT_OUT_PATH = Path("/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/arc_memo/output/gui_init_mem.txt")


def setup_logging() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	)

def main() -> None:
	setup_logging()
	parser = argparse.ArgumentParser(description="Compress ConceptMemory to string (GUI agent)")
	parser.add_argument("--memory", type=str, default=str(DEFAULT_MEMORY_PATH), help="Path to memory.json")
	parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH), help="Path to write compressed memory text")
	args = parser.parse_args()

	memory_path = Path(args.memory)
	out_path = Path(args.out)

	logging.info(f"Loading memory from: {memory_path}")
	cm = ConceptMemory()
	cm.load_from_file(memory_path)

	logging.info("Rendering memory to string")
	mem_str = cm.to_string()

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(mem_str)
	logging.info(f"Wrote compressed memory to: {out_path}")


if __name__ == "__main__":
	main()
