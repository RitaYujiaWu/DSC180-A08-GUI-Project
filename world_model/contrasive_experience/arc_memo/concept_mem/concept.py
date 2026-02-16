# concept_mem/concept.py
from __future__ import annotations
import itertools
import re
from dataclasses import asdict, dataclass, field

import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable

import json
import yaml

try:
    # Prefer package-style imports when available
    from arc_memo.concept_mem.constants import REPO_ROOT  # type: ignore
    from arc_memo.concept_mem.utils import extract_yaml_block  # type: ignore
except ModuleNotFoundError:
    # Fallback to local imports when running as a script
    from constants import REPO_ROOT
    from utils import extract_yaml_block

logger = logging.getLogger(__name__)

# --------------------------- Utilities --------------------------------- #
_TYPE_DEF_RE = re.compile(r"^\s*([^:=\s]+)\s*:=\s*(.+)$")


def maybe_parse_typedef(s: str | None) -> tuple[str, str] | None:
    """
    If `s` matches the pattern 'Name := python_like_annotation', return (Name, annotation).
    Otherwise return None.
    """
    if not s:
        return None
    m = _TYPE_DEF_RE.match(s)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


# ------------------------ Data structures ------------------------------ #
@dataclass
class ParameterSpec:
    name: str
    typing: str | None = None
    description: str | None = None


@dataclass
class Concept:
    """
    Schema
    ------
    - name: str
    - type: 'structure' | 'routine'
    - cues: List[str]                      # how to detect this concept is relevant
    - implementation: List[str]            # coding hints / pseudocode lines
    - used_in: List[str]
    """

    name: str
    type: str  # 'structure' | 'routine'
    cues: list[str] = field(default_factory=list)
    implementation: list[str] = field(default_factory=list)
    used_in: list[str] = field(default_factory=list)
    
    # Legacy fields for backward compatibility (will be ignored)
    kind: str | None = None
    routine_subtype: str | None = None
    output_typing: str | None = None
    parameters: list[ParameterSpec] = field(default_factory=list)
    description: str | None = None

    # ----------------------- Init / validation ------------------------- #
    def __post_init__(self):
        # Handle backward compatibility: if 'kind' is set but 'type' is not, copy it
        if self.kind and not self.type:
            self.type = self.kind
        # Ensure parameters is properly initialized for backward compatibility
        if not isinstance(self.parameters, list):
            self.parameters = []
        fixed_params: list[ParameterSpec] = []
        for p in self.parameters:
            if isinstance(p, dict):
                p = ParameterSpec(**p)
            elif not isinstance(p, ParameterSpec):
                logger.warning(f"Expected ParameterSpec or dict, got {type(p)}: {p}")
                continue
            fixed_params.append(p)
        self.parameters = fixed_params

    # ------------------------ Merge logic ------------------------------ #
    def update(self, problem_id: str, annotation: dict) -> None:
        """
        Update concept with new annotation.
        Note: This is now mainly for adding task_id to used_in.
        LLM-based merging happens before concepts are added to memory.
        """
        if problem_id not in self.used_in:
            self.used_in.append(problem_id)

        # Also handle used_in list from annotation (for merged concepts)
        if "used_in" in annotation:
            for task_id in annotation["used_in"]:
                if task_id not in self.used_in:
                    self.used_in.append(task_id)

        # Merge cues and implementation (mainly for initial creation)
        if "cues" in annotation and annotation["cues"]:
            self.cues = self._merge_lines(self.cues, annotation["cues"])
        if "implementation" in annotation and annotation["implementation"]:
            self.implementation = self._merge_lines(
                self.implementation, annotation["implementation"]
            )

    @staticmethod
    def _merge_lines(curr: list[str], new_lines: list[str]) -> list[str]:
        # merge cues & implementation (dedupe, keep order)
        cleaned_new_lines = []
        for line in new_lines:
            if isinstance(line, dict):
                if len(line) == 1:
                    k, v = next(iter(line.items()))
                    line = f"{k}: {v}"
                else:
                    logger.info(
                        f"merge list[str] expects a string but received a dict with multiple keys: {line}"
                    )
                    line = str(line)
            if isinstance(line, str):
                cleaned_new_lines.append(line.strip())
            else:
                logger.info(f"merge list[str] expects a string but received: {line}")

        return list(dict.fromkeys(itertools.chain(curr, cleaned_new_lines)))

    # --------------------- Rendering helpers --------------------------- #
    def to_string(
        self,
        *,
        indentation: int = 0,
        skip_type: bool = False,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        # Legacy parameters for backward compatibility
        include_description: bool = True,
        skip_kind: bool = False,
        skip_routine_subtype: bool = False,
        skip_parameters: bool = False,
        skip_parameter_description: bool = False,
    ) -> str:
        """
        Pretty-print this concept as a YAML-ish block.
        Only renders: concept name, type, cues, implementation
        """
        ind = " " * indentation
        lines: list[str] = [f"{ind}- concept: {self.name}"]

        if not skip_type:
            lines.append(f"{ind}  type: {self.type}")

        # Cues & implementation ----------------------------------------
        if self.cues and not skip_cues:
            lines.append(f"{ind}  cues:")
            for c in self.cues:
                lines.append(f"{ind}    - {c}")

        if self.implementation and not skip_implementation:
            lines.append(f"{ind}  implementation:")
            for note in self.implementation:
                lines.append(f"{ind}    - {note}")

        return "\n".join(lines)

    # -------------------------- Misc ----------------------------------- #
    def asdict(self) -> dict:
        return asdict(self)
    
    


# --------------------------------------------------------------------- #
#                               Dataclasses                             #
# --------------------------------------------------------------------- #
@dataclass
class ProblemSolution:
    problem_id: str
    solution: str | None = None
    summary: str | None = None
    pseudocode: str | None = None


# --------------------------------------------------------------------- #
#                            ConceptMemory                              #
# --------------------------------------------------------------------- #
class ConceptMemory:
    """
    Stores concepts, solutions, and custom type defs.
    """

    # Valid concept types for GUI agents
    VALID_TYPES = {
        "ui_navigation",
        "data_entry",
        "search_filter",
        "authentication",
        "verification",
        "data_extraction",
        "form_handling",
        "selection",
    }
    
    # Legacy types for backward compatibility
    LEGACY_TYPES = {"structure", "routine"}

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.categories: dict[str, list[str]] = defaultdict(list)  # kind → names
        self.solutions: dict[str, ProblemSolution] = {}
        self.custom_types: dict[str, str] = {}  # typedef name → annotation

    # ----------------------------------------------------------------- #
    #                              Ingestion                            #
    # ----------------------------------------------------------------- #
    def write_concept(self, puzzle_id: str, ann: dict) -> None:
        name = ann.get("concept") or ann.get("name")
        if not name:
            logger.info(f"[{puzzle_id}] Skipping concept: missing 'concept' field.")
            return
        concept_exists = name in self.concepts

        # Get type (with backward compatibility for 'kind')
        concept_type = ann.get("type") or ann.get("kind")
        
        # Validate type
        if not concept_exists:
            if concept_type not in self.VALID_TYPES and concept_type not in self.LEGACY_TYPES:
                logger.info(f"[{puzzle_id}] Concept '{name}' invalid type '{concept_type}'. Valid types: {self.VALID_TYPES}")
                return

        if concept_exists:
            self.concepts[name].update(puzzle_id, ann)
        else:
            c = Concept(
                name=name,
                type=concept_type,
            )
            c.update(puzzle_id, ann)
            self.concepts[name] = c
            self.categories[concept_type].append(name)

    def write_solution(self, puzzle_id: str, solution: str | None, ann: dict) -> None:
        self.solutions[puzzle_id] = ProblemSolution(
            problem_id=puzzle_id,
            solution=solution,
            summary=ann.get("summary"),
            pseudocode=ann.get("pseudocode"),
        )


    # ----------------------------------------------------------------- #
    #                              Rendering                            #
    # ----------------------------------------------------------------- #
    def to_string(
        self,
        *,
        concept_names: list[str] | None = None,
        skip_type: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        # usage-condensed rendering
        usage_threshold: int = 2,
        show_other_concepts: bool = False,
        indentation: int = 0,
        filter_concept: Callable[[Concept], bool] | None = None,
        # Legacy parameters for backward compatibility
        include_description: bool = True,
        skip_kind: bool = True,
        skip_routine_subtype: bool = True,
        skip_parameters: bool = False,
        skip_parameter_description: bool = True,
    ) -> str:
        """
        Render memory grouped by concept type.
        If `len(concept.used_in) < usage_threshold` the concept is listed only by name.
        """
        whitelist = set(concept_names) if concept_names else None
        blocks: list[str] = []

        # Group concepts by type
        type_order = [
            "ui_navigation",
            "authentication",
            "search_filter",
            "data_entry",
            "form_handling",
            "selection",
            "data_extraction",
            "verification",
        ]
        
        # Add any other types that exist but aren't in the standard list
        all_types = set(self.categories.keys())
        for t in all_types:
            if t not in type_order:
                type_order.append(t)
        
        for concept_type in type_order:
            if concept_type not in self.categories:
                continue
                
            blk = self._to_string_type_section(
                concept_type=concept_type,
                whitelist=whitelist,
                skip_type=skip_type,
                skip_cues=skip_cues,
                skip_implementation=skip_implementation,
                usage_threshold=usage_threshold,
                show_other_concepts=show_other_concepts,
                indentation=indentation,
                filter_concept=filter_concept,
            )
            if blk:
                blocks.append(blk)

        return "\n\n".join(blocks).rstrip()

    # --------------------------- Sections ----------------------------- #
    def _to_string_type_section(
        self,
        *,
        concept_type: str,
        whitelist: set | None,
        skip_type: bool,
        skip_cues: bool,
        skip_implementation: bool,
        usage_threshold: int,
        show_other_concepts: bool,
        indentation: int,
        filter_concept: Callable[[Concept], bool] | None,
    ) -> str:
        """Render a section for a specific concept type."""
        all_names = self.categories.get(concept_type, [])
        names_in = [n for n in all_names if (not whitelist or n in whitelist)]
        names_out = (
            [n for n in all_names if whitelist and n not in whitelist]
            if show_other_concepts
            else []
        )
        if not names_in and not names_out:
            return ""

        full_render: list[str] = []
        low_usage_names: list[str] = []

        for n in names_in:
            c = self.concepts[n]
            if filter_concept and not filter_concept(c):
                continue
            if len(c.used_in) < usage_threshold:
                low_usage_names.append(c.name)
            else:
                full_render.append(
                    c.to_string(
                        indentation=indentation,
                        skip_type=skip_type,
                        skip_cues=skip_cues,
                        skip_implementation=skip_implementation,
                    )
                )

        # Format section header with underscores replaced by spaces
        section_title = concept_type.replace("_", " ")
        lines: list[str] = [f"## {section_title}", *full_render]
        if low_usage_names:
            lines.append(
                f"- lower usage concepts: [{', '.join(sorted(low_usage_names))}]"
            )
        if names_out:
            lines.append(f"- other concepts: [{', '.join(sorted(names_out))}]")
        return "\n".join(lines)

    # ----------------------------------------------------------------- #
    #                   Model-output ingest & initialization             #
    # ----------------------------------------------------------------- #
    def update_from_model_output(self, puzzle_id: str, llm_output: str) -> None:
        yaml_block = extract_yaml_block(llm_output)
        try:
            parsed = yaml.safe_load(yaml_block)
        except yaml.YAMLError as e:
            logger.error(f"[{puzzle_id}] YAML parse error: {e}")
            return

        if not isinstance(parsed, list):
            logger.info(
                f"[{puzzle_id}] Expected a list of concepts, got {type(parsed)}."
            )
            return

        for concept_anno in parsed:
            self.write_concept(puzzle_id, concept_anno)

    def initialize_solutions(self, mapping: dict[str, dict]) -> None:
        for pid, ann in mapping.items():
            self.solutions[pid] = ProblemSolution(
                problem_id=pid,
                solution=ann.get("solution"),
                summary=ann.get("summary"),
                pseudocode=ann.get("pseudocode"),
            )

    def initialize_from_annotations(self, annotations: dict[str, dict]) -> None:
        for pid, ann in annotations.items():
            self.write_solution(pid, None, ann)
            for concept_ann in ann.get("concepts", []):
                self.write_concept(pid, concept_ann)

    # ----------------------------------------------------------------- #
    #                           Persistence                              #
    # ----------------------------------------------------------------- #
    def save_to_file(self, path: Path) -> None:
        blob = {
            "concepts": {n: asdict(c) for n, c in self.concepts.items()},
            "solutions": {pid: asdict(s) for pid, s in self.solutions.items()},
            "custom_types": self.custom_types,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(blob, f, indent=2)

    def load_from_file(self, path: Path) -> None:
        with open(path, 'r') as f:
            data = json.load(f)

        self.concepts = {n: Concept(**c) for n, c in data["concepts"].items()}
        self.solutions = {
            pid: ProblemSolution(**s) for pid, s in data["solutions"].items()
        }
        self.custom_types = data.get("custom_types", {})

        self.categories.clear()
        for name, concept in self.concepts.items():
            # Use 'type' field, fallback to 'kind' for backward compatibility
            concept_type = concept.type if hasattr(concept, 'type') and concept.type else concept.kind
            self.categories[concept_type].append(name)

