"""
Constructor module: Build Domain, Trajectory, and PhaseNote objects from raw trajectory data.

Responsibilities:
- Parse raw trajectory JSON/JSONL files
- Call VLM for phase segmentation (success/failure aware)
- Construct hierarchical memory objects (Domain → Trajectory → PhaseNote)
- Extract keyframes and save to disk

This module does NOT handle encoding or indexing—those are delegated to
phase_encoder.py and store.py respectively.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import uuid
from dataclasses import field
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image

from .schema import Domain, Trajectory, PhaseNote, PhaseNeighbor

from openai import OpenAI  # OpenAI-compatible client (works with vLLM api_server)


# =============================================================================
# Prompt Configuration
# =============================================================================

PROMPT_FILES = {
    "success": "success_phase_segmentation.txt",
    "failure": "failure_phase_segmentation.txt",
}


# =============================================================================
# Image Utilities
# =============================================================================

def _safe_b64_to_image(b64: str) -> Optional[Image.Image]:
    """Decode base64 string to PIL Image."""
    try:
        if b64.startswith("data:image"):
            b64 = b64.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        return None


def _save_image(img: Image.Image, out_path: str) -> None:
    """Save PIL Image to disk."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format="PNG")


def _extract_round_images(round_obj: Dict) -> List[str]:
    """Extract base64 image strings from a round's messages."""
    imgs: List[str] = []
    for msg in round_obj.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url")
                    if isinstance(url, str) and url.startswith("data:image"):
                        imgs.append(url)
    return imgs


# =============================================================================
# Text Extraction Utilities
# =============================================================================

def _extract_step_summary(round_obj: Dict) -> str:
    """Extract a short textual description from a round."""
    texts: List[str] = []
    for msg in round_obj.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    txt = item.get("text", "").strip()
                    if txt:
                        texts.append(txt)
    if texts:
        return texts[0][:500]
    return ""


def _extract_tools_used(rounds: List[Dict]) -> List[str]:
    """Extract unique tool names used across all rounds."""
    tools = set()
    for rnd in rounds:
        resp = rnd.get("response", "")
        # Simple heuristic: look for tool patterns in response
        if "click" in resp.lower():
            tools.add("click")
        if "type" in resp.lower():
            tools.add("type")
        if "scroll" in resp.lower():
            tools.add("scroll")
        if "wait" in resp.lower():
            tools.add("wait")
        if "press" in resp.lower():
            tools.add("press_key")
    return list(tools)


# =============================================================================
# VLM Segmentation
# =============================================================================

def _load_prompt(prompt_dir: str, success: bool) -> str:
    """Load the appropriate segmentation prompt."""
    fname = PROMPT_FILES["success" if success else "failure"]
    path = os.path.join(prompt_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r") as f:
        return f.read()


def _build_user_payload(
    trajectory: Dict,
    success: bool,
    existing_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build the user payload for VLM segmentation.
    
    If existing_domains is provided, they will be listed so the VLM can
    prefer selecting one of them before creating a new domain label.
    """
    task_desc = trajectory.get("task_description", "")
    total_rounds = len(trajectory.get("rounds", []))
    conversation_id = trajectory.get("conversation_id", trajectory.get("session_id", "unknown"))
    conversation_start = trajectory.get("conversation_start", "")
    conversation_end = trajectory.get("conversation_end", "")
    evaluation = trajectory.get("evaluation", {}).get("evaluation", {})
    first_error_step = evaluation.get("First_Error_Step")
    error_type = evaluation.get("Error_Type")
    correct_action = evaluation.get("Correct_Action")

    header = [f"TASK: {task_desc}", ""]

    # Optionally provide existing known domains as guidance
    if existing_domains:
        # Deduplicate and sort for stability
        uniq_domains = sorted(set(d for d in existing_domains if d))
        if uniq_domains:
            header.extend([
                "EXISTING_DOMAINS:",
                ", ".join(uniq_domains),
                "",
                "You MUST first try to choose ONE of the EXISTING_DOMAINS as the domain.",
                "Only create a NEW short domain label if none of them fit well.",
                "",
            ])
    if not success:
        header.extend([
            "EVALUATION_HINTS:",
            "Correctness: false",
            f"- First_Error_Step: {first_error_step}",
            f"- Error_Type: {error_type}",
            f"- Correct_Action: {correct_action}",
            "",
        ])

    header.extend([
        "TRAJECTORY:",
        f"- total_rounds: {total_rounds}",
        f"- conversation_id: {conversation_id}",
        f"- timestamps: {conversation_start} -> {conversation_end}",
        "",
        "STEPS:",
    ])
    parts: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(header)}]

    rounds = trajectory.get("rounds", [])
    for idx, rnd in enumerate(rounds):
        resp = rnd.get("response", "").strip().replace("\n", " ")
        page_desc = _extract_step_summary(rnd)
        page_desc = page_desc.replace("\n", " ") if page_desc else ""

        parts.append({
            "type": "text",
            "text": (
                f"Step {idx}:\n"
                f"  page_desc: {page_desc}\n"
                f"  screenshot: (attached below)\n"
                f"  agent_response: (provided after the screenshot)"
            ),
        })
        imgs = _extract_round_images(rnd)
        if not imgs:
            raise ValueError(f"No screenshot image found for step={idx}")
        # Attach only the first screenshot for this step to control payload size.
        parts.append({
            "type": "image_url",
            "image_url": {"url": imgs[0]},
        })
        parts.append({
            "type": "text",
            "text": f"Step {idx} agent_response:\n  {resp}",
        })

    return parts


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_response(text: str) -> Dict:
    """Parse JSON from VLM response."""
    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in VLM response.")
    return json.loads(match.group(0))


def _call_vlm(
    prompt: str,
    payload: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0.2,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Call VLM for segmentation."""
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "VLM API key is required by the OpenAI-compatible client. "
            "For vLLM's OpenAI API server you can use any non-empty value.\n"
            "Fix: pass --vlm_api_key 'EMPTY' or set OPENAI_API_KEY=EMPTY."
        )
    client = OpenAI(
        base_url=base_url,
        api_key=resolved_key,
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return response.choices[0].message.content


def segment_trajectory_with_vlm(
    trajectory: Dict,
    prompt_dir: str,
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    existing_domains: Optional[List[str]] = None,
) -> Dict:
    """
    Segment a trajectory into phases using VLM.
    
    Returns:
        Dict with keys: domain, trajectory_summary, phases
    """
    evaluation = trajectory.get("evaluation", {}).get("evaluation", {})
    success = bool(evaluation.get("Correctness", False))
    prompt = _load_prompt(prompt_dir, success=success)
    payload = _build_user_payload(trajectory, success=success, existing_domains=existing_domains)
    response_text = _call_vlm(
        prompt,
        payload,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )
    return _parse_json_response(response_text)


# =============================================================================
# Domain Constructor
# =============================================================================

class DomainConstructor:
    """Construct and manage Domain objects."""
    
    def __init__(self):
        # We key by the human-readable domain label returned by the VLM,
        # but each Domain has a database-like unique id.
        self.domains_by_label: Dict[str, Domain] = {}

    def get_or_create(
        self,
        domain_name: str,
    ) -> Domain:
        """Get existing domain or create new one."""
        label = (domain_name or "").strip().lower()
        if not label:
            raise ValueError("domain_name must be a non-empty string")

        if label not in self.domains_by_label:
            self.domains_by_label[label] = Domain(
                id=uuid.uuid4().hex,
                domain=label,
            )

        return self.domains_by_label[label]
    
    def get_all(self) -> List[Domain]:
        """Return all constructed domains."""
        return list(self.domains_by_label.values())


# =============================================================================
# Trajectory Constructor
# =============================================================================

class TrajectoryConstructor:
    """Construct Trajectory objects from raw JSON data."""
    
    def __init__(self, domain_constructor: DomainConstructor):
        self.domain_constructor = domain_constructor
        self.trajectories: Dict[str, Trajectory] = {}
    
    def construct_from_json(
        self,
        traj_data: Dict,
        file_path: str,
        domain_name: str,
        output_dir: Optional[str] = None, # Needed to save keyframes
    ) -> Trajectory:
        """
        Construct a Trajectory object from raw JSON data.
        
        Args:
            traj_data: Raw trajectory JSON
            file_path: Source file path
            domain_name: Inferred domain name (e.g., "shopping")
            output_dir: Output directory to save keyframes
        
        Returns:
            Trajectory object (without phases yet)
        """
        # Get or create domain
        domain = self.domain_constructor.get_or_create(
            domain_name=domain_name,
        )

        # Database-like unique trajectory id
        trajectory_id = uuid.uuid4().hex
        
        rounds = traj_data.get("rounds", [])
        evaluation = traj_data.get("evaluation", {}).get("evaluation", {})
        correctness = evaluation.get("Correctness", False)
        
        # Determine outcome
        outcome = "success" if correctness else "failure"
        
        # Extract first-frame image for trajectory embedding.
        # We no longer store an end-frame; trajectory embeddings are defined
        # as (task text + first-frame image) only.
        keyframe_paths = []
        if output_dir and rounds:
            # Ensure keyframes dir exists
            kf_dir = os.path.join(output_dir, "keyframes")
            os.makedirs(kf_dir, exist_ok=True)
            
            # First image (from first round)
            start_imgs = _extract_round_images(rounds[0])
            if start_imgs:
                img = _safe_b64_to_image(start_imgs[0])
                if img:
                    path = os.path.join(kf_dir, f"traj_{trajectory_id}_start.png")
                    _save_image(img, path)
                    keyframe_paths.append(path)

        # Construct Trajectory object
        trajectory = Trajectory(
            id=trajectory_id,
            domain=domain.domain,
            domain_id=domain.id,
            task_description=traj_data.get("task_description", ""),
            outcome=outcome,
            summary=None,  # Will be set after phase segmentation
            phase_note_ids=[],  # Will be populated after phase construction
            steps_pointer=file_path,
            keyframe_paths=keyframe_paths if keyframe_paths else None
        )
        trajectory.set_source_hash_from_file()
        
        self.trajectories[trajectory_id] = trajectory
        return trajectory
    
    def get_all(self) -> List[Trajectory]:
        """Return all constructed trajectories."""
        return list(self.trajectories.values())


# =============================================================================
# PhaseNote Constructor
# =============================================================================

class PhaseNoteConstructor:
    """Construct PhaseNote objects from VLM segmentation output."""
    
    def __init__(self):
        self.phases: Dict[str, PhaseNote] = {}
    
    def construct_phases_for_trajectory(
        self,
        trajectory: Trajectory,
        traj_data: Dict,
        vlm_segments: Dict,
        output_dir: str,
    ) -> List[PhaseNote]:
        """
        Construct PhaseNote objects for a trajectory.
        
        Args:
            trajectory: Parent Trajectory object
            traj_data: Raw trajectory JSON (for extracting images)
            vlm_segments: VLM segmentation output
            output_dir: Directory to save keyframes
        
        Returns:
            List of PhaseNote objects
        """
        rounds = traj_data.get("rounds", [])
        
        phases_data = vlm_segments.get("phases", [])
        
        # Construct PhaseNote objects from VLM segmentation output.
        # Phases are treated as an ordered list defined by start_step/end_step.
        constructed_phases: List[PhaseNote] = []
        
        for idx, phase_data in enumerate(phases_data):
            # Database-like unique phase id (do NOT trust the VLM to generate unique ids)
            phase_id = uuid.uuid4().hex
            
            # Extract phase metadata
            start_step = phase_data.get("start_step", 0)
            end_step = phase_data.get("end_step", start_step)
            phase_label = phase_data.get("phase_label", "browse")
            summary = phase_data.get("summary", "")
            title = phase_data.get("title", "")
            ui_cues = phase_data.get("ui_cues", [])
            
            # Build combined summary
            combined_summary = f"{title}: {summary}" if title else summary
            if ui_cues:
                cues_text = ", ".join(ui_cues)
                combined_summary += f" | cues: {cues_text}"
            
            # Extract and save keyframes
            keyframe_indices = phase_data.get("keyframe_indices") or [start_step, end_step]
            keyframe_paths: List[str] = []
            for ki, step_idx in enumerate(keyframe_indices[:2]):
                if step_idx < 0 or step_idx >= len(rounds):
                    continue
                imgs = _extract_round_images(rounds[step_idx])
                if imgs:
                    img = _safe_b64_to_image(imgs[0])
                    if img is not None:
                        kp = os.path.join(output_dir, "keyframes", f"{phase_id}_{ki}.png")
                        _save_image(img, kp)
                        keyframe_paths.append(kp)
            
            # Construct PhaseNote
            phase_note = PhaseNote(
                id=phase_id,
                domain=trajectory.domain,
                trajectory_id=trajectory.id,
                phase_label=phase_label,
                start_step=start_step,
                end_step=end_step,
                summary=combined_summary,
                keyframe_indices=keyframe_indices[:2],
                keyframe_paths=keyframe_paths,
                source_path=trajectory.steps_pointer,
            )

            self.phases[phase_id] = phase_note
            constructed_phases.append(phase_note)
        
        # Update trajectory with phase IDs
        trajectory.phase_note_ids = [p.id for p in constructed_phases]
        
        # Trajectory-level summary (VLM-provided).
        # This should explain WHY the task succeeded/failed in a concise, sharp way.
        traj_summary = vlm_segments.get("trajectory_summary")
        if not isinstance(traj_summary, str) or not traj_summary.strip():
            raise ValueError(
                f"[PhaseNoteConstructor] VLM did not produce a valid "
                f"'trajectory_summary' for trajectory_id={trajectory.id}"
            )
        trajectory.summary = traj_summary.strip()
        
        return constructed_phases
    
    def get_all(self) -> List[PhaseNote]:
        """Return all constructed phases."""
        return list(self.phases.values())


# =============================================================================
# Unified Memory Constructor
# =============================================================================

class MemoryConstructor:
    """
    Unified constructor for hierarchical memory (Domain → Trajectory → PhaseNote).
    
    Usage:
        constructor = MemoryConstructor(output_dir="hybrid_index/webvoyager")
        
        for file_path in trajectory_files:
            # domain_name is now just a fallback/legacy hint;
            # the actual domain is inferred by the VLM from the trajectory content.
            constructor.process_trajectory_file(
                file_path=file_path,
                domain_name="general",
            )
        
        # Get all constructed objects
        domains = constructor.get_domains()
        trajectories = constructor.get_trajectories()
        phases = constructor.get_phases()
    """
    
    def __init__(
        self,
        output_dir: str,
        prompt_dir: str = "hybrid_memory/prompts",
        vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_base_url: Optional[str] = "http://localhost:8000/v1",
        vlm_api_key: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.prompt_dir = prompt_dir
        self.vlm_model = vlm_model
        self.vlm_base_url = vlm_base_url
        self.vlm_api_key = vlm_api_key
        
        # Initialize sub-constructors
        self.domain_constructor = DomainConstructor()
        self.trajectory_constructor = TrajectoryConstructor(self.domain_constructor)
        self.phase_constructor = PhaseNoteConstructor()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def process_trajectory_file(
        self,
        file_path: str,
    ) -> Tuple[Trajectory, List[PhaseNote]]:
        """
        Process a single trajectory file and construct all memory objects.
        
        Args:
            file_path: Path to trajectory JSON/JSONL file
            domain_name: (legacy) domain hint from CLI/config; NOT used as final label
            site_host: (legacy) ignored
        
        Returns:
            Tuple of (Trajectory, List[PhaseNote])
        """
        def _load_json_file(path: str) -> Dict[str, Any]:
            with open(path, "r") as f:
                return json.load(f)

        # Load trajectory data (fail fast)
        traj_data = _load_json_file(file_path)

        # If this is a failure trajectory split into positive/negative parts,
        # merge them into a single trajectory before sending to the VLM.
        pos_token = f"{os.sep}positive{os.sep}"
        neg_token = f"{os.sep}negative{os.sep}"
        if pos_token in file_path or neg_token in file_path:
            if neg_token in file_path:
                pos_path = file_path.replace(neg_token, pos_token)
                neg_path = file_path
            else:
                pos_path = file_path
                neg_path = file_path.replace(pos_token, neg_token)

            if not os.path.exists(pos_path):
                raise FileNotFoundError(f"Positive-part file not found: {pos_path}")
            if not os.path.exists(neg_path):
                raise FileNotFoundError(f"Negative-part file not found: {neg_path}")

            pos_data = traj_data if file_path == pos_path else _load_json_file(pos_path)
            neg_data = traj_data if file_path == neg_path else _load_json_file(neg_path)

            pos_cid = pos_data.get("conversation_id")
            neg_cid = neg_data.get("conversation_id")
            if not isinstance(pos_cid, str) or not pos_cid.strip():
                raise ValueError(f"Missing conversation_id in positive-part: {pos_path}")
            if pos_cid != neg_cid:
                raise ValueError(
                    "Mismatched conversation_id between positive/negative parts: "
                    f"{pos_cid!r} vs {neg_cid!r}"
                )

            if pos_data.get("split_type") != "positive_part":
                raise ValueError(f"Expected split_type='positive_part' in {pos_path}")
            if neg_data.get("split_type") != "negative_part":
                raise ValueError(f"Expected split_type='negative_part' in {neg_path}")

            pos_task = pos_data.get("task_description")
            neg_task = neg_data.get("task_description")
            if pos_task != neg_task:
                raise ValueError("Mismatched task_description between positive/negative parts")

            pos_eval = pos_data.get("evaluation", {}).get("evaluation", {})
            neg_eval = neg_data.get("evaluation", {}).get("evaluation", {})
            if bool(pos_eval.get("Correctness", False)) or bool(neg_eval.get("Correctness", False)):
                raise ValueError("positive/negative parts must be failure trajectories (Correctness=false)")

            # Merge rounds (positive first, then negative)
            pos_rounds = pos_data.get("rounds")
            neg_rounds = neg_data.get("rounds")
            if not isinstance(pos_rounds, list) or not isinstance(neg_rounds, list):
                raise ValueError("Both positive and negative parts must contain a 'rounds' list")

            merged_rounds = pos_rounds + neg_rounds
            if not merged_rounds:
                raise ValueError("Merged failure trajectory has no rounds")

            # Merge metadata (keep positive start, negative end)
            merged = dict(pos_data)
            merged["split_type"] = "merged_failure"
            merged["rounds"] = merged_rounds
            merged["total_rounds"] = len(merged_rounds)
            merged["conversation_start"] = pos_data.get("conversation_start")
            merged["conversation_end"] = neg_data.get("conversation_end")
            if not merged.get("conversation_start"):
                raise ValueError("Missing conversation_start in merged failure trajectory")
            if not merged.get("conversation_end"):
                raise ValueError("Missing conversation_end in merged failure trajectory")

            traj_data = merged
            # Use the positive-part file as the canonical pointer for provenance.
            file_path = pos_path
        
        rounds = traj_data.get("rounds", [])
        if not rounds:
            raise ValueError(f"No rounds found in trajectory: {file_path}")
        
        # Step 1: Segment with VLM (also infers high-level domain)
        # Collect existing domains inferred so far in this constructor (if any)
        existing_domains = [d.domain for d in self.domain_constructor.get_all() if d.domain]
        vlm_segments = segment_trajectory_with_vlm(
            trajectory=traj_data,
            prompt_dir=self.prompt_dir,
            model_name=self.vlm_model,
            base_url=self.vlm_base_url,
            api_key=self.vlm_api_key,
            existing_domains=existing_domains or None,
        )
        # Domain is treated as single source of truth; if missing, skip this trajectory
        inferred_domain = vlm_segments.get("domain", None)
        if not isinstance(inferred_domain, str) or not inferred_domain.strip():
            raise ValueError(f"[MemoryConstructor] VLM did not produce a valid 'domain' for {file_path}")
        inferred_domain = inferred_domain.strip()
        
        # Step 2: Construct Trajectory with inferred domain
        trajectory = self.trajectory_constructor.construct_from_json(
            traj_data=traj_data,
            file_path=file_path,
            domain_name=inferred_domain,
            output_dir=self.output_dir, # Pass output_dir to save keyframes
        )
        
        # Step 3: Construct PhaseNotes
        phases = self.phase_constructor.construct_phases_for_trajectory(
            trajectory=trajectory,
            traj_data=traj_data,
            vlm_segments=vlm_segments,
            output_dir=self.output_dir,
        )
        
        return trajectory, phases
    
    def get_domains(self) -> List[Domain]:
        """Return all constructed domains."""
        return self.domain_constructor.get_all()
    
    def get_trajectories(self) -> List[Trajectory]:
        """Return all constructed trajectories."""
        return self.trajectory_constructor.get_all()
    
    def get_phases(self) -> List[PhaseNote]:
        """Return all constructed phases."""
        return self.phase_constructor.get_all()
