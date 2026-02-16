"""Function Call Agent for GUI Agent using direct model calls with ReAct paradigm"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import base64
from PIL import Image
from io import BytesIO
import sys
import numpy as np
# import pathlib
# path = pathlib.Path(__file__).parent.parent.parent.parent
# sys.path.append(str(path/'WebDreamer'))
# from simulation_scoring import evaluate_simulation

from browser_env import Trajectory, Action
from browser_env.actions import ActionTypes
from actions import (
    create_click_action,
    create_selection_action,
    create_type_action,
    create_scroll_action,
    create_wait_action,
    create_stop_action,
    create_none_action,
    create_key_press_action,
    create_goto_url_action,
    create_go_back_action,
    parse_action_json,
    # validate_action
)
from .llm_config import create_direct_model, load_tool_llm
from tools.gui_tools import ClickTool, TypeTool, ScrollTool, WaitTool, StopTool, PressKeyTool, PageGotoTool, SelectionTool, GoBackTool
from tools.analysis_tools import MapSearchTool, ContentAnalyzerTool
from tools.web_search_tools import WebSearchTool

from arc_memo.concept_mem.concept import ConceptMemory
from arc_memo.concept_mem.select_simple import ConceptSelector, render_concepts_as_suggestions
from memory.reasoning_bank import ReasoningBank
from hybrid_memory.retriever import HybridRetriever


def resize_image_base64(base64_string: str) -> str:
    """Simple image resize to reduce token count"""
    try:
        from PIL import Image
        import io
        
        # Decode and resize
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Simple resize to 400x300
        image = image.resize((400, 300), Image.Resampling.LANCZOS)
        
        # Save as JPEG with low quality
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=50)
        output.seek(0)
        
        # Return compressed base64
        return base64.b64encode(output.getvalue()).decode('utf-8')
        
    except:
        return base64_string


class FunctionCallAgent:
    """Custom function call agent for GUI interactions using ReAct paradigm with direct model calls"""
    
    def __init__(self, args: argparse.Namespace, **kwargs):
        """Initialize the function call agent"""
        # Initialize direct model
        self.llm = create_direct_model(args)
        # Initialize tool LLM for tools that need it
        self.tool_llm = load_tool_llm(args)
        
        # Flag to use checkpoint model for discrete memory operations
        self.discrete_memory_use_checkpoint = getattr(args, 'discrete_memory_use_checkpoint', False)
        
        # Define functions for the agent
        function_list = self._define_functions()
        # Build dynamic tool specs for prompt
        self.tool_specs = self._build_tool_specs(function_list)
    
        self.args = args
        self.logger = logging.getLogger("logger")
        self.current_step = 0
        self.discrete_memory_cache: Dict[str, str] = {}
        
        # Initialize function map for tools
        self.function_map = {}
        self._initialize_function_map()
        print('*'*50, 'function_map', '*'*50)
        print(self.function_map)
        print('*'*50, 'function_map', '*'*50)
        
        training_data_dir = getattr(args, 'training_data_dir', 'training_data')
        memory_data_dirs = getattr(args, 'memory_data_dir', ['training_data'])
        # Ensure memory_data_dirs is a list
        if isinstance(memory_data_dirs, str):
            memory_data_dirs = [memory_data_dirs]
        self.memory_data_dirs = memory_data_dirs
        # Initialize training data collector if enabled
        if hasattr(args, 'collect_training_data') and args.collect_training_data:
            from utils.training_data_collector import TrainingDataCollector, get_collector, set_collector
            from utils.llm_wrapper import wrap_llm
            self.training_collector = TrainingDataCollector(
                output_dir=training_data_dir,
                enabled=True
            )
            set_collector(self.training_collector)
            wrapped_llm = wrap_llm(self.llm)
            self.llm = wrapped_llm
        else:
            self.training_collector = None
        
        # Initialize discrete memory system if enabled
        if hasattr(args, 'use_discrete_memory') and args.use_discrete_memory:
            from memory.experience_memory import Memory
            # Determine if multimodal memory should be used
            multimodal = True
            # Check if there's a saved index path
            faiss_index_path = getattr(args, 'faiss_index_path', None)
            print(f"Initializing Discrete Memory system (multimodal: {multimodal})...")
            self.memory = Memory(training_data_path=self.memory_data_dirs[0], multimodal=multimodal, faiss_index_path=faiss_index_path, agent=self, bank_size=args.bank_size)
            print("Discrete Memory system initialized successfully")
            self.experience_memory = None
            self.experience_texts, self.experience_images, self.file_id_list = None, None, None
        else:
            self.memory = None
            
        # Initialize experience action suggestions if enabled
        if self.args.experience_action_suggestions:
            from action_scaling.trajectory_analyzer import TrajectoryAnalyzer
            self.trajectory_analyzer = TrajectoryAnalyzer(trajectory_dir=args.trajectory_dir, state_embedding_path=args.state_embedding_path, tool_llm=self.tool_llm)
            if args.state_embedding_path is None:
                self.trajectory_analyzer.compute_state_embeddings()
        else:
            self.trajectory_analyzer = None
            
        # Initialize concept memory if enabled
        if self.args.use_concept_memory:
            self.concept_memory = ConceptMemory()
            self.concept_memory.load_from_file(args.concept_memory_path)
            self.selector = ConceptSelector()
            self.selector.precompute_cue_embeddings(self.concept_memory)
        else:
            self.concept_memory, self.selector = None, None
        # Initialize world model if enabled
        self.world_model = None
        if getattr(args, 'use_world_model', False):
            try:
                from world_model import WorldModel, TrajectoryStore
                trajectory_store = TrajectoryStore(
                    training_data_path=getattr(args, 'world_model_data_path', 'training_data'),
                    faiss_index_path=getattr(args, 'world_model_index_path', None),
                    multimodal=getattr(args, 'world_model_multimodal', True)
                )
                self.world_model = WorldModel(
                    args=args,
                    trajectory_store=trajectory_store,
                    tool_llm=self.tool_llm
                )
                print("World Model initialized successfully")
                print(f"  Stats: {self.world_model.get_stats()}")
            except Exception as e:
                print(f"Failed to initialize World Model: {e}")
                self.world_model = None

        # Initialize reasoning bank (optional)
        self.reasoning_bank = None
        if hasattr(self.args, 'use_reasoning_bank') and self.args.use_reasoning_bank:
            try:
                use_mm = getattr(self.args, 'reasoning_bank_multimodal', False)
                bank_path = getattr(self.args, 'reasoning_bank_path', 'memory/reasoning_bank.jsonl')
                index_base = getattr(self.args, 'reasoning_index_base', 'memory_index/reasoning_bank_text')
                
                # Use multimodal paths if enabled
                if use_mm:
                    bank_path = getattr(self.args, 'reasoning_bank_path', 'memory/reasoning_bank_mm.jsonl')
                    index_base = getattr(self.args, 'reasoning_index_base', 'memory_index/reasoning_bank_mm')
                
                self.reasoning_bank = ReasoningBank(
                    bank_path=bank_path,
                    index_base_path=index_base,
                    use_multimodal=use_mm
                )
            except Exception as e:
                self.reasoning_bank = None
        
        # Store analysis results and map search context for next steps
        self.last_analysis_result = None
        self.last_map_search_query = None
        self.last_map_search_result = None
        self.last_page_goto_name = None
        self.last_page_goto_result = None
        self.last_web_search_result = None
        self.last_web_search_screenshots = None
        self.last_hybrid_exemplars = None
        # Initialize hybrid memory retriever (optional)
        self.hybrid_retriever = None
        try:
            if getattr(self.args, 'use_hybrid_memory', False):
                self.hybrid_retriever = HybridRetriever(getattr(self.args, 'hybrid_index_dir', 'hybrid_index'))
        except Exception as e:
            self.logger.warning(f"[HybridMemory] initialization failed: {e}")
        
        # Initialize graph memory system (optional)
        self.graph_memory = None
        self.graph_memory_retriever = None
        self.graph_memory_block = None  # Cached graph memory guidance
        if getattr(self.args, 'use_graph_memory', False):
            try:
                from graph_memory import TagExtractor, GraphBuilder, GraphMemoryRetriever
                from memory.help_functions import CLIPMultimodalSimilarity
                
                # Initialize tag extractor with tool LLM
                tag_cache_path = getattr(self.args, 'graph_tag_cache_path', 'graph_memory_cache/tags.json')
                tag_extractor = TagExtractor(llm=self.tool_llm, cache_path=tag_cache_path)
                
                # Initialize graph builder with LLM for VLM-based deduplication
                self.graph_memory = GraphBuilder(tag_extractor=tag_extractor, llm=self.tool_llm)
                
                # Load existing graph if path provided
                graph_index_path = getattr(self.args, 'graph_memory_index_path', None)
                if graph_index_path:
                    try:
                        self.graph_memory.load(graph_index_path)
                        self.logger.info(f"[GraphMemory] Loaded graph from {graph_index_path}")
                    except Exception as e:
                        self.logger.warning(f"[GraphMemory] Failed to load graph: {e}")
                
                # Initialize embedding model for retrieval (multimodal, same as discrete memory)
                embedding_model = CLIPMultimodalSimilarity()
                
                # Initialize retriever
                expand_hops = getattr(self.args, 'graph_expand_hops', 1)
                initial_seeds = getattr(self.args, 'graph_initial_seeds', None)
                self.graph_memory_retriever = GraphMemoryRetriever(
                    graph_builder=self.graph_memory,
                    embedding_model=embedding_model,
                    expand_hops=expand_hops,
                    initial_seeds=initial_seeds
                )
                
                self.logger.info(f"[GraphMemory] Initialized with {len(self.graph_memory)} trajectories")
            except Exception as e:
                self.logger.warning(f"[GraphMemory] initialization failed: {e}")
                self.graph_memory = None
                self.graph_memory_retriever = None
        
        # Dynamic memory update state (for use_dynamic_memory_update feature)
        self.current_raw_takeaways = None  # List[TaggedTrajectory] - for checkpoint comparison
        self.memory_update_count = 0        # int - track how many updates occurred
        self.clean_intent = None            # str - original intent without instructions/errors for retrieval
        
        if getattr(self.args, 'use_dynamic_memory_update', False):
            self.logger.info("[DynamicMemory] Dynamic memory update ENABLED")

    
    def _define_functions(self) -> List[str]:
        """Define the functions available to the agent"""
        # Use function names instead of tool instances
        functions = [
            'click',
            'type', 
            'press_key',
            'scroll',
            'wait',
            'go_back',
            'stop',
            'map_search',
            'content_analyzer',
            # 'goto_homepage',
            # 'goto_url',
            # 'google_web_search'
        ]
        return functions
    
    def _build_tool_specs(self, function_list: List[str]) -> List[Dict[str, Any]]:
        """Build tool specs (name, description, parameters) for prompt from function_list."""
        name_to_cls = {
            # Action tools
            'click': ClickTool,
            'selection': SelectionTool,
            'type': TypeTool,
            'scroll': ScrollTool,
            'wait': WaitTool,
            'stop': StopTool,
            'go_back': GoBackTool,
            'press_key': PressKeyTool,
            # Analysis tools
            'map_search': MapSearchTool,
            'content_analyzer': ContentAnalyzerTool,
            # 'goto_homepage': GotoHomepageTool,
            # 'goto_url': PageGotoTool,
            # 'google_web_search': WebSearchTool
            }
            
        specs: List[Dict[str, Any]] = []
        for item in function_list:
            cls = name_to_cls.get(str(item))
            tool = None
            if cls is not None:
                tool = cls()
            else:
                print(f"Tool {item} not found")
            if tool is not None:
                parameters = getattr(tool, 'parameters', None)
                if parameters is not None:
                    params_info = {}
                    parameters = parameters.get('properties', {})
                    for param_name, param_info in parameters.items():
                        params_info[param_name] = param_info
                    parameters['properties'] = params_info
                specs.append({
                    'name': getattr(tool, 'name', str(item)),
                    'description': getattr(tool, 'description', 'No description'),
                    'parameters': parameters
                })
    
        return specs
    
    def _initialize_function_map(self):
        """Initialize the function map with tool instances"""
        name_to_cls = {
            # Action tools
            'click': ClickTool,
            'selection': SelectionTool,
            'type': TypeTool,
            'scroll': ScrollTool,
            'wait': WaitTool,
            'stop': StopTool,
            'go_back': GoBackTool,
            'press_key': PressKeyTool,
            # Analysis tools
            'map_search': MapSearchTool,
            'content_analyzer': ContentAnalyzerTool,
            # 'goto_homepage': GotoHomepageTool,
            # 'goto_url': PageGotoTool,
            # 'google_web_search': WebSearchTool
        }
        
        for name, cls in name_to_cls.items():
            try:
                tool = cls()
                tool.llm = self.tool_llm  # Set the tool LLM
                self.function_map[name] = tool
            except Exception as e:
                print(f"Failed to initialize tool {name}: {e}")
    
    def _get_system_message(self, intent, trajectory, reflection=None, status_note=None) -> str:
        """Get the system message for the agent using ReAct paradigm"""
        tools_section = ""
        lines = []
        for spec in self.tool_specs:
            desc = spec.get('description') or ''
            name = spec.get('name')
            params = spec.get('parameters', {})
            
            # Build parameter description
            param_desc = ""
            if params and 'properties' in params:
                param_list = []
                for param_name, param_info in params['properties'].items():
                    param_type = param_info.get('type', 'string')
                    param_desc_text = param_info.get('description', '')
                    if 'enum' in param_info:
                        enum_values = ', '.join(param_info['enum'])
                        param_list.append(f"`{param_name}` ({param_type}: {enum_values})")
                    else:
                        param_list.append(f"`{param_name}` ({param_type}): {param_desc_text}")
                param_desc = f" - Parameters: {', '.join(param_list)}"
            
            lines.append(f"- **{name}**: {desc}{param_desc}")
        tools_section = "\n".join(lines)
        # Resolve prompt file paths relative to this file's location
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build discrete memory (only once per task - cached in self.experience_memory)
        if getattr(self.args, "use_discrete_memory", False) and self.memory is not None and self.experience_memory is None:
            first_image = self._get_first_screenshot(trajectory)
            print(f'[Discrete Memory] Retrieving similar trajectories with similar_num: {self.args.similar_num}')
            _, self.experience_texts, self.experience_images, self.file_id_list = self.memory.construct_experience_memory(
                intent, self, current_image=first_image,
                dataset=self.args.evaluation_type, domain=self.args.domain,
                similar_num=self.args.similar_num
            )
            self.experience_memory = self._build_discrete_memory_block(
                intent=intent,
                file_id_list=self.file_id_list,
                experience_actions=self.experience_texts,
                experience_images=self.experience_images,
                current_image=first_image,  # Pass current screenshot for digestion
            )
        elif self.experience_memory is None:
            examples_path = os.path.join(agent_dir, "prompts", "examples.txt")
            if os.path.exists(examples_path):
                with open(examples_path, 'r') as f:
                    self.experience_memory = f.read()
            else:
                # Fallback: empty examples if file doesn't exist
                self.experience_memory = ""
        
        # Build graph memory block if enabled (only once per task)
        if getattr(self.args, "use_graph_memory", False) and self.graph_memory_retriever is not None and self.graph_memory_block is None:
            try:
                first_image = self._get_first_screenshot(trajectory)
                # Use clean_intent for retrieval to avoid instruction/error pollution
                retrieval_intent = self.clean_intent if self.clean_intent else intent
                self.graph_memory_block = self._build_graph_memory_block(
                    intent=retrieval_intent,
                    current_image=first_image,
                )
                # Append graph memory to experience memory
                if self.graph_memory_block:
                    self.experience_memory = self.experience_memory + "\n\n" + self.graph_memory_block
                    self.logger.info(f"[GraphMemory] Added graph memory block to prompt")
            except Exception as e:
                self.logger.warning(f"[GraphMemory] Failed to build graph memory block: {e}")
                
        system_prompt_path = os.path.join(agent_dir, "prompts", "system_prompt.txt")
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, 'r') as f:
                system_prompt = f.read()
        else:
            # Fallback: basic system prompt if file doesn't exist
            system_prompt = """You are a helpful GUI automation agent. Use the available tools to complete tasks.
            
Available tools:
{tools_section}

{experience_memory}
"""
        system_prompt = system_prompt.format(experience_memory=self.experience_memory, tools_section=tools_section)

        # Inject reflection/status into system prompt if available
        if reflection or status_note:
             system_prompt += "\n\n*** SELF-REFLECTION & CORRECTION ***"
             if reflection:
                 system_prompt += f"\nThe following is an analysis of your recent performance. You must prioritize this feedback over your previous plan.\n\n{reflection}"
             # Only add hard constraint and system check when stuck is indicated (status_note present)
             if status_note:
                 system_prompt += "\n\nCONSTRAINT: If status is 'Stuck' or 'Regressing', you MUST change strategy. DO NOT REPEAT THE PREVIOUS ACTION."
                 system_prompt += f"\n\nSystem check: {status_note}"

        # Add World Model initial guidance (at step 0)
        if self.world_model is not None and self.current_step == 0:
            try:
                first_image = self._get_first_screenshot(trajectory)
                world_model_guidance = self.world_model.get_initial_guidance(
                    task=intent,
                    initial_screenshot=first_image,
                    domain=getattr(self.args, 'domain', None),
                    dataset=getattr(self.args, 'evaluation_type', None)  # e.g., 'mmina', 'webvoyager'
                )
                if world_model_guidance:
                    system_prompt = system_prompt + f"\n\n{world_model_guidance}"
                    self.logger.info("[WorldModel] Injected initial guidance into system prompt")
            except Exception as e:
                self.logger.warning(f"[WorldModel] Failed to get initial guidance: {e}")

        return system_prompt

    def _load_discrete_memory_cache(self) -> Dict[str, str]:
        cache_path = getattr(self.args, "discrete_memory_cache_path", None)
        if cache_path is None:
            raise ValueError("discrete_memory_cache_path must not be None when use_discrete_memory is enabled")
        cache_file = Path(cache_path)
        if not cache_file.exists():
            return {}
        payload = json.loads(cache_file.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid discrete memory cache format in {cache_path}: expected JSON object")
        summaries = payload.get("summaries")
        if summaries is None:
            raise ValueError(f"Invalid discrete memory cache format in {cache_path}: missing 'summaries'")
        if not isinstance(summaries, dict):
            raise ValueError(f"Invalid discrete memory cache format in {cache_path}: 'summaries' must be an object")
        for k, v in summaries.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(f"Invalid discrete memory cache entry in {cache_path}: keys/values must be strings")
        return summaries

    def _save_discrete_memory_cache(self, summaries: Dict[str, str]) -> None:
        cache_path = getattr(self.args, "discrete_memory_cache_path", None)
        if cache_path is None:
            raise ValueError("discrete_memory_cache_path must not be None when use_discrete_memory is enabled")
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summaries": summaries}
        cache_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def _format_actions_for_summary(self, actions: List[Dict], max_actions: int) -> str:
        if not isinstance(actions, list):
            raise ValueError(f"actions must be a list, got {type(actions)}")
        lines: List[str] = []
        for a in actions[:max_actions]:
            if not isinstance(a, dict):
                raise ValueError(f"action must be a dict, got {type(a)}")
            name = a.get("name")
            args = a.get("arguments")
            if not isinstance(name, str):
                raise ValueError("action['name'] must be a string")
            if not isinstance(args, dict):
                raise ValueError("action['arguments'] must be a dict")
            reasoning = args.get("reasoning", "")
            if not isinstance(reasoning, str):
                raise ValueError("action['arguments']['reasoning'] must be a string")
            lines.append(f"- {name}: {reasoning}")
        return "\n".join(lines)

    def _get_discrete_memory_llm(self):
        """Get the LLM to use for discrete memory summarization/digestion.
        
        If discrete_memory_use_checkpoint is True, use the main LLM (fine-tuned checkpoint).
        Otherwise, use tool_llm (base model via vLLM).
        """
        if self.discrete_memory_use_checkpoint:
            self.logger.info("[Discrete Memory] Using checkpoint model (main LLM) for summarization/digestion")
            return self.llm
        else:
            return self.tool_llm

    def _summarize_trajectory_with_vlm(
        self,
        task: str,
        actions_text: str,
        image_b64: Optional[str],
        experience_texts: Optional[List[List[Dict]]] = None,
        experience_images: Optional[List[List[str]]] = None,
        file_id_list: Optional[List[str]] = None,
    ) -> str:
        discrete_llm = self._get_discrete_memory_llm()
        if discrete_llm is None:
            raise ValueError("LLM is required for trajectory summary generation")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if not isinstance(actions_text, str) or not actions_text.strip():
            raise ValueError("actions_text must be a non-empty string")

        system = (
            "You extract actionable heuristics from SUCCESSFUL GUI agent trajectories.\n"
            "Return EXACTLY 1 sentence starting with 'takeaway:' in this format:\n"
            "takeaway: <ONE concise actionable heuristic>\n"
            "Constraints:\n"
            "- Focus on WHAT strategy worked (not what the agent did).\n"
            "- Must start with 'takeaway:' (exact substring).\n"
            "- Keep it under 20 words.\n"
            "- No quotes, no bullet points, no step-by-step narration, no coordinates/IDs."
        )
        user_text = f"Task: {task}\nActions:\n{actions_text}"
        if image_b64 is None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ]
        else:
            if not isinstance(image_b64, str) or not image_b64.startswith("data:image"):
                raise ValueError("image_b64 must be a data:image base64 URL when provided")
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ]},
            ]
        # Use checkpoint model with experience data (same as action generation)
        if self.discrete_memory_use_checkpoint:
            resp, _, _ = discrete_llm.chat(
                messages=messages, stream=False,
                experience_texts=experience_texts,
                experience_images=experience_images,
                file_id_list=file_id_list,
            )
        else:
            resp, _, _ = discrete_llm.chat(messages=messages, stream=False, temperature=0.0, max_tokens=128)
        if not hasattr(resp, "content"):
            raise ValueError("LLM response missing content")
        summary = resp.content.strip()
        if not summary:
            raise ValueError("Empty summary returned by LLM")
        if "\n" in summary:
            raise ValueError(f"Summary must be a single line, got: {summary!r}")
        if summary.lstrip().startswith(("-", "•", "\"")):
            raise ValueError(f"Summary must not start with bullets/quotes, got: {summary!r}")
        if not summary.lower().startswith("takeaway:"):
            raise ValueError(f"Summary must start with 'takeaway:', got: {summary!r}")
        self.logger.info(f"[Trajectory Summary] ({len(summary.split())} words): {summary}")
        return summary

    def _digest_discrete_memory(
        self,
        current_task: str,
        current_image: Optional[str],
        trajectory_summaries: List[str],
        experience_texts: Optional[List[List[Dict]]] = None,
        experience_images: Optional[List[List[str]]] = None,
        file_id_list: Optional[List[str]] = None,
    ) -> str:
        """
        Digest multiple trajectory summaries into a single, task-specific guidance.
        
        Instead of injecting all summaries directly, we ask the VLM to analyze them
        in the context of the current task and screenshot, producing focused guidance.
        """
        discrete_llm = self._get_discrete_memory_llm()
        if discrete_llm is None:
            raise ValueError("LLM is required for discrete memory digestion")
        if not trajectory_summaries:
            raise ValueError("trajectory_summaries cannot be empty")
        
        summaries_text = "\n".join(f"- {s}" for s in trajectory_summaries)
        
        system = (
            "You are an expert at analyzing past GUI agent experiences to help with a new task.\n"
            "Given the current task, current screenshot, and retrieved experience summaries,\n"
            "synthesize them into focused, actionable guidance.\n\n"
            "Output format: ONE concise paragraph (2-3 sentences) that answers:\n"
            "1. Which strategies from past experiences are MOST relevant to this specific task?\n"
            "2. What key actions or filters should be prioritized?\n\n"
            "IMPORTANT RULES:\n"
            "- Focus ONLY on navigation/search strategies, NOT on when to stop.\n"
            "- Do NOT mention stopping, completing, or finishing the task.\n"
            "- Do NOT give instructions about providing answers or explanations.\n"
            "- Be specific to the current task. Do NOT just repeat the summaries.\n"
            "- Do NOT use bullet points. Write as a coherent paragraph."
        )
        
        user_text = f"Current Task: {current_task}\n\nRetrieved Experience Summaries:\n{summaries_text}"
        
        if current_image is None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ]
        else:
            if not isinstance(current_image, str):
                raise ValueError("current_image must be a string when provided")
            # Handle both raw base64 and data URL formats
            if current_image.startswith("data:image"):
                image_url = current_image
            else:
                # Assume raw base64, add PNG data URL prefix
                image_url = f"data:image/png;base64,{current_image}"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ]
        
        self.logger.info("[Discrete Memory] Digesting summaries into task-specific guidance...")
        # Use checkpoint model with experience data (same as action generation)
        if self.discrete_memory_use_checkpoint:
            resp, _, _ = discrete_llm.chat(
                messages=messages, stream=False,
                experience_texts=experience_texts,
                experience_images=experience_images,
                file_id_list=file_id_list,
            )
        else:
            resp, _, _ = discrete_llm.chat(messages=messages, stream=False, temperature=0.0, max_tokens=256)
        if not hasattr(resp, "content"):
            raise ValueError("LLM response missing content")
        
        guidance = resp.content.strip()
        if not guidance:
            raise ValueError("Empty guidance returned by VLM")
        
        self.logger.info(f"[Discrete Memory] Digested guidance ({len(guidance.split())} words): {guidance}")
        return guidance

    def _build_discrete_memory_block(
        self,
        intent: str,
        file_id_list: List[str],
        experience_actions: List[List[Dict]],
        experience_images: List[List[str]],
        current_image: Optional[str] = None,
    ) -> str:
        """
        Build discrete memory block with two-stage processing:
        1. Summarize each retrieved trajectory (cached)
        2. Digest all summaries into task-specific guidance using current task + image
        """
        self.logger.info(f"[Discrete Memory] Building summaries for {len(file_id_list)} trajectories...")
        if self.memory is None:
            raise ValueError("memory must be initialized to build discrete memory")
        if not isinstance(file_id_list, list) or not isinstance(experience_actions, list) or not isinstance(experience_images, list):
            raise ValueError("file_id_list/experience_actions/experience_images must be lists")
        if not (len(file_id_list) == len(experience_actions) == len(experience_images)):
            raise ValueError("file_id_list, experience_actions, and experience_images must have the same length")

        # Load cache once per run (lazy)
        if not self.discrete_memory_cache:
            self.discrete_memory_cache = self._load_discrete_memory_cache()

        max_actions = int(getattr(self.args, "discrete_memory_max_actions", 8))
        selected_files = getattr(self.memory, "selected_conversations", None)
        if not isinstance(selected_files, list):
            raise ValueError("memory.selected_conversations must be a list of file paths")

        # Stage 1: Collect all trajectory summaries
        summaries: List[str] = []
        updated = False
        for file_id, actions, imgs in zip(file_id_list, experience_actions, experience_images):
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("file_id must be a non-empty string")
            cached = self.discrete_memory_cache.get(file_id)
            if cached is None:
                matches = [fp for fp in selected_files if isinstance(fp, str) and fp.endswith(f"/{file_id}.jsonl")]
                if len(matches) != 1:
                    raise ValueError(f"Could not uniquely map file_id={file_id!r} to a selected conversation file")
                memory_file = json.loads(Path(matches[0]).read_text())
                task = memory_file.get("task_description")
                if not isinstance(task, str) or not task.strip():
                    raise ValueError(f"Missing/invalid task_description in {matches[0]}")
                actions_text = self._format_actions_for_summary(actions, max_actions=max_actions)
                first_img = imgs[0] if (isinstance(imgs, list) and len(imgs) > 0) else None
                summary = self._summarize_trajectory_with_vlm(
                    task=task,
                    actions_text=actions_text,
                    image_b64=first_img,
                    experience_texts=experience_actions,
                    experience_images=experience_images,
                    file_id_list=file_id_list,
                )
                self.discrete_memory_cache[file_id] = summary
                updated = True
            else:
                summary = cached
                self.logger.info(f"[Discrete Memory] (cached) {file_id}: {summary}")
            summaries.append(summary)

        if updated:
            self._save_discrete_memory_cache(self.discrete_memory_cache)

        # Stage 2: Digest summaries into task-specific guidance
        digested_guidance = self._digest_discrete_memory(
            current_task=intent,
            current_image=current_image,
            trajectory_summaries=summaries,
            experience_texts=experience_actions,
            experience_images=experience_images,
            file_id_list=file_id_list,
        )

        return f"[Experience Guidance]\n{digested_guidance}"

    def _build_graph_memory_block(
        self,
        intent: str,
        current_image: Optional[str] = None,
    ) -> str:
        """
        Build graph memory block with diverse trajectory retrieval + digestion.
        
        Uses FAISS + graph expansion for diversity, then:
        1. DISCRETE: Digests takeaways into text guidance for system prompt
        2. CONTINUOUS: Extracts actions/images for Q-Former (if use_continuous_memory)
        
        This makes graph memory TRULY HYBRID.
        
        Args:
            intent: Current task intent
            current_image: Current screenshot (optional, for multimodal retrieval)
        
        Returns:
            Formatted experience guidance string
        """
        if self.graph_memory_retriever is None:
            return ""
        
        self.logger.info(f"[GraphMemory] Building graph memory block for: {intent}...")
        
        # Multimodal query text (match discrete-memory convention) + current screenshot
        query_embedding = None
        dataset = getattr(self.args, "evaluation_type", "")
        domain_name = getattr(self.args, "domain", "")
        query_text = f"{dataset}_{domain_name}: {intent}" if dataset and domain_name else intent
        
        # NOTE: We intentionally skip tag-based query expansion for now while validating
        # the base multimodal retrieval. (Graph expansion/diversity can be re-added later.)
        query_tags = None
        
        # Retrieve diverse trajectories
        k = getattr(self.args, 'graph_similar_num', 5)
        results = self.graph_memory_retriever.retrieve(
            query_embedding=query_embedding,
            query_text=query_text,
            query_image=current_image,
            query_tags=query_tags,
            k=k
        )
        
        if not results:
            self.logger.warning("[GraphMemory] No trajectories retrieved")
            return ""
        
        self.logger.info(f"[GraphMemory] Retrieved {len(results)} diverse trajectories")
        
        # Store raw takeaways for dynamic memory checkpoint
        self.current_raw_takeaways = results
        
        # Collect takeaways for digestion (DISCRETE MEMORY)
        takeaways = [traj.takeaway for traj in results]
        
        # Log retrieved takeaways and their tags
        for i, traj in enumerate(results):
            tag_str = ", ".join(sorted(list(traj.tags)[:5]))
            self.logger.info(f"  [{i+1}] {traj.id}: {traj.takeaway} | tags: {tag_str}")
        
        # ================================================================
        # CONTINUOUS MEMORY: Extract actions/images for Q-Former
        # ================================================================
        if getattr(self.args, 'use_continuous_memory', False):
            self._extract_continuous_memory_from_graph(results)
        
        # Digest takeaways into task-specific guidance (DISCRETE MEMORY)
        digested_guidance = self._digest_graph_memory(
            current_task=intent,
            current_image=current_image,
            takeaways=takeaways,
        )
        
        return f"[Graph Memory Guidance]\n{digested_guidance}"
    
    def _find_trajectory_file(self, stored_path: str, trajectory_id: str) -> Optional[str]:
        """
        Find a trajectory file by searching across multiple memory data directories.
        
        The stored_path might be from a different machine/directory structure, so we try:
        1. The stored path directly (if it exists)
        2. Search in each memory_data_dir for the file by name
        
        Args:
            stored_path: The file path stored in the graph (may not exist locally)
            trajectory_id: The trajectory ID (used as fallback for filename)
            
        Returns:
            The actual file path if found, None otherwise
        """
        import glob
        
        # Try the stored path first
        if stored_path and os.path.exists(stored_path):
            return stored_path
        
        # Extract filename from stored path
        filename = os.path.basename(stored_path) if stored_path else f"{trajectory_id}.jsonl"
        
        # Search across all memory data directories
        for base_dir in self.memory_data_dirs:
            # Search recursively for the file
            pattern = os.path.join(base_dir, '**', filename)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                self.logger.debug(f"[GraphMemory] Found {filename} at {matches[0]}")
                return matches[0]
        
        # Not found
        self.logger.debug(f"[GraphMemory] Could not find {filename} in any memory_data_dir")
        return None
    
    def _extract_continuous_memory_from_graph(self, results: List) -> None:
        """
        Extract actions and images from graph memory results for continuous memory (Q-Former).
        
        Populates self.experience_texts, self.experience_images, and self.file_id_list
        so they can be passed to the Q-Former model during action generation.
        
        If actions/images are not in full_data (e.g., loaded from saved graph),
        tries to reload from the original trajectory file.
        
        Args:
            results: List of TaggedTrajectory objects from graph memory retrieval
        """
        # Only populate if not already set by discrete memory
        if self.experience_texts is not None and self.experience_images is not None:
            self.logger.info("[GraphMemory] Continuous memory already populated by discrete memory, skipping")
            return
        
        experience_texts = []
        experience_images = []
        file_id_list = []
        
        for traj in results:
            full_data = traj.full_data or {}
            
            actions = full_data.get('actions', [])
            images = full_data.get('images', [])
            
            # If actions/images not in full_data, try to load from original file
            if not actions or not images:
                stored_path = full_data.get('file_path', '')
                # Search across all memory_data_dirs for the file
                actual_path = self._find_trajectory_file(stored_path, traj.id)
                if actual_path:
                    loaded_actions, loaded_images = self._load_trajectory_for_continuous_memory(actual_path)
                    if loaded_actions and loaded_images:
                        actions = loaded_actions
                        images = loaded_images
                        self.logger.info(f"[GraphMemory] Loaded {len(actions)} actions from {actual_path}")
            
            if not actions or not images:
                self.logger.warning(f"[GraphMemory] {traj.id} missing actions/images, skipping for continuous memory")
                continue
            
            # Format actions as expected by Q-Former (list of action dicts with reasoning)
            formatted_actions = []
            for action in actions:
                if isinstance(action, dict):
                    # Ensure proper format: {"name": "...", "arguments": {"reasoning": "..."}}
                    formatted_actions.append(action)
                elif isinstance(action, str):
                    # Handle string actions (legacy format)
                    formatted_actions.append({"name": "unknown", "arguments": {"reasoning": action}})
            
            # Format images with data URL prefix if needed
            formatted_images = []
            for img in images:
                if img is None:
                    continue
                if isinstance(img, str):
                    if img.startswith('data:image'):
                        formatted_images.append(img)
                    else:
                        formatted_images.append(f"data:image/png;base64,{img}")
            
            if formatted_actions and formatted_images:
                experience_texts.append(formatted_actions)
                experience_images.append(formatted_images)
                file_id_list.append(traj.id)
                self.logger.info(f"[GraphMemory→Continuous] {traj.id}: {len(formatted_actions)} actions, {len(formatted_images)} images")
        
        if experience_texts and experience_images:
            self.experience_texts = experience_texts
            self.experience_images = experience_images
            self.file_id_list = file_id_list
            self.logger.info(f"[GraphMemory] Populated continuous memory with {len(experience_texts)} trajectories")
    
    def _load_trajectory_for_continuous_memory(self, file_path: str) -> tuple:
        """
        Load actions and images from a trajectory file for continuous memory.
        
        Args:
            file_path: Path to the trajectory JSONL file
            
        Returns:
            (actions, images) tuple, or ([], []) if loading fails
        """
        import re
        
        def parse_action_from_response(response: str) -> Optional[Dict]:
            """Parse action JSON from response string, handling double-encoded JSON."""
            if not isinstance(response, str):
                return None
            
            # Handle double-encoded JSON (guiact_converted format)
            # The response might be a JSON string that contains another JSON string
            try:
                if response.startswith('"') and response.endswith('"'):
                    decoded = json.loads(response)
                    if isinstance(decoded, str):
                        response = decoded
            except json.JSONDecodeError:
                pass
            
            # Try ```json block
            if "```json" in response:
                match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except json.JSONDecodeError:
                        pass
            # Try Action: prefix
            match = re.search(r'Action:\s*(\{.*\})', response)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            # Try direct JSON
            try:
                obj = json.loads(response)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
            return None
        
        def extract_base64_image(round_data: Dict) -> Optional[str]:
            """Extract base64 image from round messages."""
            if 'messages' not in round_data:
                return None
            for msg in round_data['messages']:
                content = msg.get('content')
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'image_url':
                            return item['image_url']['url']
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            rounds = data.get('rounds', [])
            actions = []
            images = []
            
            for r in rounds:
                response = r.get('response', '')
                if isinstance(response, list):
                    response = response[0] if response else ''
                if isinstance(response, dict) and 'content' in response:
                    response = response['content']
                
                action = parse_action_from_response(response)
                image = extract_base64_image(r)
                
                if action and 'name' in action and image:
                    actions.append(action)
                    images.append(image)
            
            return actions, images
            
        except Exception as e:
            self.logger.warning(f"[GraphMemory] Error loading {file_path}: {e}")
            return [], []
    
    def _digest_graph_memory(
        self,
        current_task: str,
        current_image: Optional[str],
        takeaways: List[str],
        progress_context: Optional[str] = None,
    ) -> str:
        """
        Digest retrieved graph memory takeaways into task-specific guidance.
        
        Similar to _digest_discrete_memory but for graph memory results.
        
        Args:
            current_task: The task intent
            current_image: Current screenshot (base64)
            takeaways: List of retrieved experience takeaways
            progress_context: Optional context about what the agent has already done
                              and what it should focus on next (used during mid-task updates)
        """
        if not takeaways:
            raise ValueError("takeaways cannot be empty")
        
        summaries_text = "\n".join(f"- {s}" for s in takeaways)
        
        if progress_context:
            # Mid-task update: acknowledge progress and focus on next steps
            system = (
                "You are an expert at analyzing past GUI agent experiences to help with an ONGOING task.\n"
                "The agent is ALREADY IN PROGRESS - it has completed some steps and now needs guidance for the NEXT phase.\n\n"
                "Given:\n"
                "- The overall task goal\n"
                "- What the agent has already accomplished and what it needs to focus on next\n"
                "- The current screenshot showing where the agent is NOW\n"
                "- Retrieved experience takeaways relevant to the NEXT steps\n\n"
                "Synthesize guidance that helps the agent move forward from its CURRENT position.\n\n"
                "Output format: ONE concise paragraph (2-3 sentences) that:\n"
                "1. Acknowledges the current progress (briefly)\n"
                "2. Provides specific guidance for the NEXT steps based on the new focus\n\n"
                "IMPORTANT RULES:\n"
                "- Do NOT suggest actions the agent has already completed (like initial search)\n"
                "- Focus on what to do NEXT from the current screen state\n"
                "- Be specific about filters, buttons, or elements to interact with\n"
                "- Do NOT mention stopping, completing, or finishing the task.\n"
                "- Do NOT use bullet points. Write as a coherent paragraph."
            )
            user_text = (
                f"Overall Task: {current_task}\n\n"
                f"Current Progress & Next Focus:\n{progress_context}\n\n"
                f"Retrieved Experience Takeaways (for next steps):\n{summaries_text}"
            )
        else:
            # Initial stage: starting from scratch
            system = (
                "You are an expert at analyzing past GUI agent experiences to help with a new task.\n"
                "Given the current task, current screenshot, and retrieved experience takeaways,\n"
                "synthesize them into focused, actionable guidance.\n\n"
                "Output format: ONE concise paragraph (2-3 sentences) that answers:\n"
                "1. Which strategies from past experiences are MOST relevant to this specific task?\n"
                "2. What key actions or filters should be prioritized?\n\n"
                "IMPORTANT RULES:\n"
                "- Focus ONLY on navigation/search strategies, NOT on when to stop.\n"
                "- Do NOT mention stopping, completing, or finishing the task.\n"
                "- Do NOT give instructions about providing answers or explanations.\n"
                "- Be specific to the current task. Do NOT just repeat the summaries.\n"
                "- Do NOT use bullet points. Write as a coherent paragraph."
            )
            user_text = f"Current Task: {current_task}\n\nRetrieved Experience Takeaways:\n{summaries_text}"
        
        if current_image is None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ]
        else:
            if not isinstance(current_image, str):
                raise ValueError("current_image must be a string when provided")
            # Handle both raw base64 and data URL formats
            if current_image.startswith("data:image"):
                image_url = current_image
            else:
                image_url = f"data:image/png;base64,{current_image}"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ]
        
        self.logger.info("[GraphMemory] Digesting takeaways into task-specific guidance...")
        resp, _, _ = self.tool_llm.chat(messages=messages, stream=False, temperature=0.0, max_tokens=256)
        
        if not hasattr(resp, "content"):
            raise ValueError("LLM response missing content")
        
        guidance = resp.content.strip()
        if not guidance:
            raise ValueError("Empty guidance returned by VLM")
        
        self.logger.info(f"[GraphMemory] Digested guidance ({len(guidance.split())} words): {guidance}")
        return guidance

    def _check_memory_update_needed(
        self,
        intent: str,
        last_action: str,
        before_image: str,
        after_image: str,
    ) -> Tuple[bool, str, Optional[str], Optional[List[int]]]:
        """
        VLM checkpoint to decide if memory update is needed.
        
        Compares the current raw takeaways against the before/after screenshots
        to determine if the retrieved memories are still relevant.
        
        Args:
            intent: Current task intent
            last_action: The action just executed
            before_image: Screenshot before the action (base64)
            after_image: Screenshot after the action (base64)
        
        Returns:
            Tuple of (update_needed, reason, new_focus or None, preserved_indices or None)
            preserved_indices: 1-indexed list of takeaway indices to preserve when updating
        """
        # Limit max updates per task to prevent memory drift
        max_updates = getattr(self.args, 'max_memory_updates', 3)
        if self.memory_update_count >= max_updates:
            self.logger.info(f"[DynamicMemory] Skipping checkpoint: reached max updates ({max_updates})")
            return False, f"Reached max updates limit ({max_updates})", None, None
        
        if self.current_raw_takeaways is None or not self.current_raw_takeaways:
            self.logger.warning("[DynamicMemory] No raw takeaways available for checkpoint")
            return False, "No takeaways to check", None, None
        
        # Format raw takeaways with tags for VLM input
        takeaways_lines = []
        for i, traj in enumerate(self.current_raw_takeaways):
            tag_str = ", ".join(sorted(list(traj.tags)[:5]))
            takeaways_lines.append(f"[{i+1}] {traj.takeaway} | tags: {tag_str}")
        raw_takeaways_text = "\n".join(takeaways_lines)
        
        # Memory checkpoint prompt - let VLM decide conservatively
        # Key insight: Distinguish OPERATIONAL ERRORS from MEMORY RELEVANCE issues
        system_prompt = """You are evaluating whether the retrieved experience memories are still relevant for the task.

KEY PRINCIPLES:
1. Memories provide NAVIGATION STRATEGIES (how to search, filter, click, scroll, etc.), NOT answers.
2. Be VERY CONSERVATIVE - default to keeping current memories.
3. CRITICAL: Distinguish between OPERATIONAL ERRORS vs MEMORY RELEVANCE:

   OPERATIONAL ERRORS (DO NOT update memory):
   - Agent landed on wrong page (login page, unrelated website, error page)
   - Agent is stuck or action failed
   - Agent made a navigation mistake
   → These are execution errors, NOT memory problems. The current memories are still valid for when agent gets back on track.

   MEMORY RELEVANCE ISSUE (May update memory):
   - Task has genuinely progressed to a completely new phase
   - Agent successfully completed initial steps and now needs different strategies
   - Current screen shows the task is in a fundamentally different domain than memories cover
   → Only update if memories are irrelevant to the ACTUAL TASK PROGRESS, not temporary navigation mistakes.

4. PRESERVE USEFUL MEMORIES: When updating, some existing takeaways may still be valuable:
   - General domain knowledge that applies across task phases
   - Strategies that might be needed again later (e.g., navigation, filtering)
   - Takeaways that provide complementary information to new phase requirements

Your job: Decide if memories should be updated, but ONLY for genuine relevance issues, NOT operational errors.
When updating, also identify which existing takeaways should be PRESERVED and combined with new retrieval."""

        num_takeaways = len(self.current_raw_takeaways)
        user_text = f"""## Task Goal
{intent}

## Currently Retrieved Experience Takeaways ({num_takeaways} total)
{raw_takeaways_text}

## Action Just Taken
{last_action}

## Your Task
Look at the BEFORE and AFTER screenshots. Evaluate:

1. Is this an OPERATIONAL ERROR? (wrong page, login page, stuck, action failed)
   → If YES: Do NOT update memory. The agent just needs to recover, memories are still valid.

2. Is this a genuine TASK PROGRESSION where current memories no longer apply?
   → If YES: Consider updating memory, but also identify which existing takeaways to PRESERVE.

Remember: If agent accidentally went to a wrong page (like a login page or different website), 
the original memories are STILL RELEVANT for when it gets back to the correct page.

## Output Format
First write your reasoning, then output a JSON decision block:

### Reasoning:
<Analyze: Is this an operational error or genuine memory irrelevance?>

### Decision:
```json
{{"update_needed": false}}
```

OR (ONLY if genuine memory irrelevance, NOT operational error):

### Reasoning:
<Why this is genuine task progression requiring different strategies>
<Which existing takeaways (by number) should be preserved and why>

### Decision:
```json
{{"update_needed": true, "new_focus": "<what new navigation strategies are needed>", "preserve": [<list of takeaway numbers to keep, e.g., 1, 3, 5>]}}
```

IMPORTANT: "preserve" must only contain numbers from 1 to {num_takeaways} (the takeaways listed above).
These preserved takeaways will be combined with newly retrieved memories. Use empty list [] if none should be kept."""

        # Build message with before/after images
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "text", "text": "BEFORE action:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_image}"}},
                {"type": "text", "text": "AFTER action:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_image}"}},
            ]},
        ]
        
        self.logger.info("[DynamicMemory] Running memory checkpoint...")
        
        resp, _, _ = self.tool_llm.chat(messages=messages, stream=False, temperature=0.0, max_tokens=500)
        
        if not hasattr(resp, "content"):
            self.logger.warning("[DynamicMemory] VLM response missing content")
            return False, "VLM response missing content", None
        
        response_text = resp.content.strip()
        self.logger.info(f"[DynamicMemory] Checkpoint response: {response_text}")
        
        # Parse JSON response - look for ```json block first, then fallback to raw JSON
        import re
        
        # Try to find ```json block first (aligned with agent CoT format)
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_block_match:
            json_str = json_block_match.group(1).strip()
        else:
            # Fallback: try to find raw JSON object
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if not json_match:
                self.logger.warning(f"[DynamicMemory] Could not parse JSON from response: {response_text}")
                return False, "Failed to parse checkpoint response", None
            json_str = json_match.group()
        
        # Fix invalid escape sequences (VLM sometimes outputs \' which is not valid JSON)
        json_str = json_str.replace("\\'", "'")
        
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.warning(f"[DynamicMemory] JSON parse error: {e}")
            return False, f"JSON parse error: {e}", None, None
        
        update_needed = result.get("update_needed", False)
        new_focus = result.get("new_focus") if update_needed else None
        preserved_indices = result.get("preserve", []) if update_needed else None
        
        # Validate preserved_indices: should be list of integers
        if preserved_indices is not None:
            try:
                preserved_indices = [int(idx) for idx in preserved_indices]
            except (TypeError, ValueError):
                self.logger.warning(f"[DynamicMemory] Invalid preserve format: {preserved_indices}, ignoring")
                preserved_indices = []
        
        # Extract reasoning from the response text
        reasoning_match = re.search(r'### Reasoning:\s*([\s\S]*?)(?=###|```|$)', response_text)
        reason = reasoning_match.group(1).strip() if reasoning_match else str(result)
        
        if update_needed:
            self.logger.info(f"[DynamicMemory] Update needed! Reason: {reason}...")
            self.logger.info(f"[DynamicMemory] New focus: {new_focus}")
            if preserved_indices:
                self.logger.info(f"[DynamicMemory] Preserving takeaways: {preserved_indices}")
        else:
            self.logger.info(f"[DynamicMemory] No update needed. Reason: {reason}...")
        
        return update_needed, reason, new_focus, preserved_indices

    def _update_graph_memory(
        self,
        intent: str,
        new_focus: str,
        current_image: str,
        reason: str = "",
        preserved_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Re-retrieve from graph memory with new focus and update both discrete and continuous memory.
        
        Supports SELECTIVE PRESERVATION: keeps useful old takeaways and combines with new ones.
        
        Args:
            intent: Current task intent
            new_focus: The new focus/guidance type to retrieve
            current_image: Current screenshot (base64) for multimodal retrieval
            reason: The checkpoint reason explaining what's been done and why update is needed
            preserved_indices: 1-indexed list of old takeaway indices to preserve (from VLM decision)
        """
        if self.graph_memory_retriever is None:
            self.logger.warning("[DynamicMemory] Cannot update: graph_memory_retriever is None")
            return
        
        self.memory_update_count += 1
        self.logger.info(f"[DynamicMemory] Updating memory (update #{self.memory_update_count})")
        self.logger.info(f"[DynamicMemory] Focus hint (for digestion): {new_focus}")
        
        # ================================================================
        # STEP 1: Extract preserved trajectories from old memories
        # ================================================================
        preserved_trajectories = []
        if preserved_indices and self.current_raw_takeaways:
            for idx in preserved_indices:
                # Convert 1-indexed (from VLM output) to 0-indexed
                zero_idx = idx - 1
                if 0 <= zero_idx < len(self.current_raw_takeaways):
                    preserved_trajectories.append(self.current_raw_takeaways[zero_idx])
                else:
                    self.logger.warning(f"[DynamicMemory] Invalid preserve index {idx}, skipping")
            
            if preserved_trajectories:
                self.logger.info(f"[DynamicMemory] Preserving {len(preserved_trajectories)} old trajectories:")
                for i, traj in enumerate(preserved_trajectories):
                    self.logger.info(f"  [preserved-{i+1}] {traj.id}: {traj.takeaway[:80]}...")
        
        # ================================================================
        # STEP 2: Retrieve new trajectories
        # ================================================================
        # Use CLEAN intent for retrieval - avoid instruction/error text polluting semantic search
        # The focus is used as a "soft boost" during digestion, not query modification
        retrieval_intent = self.clean_intent if self.clean_intent else intent
        dataset = getattr(self.args, "evaluation_type", "")
        domain_name = getattr(self.args, "domain", "")
        query_text = f"{dataset}_{domain_name}: {retrieval_intent}" if dataset and domain_name else retrieval_intent
        
        self.logger.info(f"[DynamicMemory] Query (clean intent): {query_text}")
        
        # Retrieve k new trajectories
        k = getattr(self.args, 'graph_similar_num', 5)
        new_results = self.graph_memory_retriever.retrieve(
            query_embedding=None,
            query_text=query_text,
            query_image=current_image,
            query_tags=None,
            k=k
        )
        
        if not new_results:
            self.logger.warning("[DynamicMemory] No new trajectories retrieved")
            # If we have preserved ones, still use them
            if preserved_trajectories:
                new_results = []
            else:
                return
        
        self.logger.info(f"[DynamicMemory] Retrieved {len(new_results)} new trajectories")
        
        # Log new takeaways
        for i, traj in enumerate(new_results):
            tag_str = ", ".join(sorted(list(traj.tags)[:5]))
            self.logger.info(f"  [new-{i+1}] {traj.id}: {traj.takeaway} | tags: {tag_str}")
        
        # ================================================================
        # STEP 3: Combine preserved + new (deduplicate by ID)
        # ================================================================
        preserved_ids = {traj.id for traj in preserved_trajectories}
        
        # Filter out duplicates from new results
        deduplicated_new = [traj for traj in new_results if traj.id not in preserved_ids]
        if len(deduplicated_new) < len(new_results):
            self.logger.info(f"[DynamicMemory] Removed {len(new_results) - len(deduplicated_new)} duplicate(s) from new results")
        
        # Combine: preserved first (they have proven value), then new
        combined_results = preserved_trajectories + deduplicated_new
        self.logger.info(f"[DynamicMemory] Combined memory: {len(preserved_trajectories)} preserved + {len(deduplicated_new)} new = {len(combined_results)} total")
        
        # Update raw takeaways for next checkpoint
        self.current_raw_takeaways = combined_results
        
        # ================================================================
        # UPDATE DISCRETE MEMORY (System Prompt)
        # ================================================================
        # Use ALL combined takeaways (preserved + new) for digestion
        takeaways = [traj.takeaway for traj in combined_results]
        
        # Build progress context to inform the VLM about current state
        # This helps generate guidance appropriate for the CURRENT stage, not from scratch
        progress_parts = []
        if reason:
            progress_parts.append(f"Current situation: {reason}")
        progress_parts.append(f"Next focus: {new_focus}")
        if preserved_trajectories:
            progress_parts.append(f"Note: {len(preserved_trajectories)} preserved memories from previous phase are included.")
        progress_context = "\n".join(progress_parts)
        
        # Re-digest with combined takeaways AND progress context
        new_guidance = self._digest_graph_memory(
            current_task=intent,
            current_image=current_image,
            takeaways=takeaways,
            progress_context=progress_context,
        )
        
        # Update the graph memory block
        preserved_note = f" ({len(preserved_trajectories)} preserved)" if preserved_trajectories else ""
        self.graph_memory_block = f"[Graph Memory Guidance - Updated at Step {self.current_step}{preserved_note}]\n{new_guidance}"
        
        # Also update experience_memory to include the new block
        if self.experience_memory is not None:
            # Replace old graph memory guidance if present
            if "[Graph Memory Guidance" in self.experience_memory:
                import re
                self.experience_memory = re.sub(
                    r'\[Graph Memory Guidance[^\]]*\].*?(?=\[|$)',
                    self.graph_memory_block,
                    self.experience_memory,
                    flags=re.DOTALL
                )
            else:
                self.experience_memory = self.experience_memory + "\n\n" + self.graph_memory_block
        
        self.logger.info(f"[DynamicMemory] Updated discrete memory (graph_memory_block)")
        
        # ================================================================
        # UPDATE CONTINUOUS MEMORY (Q-Former Embeddings)
        # ================================================================
        if getattr(self.args, 'use_continuous_memory', False):
            # Clear existing continuous memory to force re-extraction
            self.experience_texts = None
            self.experience_images = None
            self.file_id_list = None
            
            # Re-extract from ALL combined trajectories (preserved + new)
            # This ensures Q-Former gets the same unified memory as discrete digestion
            self._extract_continuous_memory_from_graph(combined_results)
            
            self.logger.info(f"[DynamicMemory] Updated continuous memory: {len(combined_results)} trajectories (experience_texts/images)")
        
        self.logger.info(f"[DynamicMemory] Memory update complete (update #{self.memory_update_count})")

    def _generate_k_candidate_actions(self, messages: List[Dict], k: int) -> str:
        """Generate k candidate actions by asking LLM to generate multiple actions"""
        k_generation_message = {
            'role': 'user',
            'content': f"Generate {k} different possible actions you could take next. Each action should be distinct and reasonable. For each action, provide a function_call in the standard format with name and arguments. Seperate each action by ';'."
        }
        k_messages = messages + [k_generation_message]
        actions = []
        while len(actions) < k:
            responses, _, _ = self.llm.chat(messages=k_messages, stream=False)
            actions = responses.content.split(';')
            actions = [action for action in actions if action != '']
        return actions

    def _self_refine_actions(self, candidate_actions: List[str], intent: str, current_observation: str) -> List[str]:
        """Self-refine actions to remove low quality ones"""
        if not candidate_actions:
            return []
        
        refinement_prompt = f"""You are evaluating {len(candidate_actions)} candidate actions for the task: {intent}

For each action, determine if it is:
1. Relevant to the task
2. Feasible given the current state
3. Not redundant or low quality

Actions to evaluate:
"""
        for i, action in enumerate(candidate_actions):
            refinement_prompt += f"\n{i+1}. {action}\n"

        refinement_prompt += "\n\nReturn a list of indices (0-indexed) of actions that should be KEPT (high quality actions). Only keep actions that are clearly relevant, feasible, and high quality."

        messages = [
            {'role': 'system', 'content': 'You are an expert at evaluating action quality. Return only a list of indices.'},
            {'role': 'user', 'content': [
                {"type": "text", "text": "Current observation:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_observation}"}},
                {'type': 'text', 'text': refinement_prompt}]}
        ]
        
        response, _, _ = self.tool_llm.chat(messages=messages, stream=False)
        content = response.content.strip()
        
        kept_indices = []
        if content.startswith('[') and content.endswith(']'):
            kept_indices = json.loads(content)
        else:
            import re
            matches = re.findall(r'\d+', content)
            kept_indices = [int(m) for m in matches]
        
        refined_actions = [candidate_actions[i] for i in kept_indices if 0 <= i < len(candidate_actions)]
        return refined_actions if refined_actions else candidate_actions

    def next_action_custom(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: Dict[str, Any],
    ):
        """Generate the next action using function calling with ReAct paradigm"""
        
        self.current_step += 1
        print('*'*50, 'current step: ', self.current_step, '*'*50)
        # Prepare messages for the LLM
        messages, meta_data = self._prepare_messages(trajectory, intent, meta_data)
        
        use_implicit_world_model = getattr(self.args, 'use_implicit_world_model', False)
        use_implicit_world_model = False
        # if use_implicit_world_model:
        #     k = getattr(self.args, 'implicit_world_model_k', 3)
        #     for obs_data in trajectory:
        #         if isinstance(obs_data, dict) and 'observation' in obs_data:
        #             image_b64 = obs_data['observation'].get('image', '')
        #     candidate_actions = self._generate_k_candidate_actions(messages, k)
        #     print('*'*50, 'candidate actions', '*'*50)
        #     print(len(candidate_actions))
        #     print(candidate_actions)
        #     print('*'*50, 'candidate actions', '*'*50)
        #     # refined_actions = self._self_refine_actions(candidate_actions, intent, image_b64)
        #     # print('*'*50, 'refined actions', '*'*50)
        #     # print(len(refined_actions))
        #     # print(refined_actions)
        #     # print('*'*50, 'refined actions', '*'*50)
        #     refined_actions = candidate_actions
        #     if refined_actions and isinstance(refined_actions, list):
        #         screenshots_pil = [Image.open(BytesIO(base64.b64decode(image_b64)))]
        #         actions_history = meta_data.get('action_history', [])
        #         current_url = meta_data.get('url', '')
                
        #         scores, all_simulations = evaluate_simulation(
        #             screenshots_pil,
        #             actions_history,
        #             intent,
        #             current_url,
        #             refined_actions
        #         )
                
        #         if scores:
        #             best_action_description = max(scores.items(), key=lambda x: x[1])[0]
        #             best_action_idx = refined_actions.index(best_action_description) if best_action_description in refined_actions else 0
        #         else:
        #             best_action_idx = 0
        #         selected_action = refined_actions[best_action_idx]
        #         action = self._process_response(selected_action, trajectory, meta_data.get('page'), intent)
        #         meta_data['response_history'].append(selected_action)
        #         meta_data['candidate_actions_scores'] = str(scores)
        #         print('*'*50, 'implicit world model scores', '*'*50)
        #         print(scores)
        #         print('*'*50, 'selected action', '*'*50)
        #         print(selected_action)
                
        #         return (action, meta_data)
        
        # Continuous memory (Q-Former) should use whatever experience has been populated,
        # regardless of whether the legacy discrete-memory retriever is enabled.
        # - With --use_discrete_memory: experience_* comes from FAISS retrieval.
        # - With --use_graph_memory:   experience_* comes from graph retrieval.
        if self.args.use_continuous_memory and (self.experience_texts is not None or self.experience_images is not None):
            responses, original_inputs, original_outputs = self.llm.chat(messages=messages, stream=False, 
                                        experience_texts=self.experience_texts, experience_images=self.experience_images,
                                        file_id_list=self.file_id_list)
        else:
            # Optional: feed hybrid exemplars as experience when no memory is present
            if getattr(self.args, 'use_continuous_memory', False) and self.memory is None and self.last_hybrid_exemplars:
                try:
                    # Build minimal experience inputs from latent packs (summary + keyframes)
                    exp_texts = []
                    exp_images = []
                    for ex in self.last_hybrid_exemplars:
                        lp_path = ex.get("latent_pack_path")
                        if not lp_path or not os.path.exists(lp_path):
                            continue
                        npz = np.load(lp_path)
                        meta_json = bytes(npz["meta_json"].astype(np.uint8).tolist()).decode("utf-8")
                        import json as _json
                        meta = _json.loads(meta_json)
                        summary_text = meta.get("summary_text", "")
                        kfps = meta.get("keyframe_paths", []) or []
                        imgs_b64 = []
                        for p in kfps[:2]:
                            try:
                                with open(p, "rb") as f:
                                    b64 = base64.b64encode(f.read()).decode("utf-8")
                                    imgs_b64.append(f"data:image/png;base64,{b64}")
                            except Exception:
                                pass
                        if summary_text:
                            exp_texts.append([summary_text])
                        if imgs_b64:
                            exp_images.append(imgs_b64)
                    if exp_texts or exp_images:
                        responses, original_inputs, original_outputs = self.llm.chat(
                            messages=messages, stream=False,
                            experience_texts=exp_texts if exp_texts else None,
                            experience_images=exp_images if exp_images else None
                        )
                    else:
                        responses, original_inputs, original_outputs = self.llm.chat(messages=messages, stream=False)
                except Exception as e:
                    self.logger.warning(f"[HybridMemory] continuous bridge failed: {e}")
                    responses, original_inputs, original_outputs = self.llm.chat(messages=messages, stream=False)
            else:
                # Call the LLM with function calling
                responses, original_inputs, original_outputs = self.llm.chat(messages=messages, stream=False)
        meta_data['original_inputs'] = original_inputs
        meta_data['original_outputs'] = original_outputs
        meta_data['original_responses'] = responses
        if self.training_collector:
            self.llm.save_conversation(messages, responses.content)
        print('*'*50, 'responses', '*'*50)
        print(responses.content)
        print('*'*50, 'responses', '*'*50)
        meta_data['response_history'].append(responses.content)
                
        # Extract page if available in meta_data or elsewhere; pass explicitly
        page_for_tools = meta_data.get('page')
            
        # Process the response
        action = self._process_response(responses.content, trajectory, page_for_tools, intent)
        print('*'*50, 'action', '*'*50)
        print(action)
        print('*'*50, 'action', '*'*50)
        
        return (action, meta_data)
    
    def _get_current_screenshot(self, trajectory: Trajectory) -> Optional[str]:
        """Extract the current screenshot from the trajectory for multimodal memory retrieval."""
        recent_obs = trajectory[-1]
        if isinstance(recent_obs, dict) and 'observation' in recent_obs:
            obs = recent_obs['observation']
            if 'image' in obs:
                return obs['image']
        return None
    
    def _get_first_screenshot(self, trajectory: Trajectory) -> Optional[str]:
        """Extract the first screenshot from the trajectory for multimodal memory retrieval."""
        recent_obs = trajectory[0]
        return recent_obs['observation']['image']
    
    def _detect_repetition_and_no_progress(self, trajectory: Trajectory, meta_data: Dict[str, Any]) -> Optional[str]:
        """Detect if the last action made no progress (same URL and screenshot) or is repeating."""
        try:
            if not trajectory or len(trajectory) < 2:
                return None
            last_state = trajectory[-1]
            prev_state = trajectory[-2]
            # URL comparison (from info.page.url if available)
            last_url = None
            prev_url = None
            try:
                if isinstance(last_state, dict) and 'info' in last_state and 'page' in last_state['info']:
                    last_url = getattr(last_state['info']['page'], 'url', None)
                if isinstance(prev_state, dict) and 'info' in prev_state and 'page' in prev_state['info']:
                    prev_url = getattr(prev_state['info']['page'], 'url', None)
            except Exception:
                pass
            # Image comparison (base64 exact match)
            last_img = last_state.get('observation', {}).get('image') if isinstance(last_state, dict) else None
            prev_img = prev_state.get('observation', {}).get('image') if isinstance(prev_state, dict) else None
            url_same = (last_url is not None and prev_url is not None and last_url == prev_url)
            img_same = (last_img is not None and prev_img is not None and last_img == prev_img)
            
            # Log detection details
            self.logger.debug(f"[StuckDetection] URL same: {url_same}, Image same: {img_same}")
            if last_url and prev_url:
                self.logger.debug(f"[StuckDetection] URLs: {prev_url[:80]} -> {last_url[:80]}")
            
            # Repetition detection from textual action history
            repeated_count = 0
            last_action_text = None
            if meta_data and 'action_history' in meta_data:
                ah = meta_data['action_history']
                if isinstance(ah, list) and len(ah) >= 2:
                    last_action_text = ah[-1]
                    # Count consecutive same action strings from the end
                    ref = ah[-1]
                    for s in reversed(ah):
                        if s == ref:
                            repeated_count += 1
                        else:
                            break
                    if repeated_count >= 2:
                        self.logger.debug(f"[StuckDetection] Action repeated {repeated_count} times: {last_action_text[:100]}")
            
            # If no progress or repeated actions, craft feedback
            if (url_same and img_same) or repeated_count >= 2:
                parts = []
                if url_same and img_same:
                    parts.append("The last action did not change the page (URL and screenshot unchanged).")
                if repeated_count >= 2 and last_action_text:
                    parts.append(f"The action was repeated {repeated_count} times: {last_action_text}")
                if meta_data and 'error_message' in meta_data and meta_data['error_message']:
                    parts.append(f"Environment feedback: {meta_data['error_message']}")
                parts.append("Do NOT repeat the same action/target. Try a different strategy (e.g., go_back to the previous page, click a different element, scroll, type into inputs, press Enter, or use content_analyzer).")
                return " ".join(parts)
        except Exception as e:
            self.logger.error(f"[StuckDetection] Error in detection: {e}")
            return None
        return None
    
    def _generate_action_history_summary(self, intent: str, action_history: List[str], trajectory: Trajectory) -> Optional[str]:
        """Generate a summary and reflection on the last 5 actions using tool LLM."""
        if not action_history or not self.tool_llm or not trajectory:
            return None
        recent_obs = [i['observation']['image'] for i in trajectory if 'observation' in i][-5:]
        recent_actions = action_history[-5:]
        # IMPROVED PROMPT: Focus on Quality and Progress
        prompt = f"""You are a strict coach speaking DIRECTLY to a GUI agent.
        
        USER INTENT: {intent}
        
        Analyze YOUR recent actions (Screenshots + Actions).
        
        Evaluate YOUR WORK QUALITY:
        1. **Effectiveness**: Did your actions actually change the state or move closer to the goal?
        2. **Errors**: Are you hallucinating elements or getting stuck?
        3. **Guidance**: If your work quality is low, what SPECIFIC distinct action should you take next?
        
        CRITICAL FORMATTING RULE:
        - You MUST use "You" or "Your".
        - NEVER use "The agent".
        
        Response Format:
        "Status: [Progressing / Stuck / Regressing]
        Critique: [Your evaluation, e.g., 'You have clicked...']
        Correction: [Specific instruction, e.g., 'You must go back...']"
        """
        messages = [{'role': 'system', 'content': prompt}]
        for obs, action in zip(recent_obs, recent_actions):
            messages.append(
                {'role': 'user', 'content': [
                    {"type": "text", "text": "The following is the task for the agent to accomplish: " + intent},
                    {"type": "text", "text": "The following is the screenshot: "},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{obs}"}},
                    {"type": "text", "text": "The following is the corresponding action: " + action}
                ]
            })
        response, _, _ = self.tool_llm.chat(messages=messages, stream=False)
        return response.content
    
    def _prepare_messages(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> List[Dict]:
        """Prepare messages for the LLM with ReAct context"""
        messages = []
        
        # Extract and store clean_intent for retrieval (avoid instruction/error pollution)
        if self.clean_intent is None and meta_data and 'clean_intent' in meta_data:
            self.clean_intent = meta_data['clean_intent']
            self.logger.info(f"[DynamicMemory] Stored clean intent: {self.clean_intent[:80]}...")
        
        # Generate reflection early to inject into system prompt
        history_summary = None
        if meta_data and 'action_history' in meta_data and self.args.use_history:
            self.logger.info("[Reflexion] use_history=True, generating action history reflection...")
            action_history = meta_data['action_history']
            history_summary = self._generate_action_history_summary(intent, action_history, trajectory)
            if history_summary:
                meta_data['step_history_reflection'] = history_summary
                self.logger.info(f"[Reflexion] Generated reflection:\n{history_summary}")
            else:
                self.logger.info("[Reflexion] No reflection generated (insufficient history)")

        # Early stuck detection and flag for system prompt injection
        is_stuck = False
        status_note = None
        early_feedback = None
        try:
            early_feedback = self._detect_repetition_and_no_progress(trajectory, meta_data)
            if early_feedback:
                status_note = early_feedback
                is_stuck = True
        except Exception:
            pass

        # Also consider the reflection content for stuck indications
        try:
            if (not is_stuck) and history_summary and any(k in history_summary.lower() for k in ['status: stuck', 'status: regressing', 'stuck', 'regressing']):
                is_stuck = True
        except Exception:
            pass

        # ================================================================
        # DYNAMIC MEMORY CHECKPOINT
        # ================================================================
        # Check if memory update is needed based on before/after screenshots
        use_dynamic = getattr(self.args, 'use_dynamic_memory_update', False)
        has_retriever = self.graph_memory_retriever is not None
        has_takeaways = self.current_raw_takeaways is not None
        has_trajectory = len(trajectory) >= 2
        
        # Early exit if max updates already reached - skip entire checkpoint (saves VLM call)
        max_updates = getattr(self.args, 'max_memory_updates', 3)
        if use_dynamic and self.memory_update_count >= max_updates:
            self.logger.info(f"[DynamicMemory] Skipping checkpoint: already reached max updates ({self.memory_update_count}/{max_updates})")
            use_dynamic = False  # Disable for this step
        
        if use_dynamic:
            self.logger.info(f"[DynamicMemory] Check conditions: has_retriever={has_retriever}, has_takeaways={has_takeaways}, has_trajectory={has_trajectory}")
        
        if use_dynamic and has_retriever and has_takeaways and has_trajectory:
            
            try:
                # Trajectory alternates: [obs, action, obs, action, obs, ...]
                # We need to find the last TWO observations (not actions)
                observations = [
                    entry for entry in trajectory 
                    if isinstance(entry, dict) and 'observation' in entry
                ]
                
                self.logger.info(f"[DynamicMemory] trajectory length: {len(trajectory)}, observations found: {len(observations)}")
                
                before_img = None
                after_img = None
                
                if len(observations) >= 2:
                    prev_obs = observations[-2]  # Second-to-last observation
                    curr_obs = observations[-1]  # Last observation
                    
                    before_img = prev_obs['observation'].get('image')
                    after_img = curr_obs['observation'].get('image')
                    
                    self.logger.info(f"[DynamicMemory] before_img exists: {before_img is not None}, after_img exists: {after_img is not None}")
                else:
                    self.logger.info(f"[DynamicMemory] Not enough observations yet ({len(observations)} < 2)")
                
                # Get last action from history
                last_action = ''
                if meta_data and 'action_history' in meta_data:
                    ah = meta_data['action_history']
                    if isinstance(ah, list) and len(ah) > 0:
                        last_action = ah[-1]
                
                if before_img and after_img:
                    update_needed, reason, new_focus, preserved_indices = self._check_memory_update_needed(
                        intent=intent,
                        last_action=last_action,
                        before_image=before_img,
                        after_image=after_img,
                    )
                    
                    if update_needed and new_focus:
                        self._update_graph_memory(
                            intent=intent,
                            new_focus=new_focus,
                            current_image=after_img,
                            reason=reason,
                            preserved_indices=preserved_indices,
                        )
                        self.logger.info("[DynamicMemory] Memory updated, will use new guidance in this step")
                else:
                    self.logger.warning(f"[DynamicMemory] Skipping checkpoint: before_img={before_img is not None}, after_img={after_img is not None}")
            except Exception as e:
                self.logger.warning(f"[DynamicMemory] Checkpoint failed: {e}")
                import traceback
                self.logger.warning(f"[DynamicMemory] Traceback: {traceback.format_exc()}")

        # Add system message; always include reflection if available, but only include stuck status/constraint when stuck
        messages.append({
            'role': 'system',
            'content': self._get_system_message(
                intent,
                trajectory,
                reflection=history_summary,
                status_note=status_note if is_stuck else None
            )
        })
        
        # Add current intent with ReAct prompt
        current_task = intent
        
        # Add analysis results context if available
        if self.last_analysis_result:
            analysis_summary = self.last_analysis_result
            messages.append({
                'role': 'user',
                'content': f"**Content analysis results:** {analysis_summary}"
            })
            # Clear the analysis result after using it
            self.last_analysis_result = None
        
        # Add web search results context if available
        if self.last_web_search_result:
            web_search_summary = self.last_web_search_result
            # Create content with text and screenshots if available
            if self.last_web_search_screenshots:                
                content_items = [
                    {"type": "text", "text": f"**Web search results:** {web_search_summary}"}
                ]
                # Add screenshot images and collect paths for cleanup
                screenshot_files_to_delete = []
                for screenshot_path in self.last_web_search_screenshots:
                    if os.path.exists(screenshot_path):
                        # Convert image to base64
                        import base64
                        with open(screenshot_path, 'rb') as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            content_items.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            })
                        screenshot_files_to_delete.append(screenshot_path)
                
                messages.append({
                    'role': 'user',
                    'content': content_items
                })
                
                # Delete screenshot files after adding them to messages
                for screenshot_path in screenshot_files_to_delete:
                    try:
                        os.remove(screenshot_path)
                    except Exception as e:
                        continue
            else:
                messages.append({
                    'role': 'user',
                    'content': f"**Web search results:** {web_search_summary}"
                })
            # Clear the web search result after using it
            self.last_web_search_result = None
            self.last_web_search_screenshots = None
        
        # Add page goto results context if available
        if self.last_page_goto_result:
            page_goto_summary = f"Successfully navigated to {self.last_page_goto_name} at {self.last_page_goto_result}"
            messages.append({
                'role': 'user',
                'content': f"**Page navigation results:** {page_goto_summary}"
            })
            
            # Clear the page goto result after using it
            self.last_page_goto_result = None
            self.last_page_goto_name = None
        
        # Add action history (Reflection already generated and injected into system prompt, just check actions left)
        if meta_data and 'action_history' in meta_data:
            action_number_left = getattr(self.args, 'max_steps', 15) - len(meta_data['action_history'])
            if action_number_left > 0:
                messages.append({
                    'role': 'user',
                    'content': f"ACTION NUMBER LEFT: You have **{action_number_left} actions left**, You MUST finish the task within the remaining actions! If the left action number is 1, YOU MUST yield the STOP action and provide the answer!"
                })
            # Inject stuck/repetition feedback when no progress or repeated actions are detected
            feedback = early_feedback
            if feedback and not is_stuck:
                self.logger.warning(f"[StuckDetection] Detected issue: {feedback}")
                messages.append({
                    'role': 'user',
                    'content': f"Feedback: {feedback}"
                })
            else:
                if not feedback:
                    self.logger.info("[StuckDetection] No stuck/repetition detected")
                
            # Note: visual history (image, action) pairs are intentionally not injected to reduce context.
                
        # Add experience action suggestions
        if self.args.experience_action_suggestions:
            experience_action_suggestions = self.trajectory_analyzer.get_action_recommendations( f"data:image/png;base64,{trajectory[-1]['observation']['image']}")
            meta_data['experience_action_suggestions'] = experience_action_suggestions
            messages.append({'role': 'user','content': [
                {"type": "text", "text": "The following is the experience action suggestions based on the current state: "},
                {"type": "text", "text": experience_action_suggestions}
            ]})
        if self.args.use_concept_memory:
            # Temperary use task query to retrieve conncept, can be updated dynamically
            concept_suggestions = self.selector.select_concepts(intent, self.concept_memory, top_k=5)
            concept_suggestions = render_concepts_as_suggestions(self.concept_memory, concept_suggestions)
            meta_data['concept_suggestions'] = concept_suggestions
            messages.append({'role': 'user','content': [
                {"type": "text", "text": "The following are some suggestions based on the current task: "},
                {"type": "text", "text": concept_suggestions}
            ]})

        # Add World Model step guidance (after first step)
        # Uses DYNAMIC retrieval with current screenshot to find state-relevant examples
        if self.world_model is not None and self.current_step > 0:
            try:
                current_screenshot = self._get_current_screenshot(trajectory)
                action_history = meta_data.get('action_history', [])
                step_guidance = self.world_model.get_step_guidance(
                    task=intent,
                    current_state=current_screenshot,  # Current screenshot for dynamic retrieval
                    action_history=action_history,
                    step_num=self.current_step,
                    domain=getattr(self.args, 'domain', None),
                    dataset=getattr(self.args, 'evaluation_type', None)
                )
                if step_guidance:
                    messages.append({
                        'role': 'user',
                        'content': step_guidance
                    })
                    self.logger.info(f"[WorldModel] Injected step {self.current_step} guidance (dynamic retrieval)")
            except Exception as e:
                self.logger.warning(f"[WorldModel] Failed to get step guidance: {e}")
        # Add recent trajectory information
        if trajectory:
            recent_obs = trajectory[-1]
            if isinstance(recent_obs, dict) and 'observation' in recent_obs:
                obs = recent_obs['observation']
                if 'image' in obs:
                    # Generate a description of the current page using LLM
                    page_description = self._generate_page_description(obs["image"])
                    # Inject hybrid memory phase exemplars if enabled
                    try:
                        if self.hybrid_retriever is not None:
                            topk = getattr(self.args, 'hybrid_k', 3)
                            exemplars = self.hybrid_retriever.retrieve(
                                intent=current_task,
                                image_b64=obs['image'],
                                domain=self.args.domain,
                                k=topk,
                                page_description=page_description
                            )
                            if exemplars:
                                self.last_hybrid_exemplars = exemplars
                                # Build a compact exemplar block
                                lines = ["[Phase Exemplars]"]
                                for ex in exemplars:
                                    priors = ex.get("priors") or {}
                                    priors_str = ", ".join(f"{k}:{v}" for k, v in list(priors.items())[:3])
                                    lines.append(f"- {ex.get('role','')} | {ex.get('summary','')}")
                                    if priors_str:
                                        lines.append(f"  priors: {priors_str}")
                                messages.append({
                                    'role': 'user',
                                    'content': "\n".join(lines)
                                })
                    except Exception as e:
                        self.logger.warning(f"[HybridMemory] retrieval failed: {e}")
                    # Inject reasoning bank hints at the first turn when enabled
                    if getattr(self.args, 'use_reasoning_bank', False) and self.reasoning_bank is not None:
                        try:
                            query_text = f"{self.args.domain}: {current_task}\n{page_description}"
                            top_k = getattr(self.args, 'reasoning_top_k', 2)
                            domain_filter = self.args.domain if getattr(self.args, 'reasoning_domain_filter', True) else None
                            
                            # Use multimodal query if bank supports it
                            query_image = obs['image'] if self.reasoning_bank.use_multimodal else None
                            
                            # Pure top-k retrieval without label balancing
                            idx_scores = self.reasoning_bank.retrieve(
                                query_text=query_text, top_k=top_k, domain=domain_filter,
                                query_image_base64=query_image
                            )
                            # If nothing returned, no-op; fallback handled below if needed
                            if not idx_scores:
                                idx_scores = []
                            # Log retrieved indices and quick labels for traceability
                            try:
                                retrieved_info = []
                                for i, score in idx_scores[:top_k]:
                                    it = self.reasoning_bank.items[i] if 0 <= i < len(self.reasoning_bank.items) else {}
                                    retrieved_info.append({
                                        "index": int(i),
                                        "score": float(score),
                                        "label": it.get("label", ""),
                                        "title": it.get("title", "")[:80] if "title" in it else it.get("key_takeaway", "")[:80],
                                        "task_id": it.get("task_id", "")
                                    })
                                self.logger.info(f"[ReasoningBank] retrieved={retrieved_info}")
                            except Exception:
                                pass
                            
                            # Use multimodal hints if available, otherwise text-only
                            if self.reasoning_bank.use_multimodal:
                                hints_content = self.reasoning_bank.format_hints_multimodal(
                                    idx_scores[:top_k], max_images_per_hint=2
                                )
                                if hints_content:
                                    self.logger.info(f"[ReasoningBank] injected multimodal hints (count={len(hints_content)})")
                                    # Log full key takeaways and image paths for debugging
                                    try:
                                        for i, score in idx_scores[:top_k]:
                                            it = self.reasoning_bank.items[i] if 0 <= i < len(self.reasoning_bank.items) else {}
                                            takeaway = it.get("key_takeaway", "")
                                            img_path = it.get("after_image_path") or it.get("state_image_path", "")
                                            self.logger.info(f"  [{i}] {it.get('label', '')} | {takeaway}")
                                            self.logger.info(f"       image: {img_path}")
                                    except Exception:
                                        pass
                                    messages.append({'role': 'user', 'content': hints_content})
                            else:
                                hints_text = self.reasoning_bank.format_hints(idx_scores[:top_k])
                                if hints_text:
                                    # Log the final hint text injected to the prompt
                                    self.logger.info(f"[ReasoningBank] injected hints:\n{hints_text}")
                                    messages.append({'role': 'user', 'content': hints_text})
                        except Exception as e:
                            self.logger.warning(f"[ReasoningBank] injection failed: {e}")
                            pass
                    
                    # Add SoM legend if available
                    if 'content_str' in obs:
                        messages.append({
                            'role': 'user',
                            'content': f"**SoM Element Legend** (use these numeric IDs for element_id when possible):\n{obs['content_str']}"
                        })
                    
                    # Add the current screenshot with generated description
                    messages.append({
                        'role': 'user',
                        'content': [
                            {"type": "text", "text": "The following is the current page description and screenshot: " + page_description},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{obs['image']}"
                                }
                            } 
                        ]
                    })
            reminders_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "reminders.txt")
            reminders_content = ""
            if os.path.exists(reminders_path):
                with open(reminders_path, 'r') as f:
                    reminders_content = f.read()
            messages.append({
                'role': 'user',
                'content': reminders_content + f"""**Current task:** {current_task}\nWhat would you like to do next?"""})
        
        return messages, meta_data
    
    def _process_response(self, response: str, trajectory: Trajectory, page: Optional[Any] = None, intent: Optional[str] = None) -> Action:
        """Process the LLM response and convert to Action"""
        manual_action = getattr(self.args, 'manual_action', False)
        if not manual_action:
            # ############### Case for UI-Ins-7B ###############
            # if '<think>' in response and '</think>' in response:
            #     # Find positions
            #     first_end_think = response.find('</think>')
            #     second_start_think = response.find('<think>', first_end_think)
            #     # Extract content between first </think> and second <think>
            #     if first_end_think != -1 and second_start_think != -1:
            #         response = response[first_end_think + len('</think>'):second_start_think].strip()
            #     # Handle case where there's no second <think>
            #     elif first_end_think != -1:
            #         response = response[first_end_think + len('</think>'):].strip()
            # ############### Case for UI-Ins-7B ###############
            parsed_response = parse_action_json(response)
            try:
                func_name = parsed_response['function_call']['name']
                func_args = parsed_response['function_call']['arguments']
                
                # Heuristic: if the intent is to navigate back but model returned a click with "go back" semantics, coerce to go_back
                try:
                    desc_lower = str(func_args.get('description', '')).lower()
                    if func_name in ['click', 'selection'] and any(phrase in desc_lower for phrase in [
                        'go back', 'back to', 'navigate back', 'return to previous page', 'previous page'
                    ]):
                        return create_go_back_action()
                except Exception:
                    pass
                
                # Handle different function calls using the actions module
                if func_name == 'click':
                    return create_click_action(
                        element_id=func_args.get('element_id', ''),
                        coords=func_args.get('coords', ''),
                        description=func_args.get('description', ''),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name in ['selection', 'select']:
                    return create_selection_action(
                        element_id=func_args.get('element_id', ''),
                        coords=func_args.get('coords', ''),
                        description=func_args.get('description', ''),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name in ['type', 'search']:
                    return create_type_action(
                        text=func_args.get('text', ''),
                        element_id=func_args.get('element_id', ''),
                        coords=func_args.get('coords', ''),
                        field_description=func_args.get('field_description', ''),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'scroll':
                    return create_scroll_action(
                        direction=func_args.get('direction', 'down'),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'wait':
                    return create_wait_action(
                        seconds=2.0,  # Default as requested
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'go_back':
                    return create_go_back_action()
                elif func_name == 'press_key':
                    return create_key_press_action(
                        key_comb=func_args.get('key', 'enter'),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'stop':
                    # Check for 'answer', 'reason', or 'reasoning' fields
                    answer = func_args.get('answer') or func_args.get('reason') or func_args.get('reasoning') or 'Task completed'
                    self.logger.info(f"Agent answer: {answer}")
                    return create_stop_action(
                        answer=answer,
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'content_analyzer':
                    # Execute content analyzer and store results for next step
                    # Get the tool from function_map
                    tool = self.function_map.get(func_name)
                    if tool:
                        # Add trajectory context, page (if available), and LLM to kwargs
                        kwargs = {'page': page}
                        tool.llm = self.tool_llm

                        # Handle async call
                        import asyncio
                        import concurrent.futures

                        # Check if event loop is already running
                        try:
                            # Try to get running loop
                            asyncio.get_running_loop()
                            # Loop is running, use ThreadPoolExecutor
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, tool.call(json.dumps(func_args), **kwargs))
                                result = future.result(timeout=120)  # 2 minute timeout
                        except RuntimeError:
                            # No loop running, can use asyncio.run directly
                            result = asyncio.run(tool.call(json.dumps(func_args), **kwargs))
                        # self.logger.info(f"Content analyzer result: {result}")
                        
                        # Store the analysis result for next step context
                        # ContentAnalyzerTool returns JSON string, so store it directly
                        self.last_analysis_result = result
                        
                        # Return a wait action to allow the agent to process the analysis result
                        return create_wait_action(
                            seconds=1.0,
                            reasoning=f"Content analysis completed. Analysis results will be available for the next step."
                        )
                    else:
                        self.logger.error(f"Tool {func_name} not found in function_map")
                        return create_none_action()
                    
                elif func_name == 'map_search':
                    # Execute map search tool which now returns a Google Maps URL
                    tool = self.function_map.get(func_name)
                    if tool:
                        result = tool.call(json.dumps(func_args))
                        # Expect result to be a URL string; try to extract
                        url = result.strip()
                        # Store context
                        self.last_map_search_query = func_args.get('query', '')
                        self.last_map_search_result = result
                        # If we got a URL, emit a goto action so the env updates the page
                        return create_goto_url_action(url)
                        
                elif func_name == 'goto_url':
                    # Execute page goto tool which returns a target URL
                    tool = self.function_map.get(func_name)
                    if tool:
                        tool.llm = self.tool_llm
                        result = tool.call(json.dumps(func_args))
                        # Expect result to be a URL string
                        url = result.strip()
                        # Store context
                        self.last_page_goto_name = func_args.get('page_name', '')
                        self.last_page_goto_result = result
                        # If we got a URL, emit a goto action so the env updates the page
                        return create_goto_url_action(url)
                        
                elif func_name == 'google_web_search':
                    return create_type_action(
                        text=func_args.get('text', ''),
                        element_id=func_args.get('element_id', ''),
                        coords=func_args.get('coords', ''),
                        field_description=func_args.get('field_description', 'search input field'),
                        reasoning=func_args.get('reasoning', '')
                    )
                    # Execute web search tool and store results for next step
                    tool = self.function_map.get(func_name)
                    if tool:
                        # Set the LLM for the web search tool
                        tool.set_llm(self.tool_llm)
                        
                        # Handle async call
                        import asyncio
                        import concurrent.futures
                        
                        # Check if event loop is already running
                        try:
                            # Try to get running loop
                            asyncio.get_running_loop()
                            # Loop is running, use ThreadPoolExecutor
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, tool.call(json.dumps(func_args)))
                                result = future.result(timeout=120)  # 2 minute timeout
                        except RuntimeError:
                            # No loop running, can use asyncio.run directly
                            result = asyncio.run(tool.call(json.dumps(func_args)))
                        # self.logger.info(f"Web search result: {result}")
                        
                        # Extract screenshot information from result
                        screenshot_paths = []
                        if result and "[Screenshot available:" in result:
                            import re
                            screenshot_matches = re.findall(r'\[Screenshot available: ([^\]]+)\]', result)
                            screenshot_paths = screenshot_matches
                            self.logger.info(f"Found screenshots: {screenshot_paths}")
                        
                        # Store the web search result and screenshots for next step context
                        self.last_web_search_result = result
                        self.last_web_search_screenshots = screenshot_paths
                        
                        # Return a wait action to allow the agent to process the search result
                        return create_wait_action(
                            seconds=1.0,
                            reasoning=f"Web search completed. Search results will be available for the next step."
                        )
            except Exception as e:
                print('*'*50, 'no function call', '*'*50)
        # If no function call, try to parse natural language responses
        print(f"Parsing natural language content: {response}")
        # If pattern matching fails, use LLM to parse the content
        action = self._parse_natural_language_with_llm(response, page)
        if action and action.get('action_type') != ActionTypes.NONE:
            print(f"Successfully parsed natural language action with LLM: {action}")
            return action
        
        # If no function call and no natural language action found, return none action
        action = create_none_action()

        return action
    

    def _parse_natural_language_with_llm(self, content: str, page: Optional[Any] = None, pure_text=False) -> Action:
        """Use LLM to parse natural language content and extract action information"""
        try:
            # Create a prompt for the LLM to parse the content
            system_prompt = """You are an expert at parsing natural language responses and converting them into structured actions for a GUI automation agent.

Available actions:
- click: Click on elements by describing what you want to click
- selection: Select an option from a dropdown menu by describing what you want to select
- type: Type text into input fields by describing the field
- press_key: Press specific keys (enter, delete, space, etc.)
- go_back: Navigate back to the previous page
- scroll: Scroll the page in different directions (up, down, left, right)
- wait: Wait for a specified number of seconds
- stop: Stop the task and provide final answer
- map_search: Navigate to Google Maps for geographical searches
- content_analyzer: Analyze page content and images (results will be available for next step)

Parse the given content and return a key-value list with the following structure:

action_type: click|selection|type|press_key|go_back|scroll|wait|stop|map_search|content_analyzer
element_id: id of the element to interact with (for click, selection and type action)
coords: coordinates of the element to interact with (for click, selection and type action), in the format of "<point>x1 y1</point>", and it should be valid with two numbers, without any other text!
description: description of what to click or select (for click and selection action)
text: text to type (for type action)
field_description: description of the field (for type action)
key: key to press (for press_key action)
direction: scroll direction (for scroll action)
seconds: number of seconds (for wait action)
answer: final answer (for stop action)
query: query to search (for map_search action)
reasoning: why this action is needed

EXAMPLES:

For a click action:

action_type: click
element_id: x
coords: <point>x1 y1</point>
description: the search button
reasoning: Need to click the search button to submit the query

For a selection action:

action_type: selection
element_id: x
coords: <point>x1 y1</point>
description: the price option of the dropdown menu want to select from
reasoning: Need to select the price option of the dropdown menu to select the option

For a type action:

action_type: type
element_id: x
coords: <point>x1 y1</point>
text: Sydney Opera House
field_description: the search input field
reasoning: Need to type the search query into the search field

For a press_key action:

action_type: press_key
key: enter
reasoning: Need to press the enter key to submit the query

For a scroll action:

action_type: scroll
direction: down
reasoning: Need to scroll down to load more content


For a wait action:

action_type: wait
seconds: 2.0
reasoning: Need to wait for 2 seconds to load the page


For a stop action (task completion):

action_type: stop
answer: Yes, there is a Ferris wheel in the center of Shanghai. It is the Sky Ring Ferris Wheel in Joy City Shanghai.
reasoning: The information confirms the presence of the Sky Ring Ferris Wheel in the center of Shanghai

For a map_search action:

action_type: map_search
query: Sydney Opera House
reasoning: Need to search for the Sydney Opera House on Google Maps

For a content_analyzer action:

action_type: content_analyzer
reasoning: Need to analyze the page content and images


IMPORTANT: 
1. If the content indicates that a task is complete or provides a final answer, use action_type "stop" with the answer.
2. Ensure all key-value pairs are on separate lines.
3. Values should not contain extra quotes.
4. If there are multiple actions, only output the first action.

REMEMBER: Output ONLY valid key-value pairs, nothing else."""

            user_prompt = f"Parse this content and extract the action:\n\n{content.split('assistant')[0]}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call the LLM
            result = ''
            while result == '':
                response, _, _ = self.tool_llm.chat(messages=messages, stream=False)
                result = response.content
                print(f"LLM response: {result}")
                if result != '':
                    break
            
            # Extract individual fields
            action_data = {}
            key_list = ['action_type', 'element_id', 'reasoning', 'coords',
                        'description', 
                        'text', 'field_description', 
                        'page_name', 'url',
                        'query',
                        'key', 'direction', 'seconds', 'answer']
            for line in result.strip().splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    if key.strip() in key_list:
                        action_data[key.strip()] = value.strip()
            
            # Check if we found any action data
            if not action_data or 'action_type' not in action_data:
                try:
                    action_data = json.loads(result)
                    action_data['action_type'] = action_data['name']
                    for key, value in action_data['arguments'].items():
                        action_data[key] = value
                except:
                    print("No valid action data found in LLM response")
                    return create_none_action()
            
            # print(f"Extracted action data: {action_data}")
            
            # Convert to Action based on action_type
            action_type = action_data.get('action_type', '').lower()
            reasoning = action_data.get('reasoning', '')
            
            if pure_text:
                return {'name': action_data['action_type'], 'arguments': {'reasoning': action_data['reasoning'],
                                                                          'description': action_data.get('description', '')}}
            
            # Heuristic: coerce to go_back when NL indicates back navigation but action parsed as click/selection
            try:
                desc_lower = str(action_data.get('description', '')).lower()
                if action_type in ['click', 'selection'] and any(phrase in desc_lower for phrase in [
                    'go back', 'back to', 'navigate back', 'return to previous page', 'previous page'
                ]):
                    return create_go_back_action()
            except Exception:
                pass
            
            if action_data['action_type'] == 'goto_url' and 'url' not in action_data:
                page_name = action_data.get('page_name', '')
                page_urls = {
                        'wiki': 'https://www.wikipedia.org/',
                        'ticket': 'https://www.trip.com/flights/',
                        'car': 'https://sg.trip.com/carhire/?channelid=14409&locale=en-SG&curr=USD',
                        'flight': 'https://www.momondo.com/',
                        'hotel': 'https://sg.trip.com/hotels/?locale=en-SG&curr=USD',
                        'shopping': 'http://ec2-3-146-212-252.us-east-2.compute.amazonaws.com:7770/',
                        'event': 'https://www.eventbrite.com/',
                        'map': 'https://www.google.com/maps',
                        'youtube': 'https://www.youtube.com/',
                        'food': 'https://www.timeout.com/',
                        'travel': 'https://www.nomadicmatt.com/',
                        'dollars': 'https://www.xe.com/',
                        'twitter': 'https://twitter.com/home',
                    }
                if 'car' in page_name:
                    page_name = 'car'
                elif 'wiki' in page_name:
                    page_name = 'wiki'
                elif 'ticket' in page_name:
                    page_name = 'ticket'
                elif 'flight' in page_name:
                    page_name = 'flight'
                elif 'hotel' in page_name:
                    page_name = 'hotel'
                elif 'shopping' in page_name:
                    page_name = 'shopping'
                elif 'event' in page_name:
                    page_name = 'event'
                elif 'map' in page_name:
                    page_name = 'map'
                elif 'youtube' in page_name:
                    page_name = 'youtube'
                elif 'food' in page_name:
                    page_name = 'food'
                elif 'travel' in page_name:
                    page_name = 'travel'
                elif 'dollars' in page_name:
                    page_name = 'dollars'
                elif 'twitter' in page_name:
                    page_name = 'twitter'
                try:
                    action_data['url'] = page_urls[page_name]
                except:
                    action_data['url'] = ''
            
            if action_type == 'click':
                return create_click_action(
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    description=action_data.get('description', ''),
                    reasoning=reasoning
                )
            elif action_type in ['selection', 'select']:
                return create_selection_action(
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    description=action_data.get('description', ''),
                    reasoning=reasoning
                )
            elif action_type in ['type', 'search']:
                return create_type_action(
                    text=action_data.get('text', ''),
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    field_description=action_data.get('field_description', ''),
                    reasoning=reasoning
                )
            elif action_type == 'press_key':
                return create_key_press_action(
                    key_comb=action_data.get('key', 'enter'),
                    reasoning=reasoning
                )
            elif action_type == 'scroll':
                return create_scroll_action(
                    direction=action_data.get('direction', 'down'),
                    reasoning=reasoning
                )
            elif action_type == 'wait':
                return create_wait_action(
                    seconds=float(action_data.get('seconds', 2.0)),
                    reasoning=reasoning
                )
            elif action_type == 'go_back':
                return create_go_back_action()
            elif action_type == 'stop':
                # Check for 'answer', 'reason', or 'reasoning' fields
                answer = action_data.get('answer') or action_data.get('reason') or action_data.get('reasoning') or 'Task completed'
                return create_stop_action(
                    answer=answer,
                    reasoning=reasoning
                )
            elif action_type == 'map_search':
                tool = self.function_map.get(action_type)
                if tool:
                    func_args = {
                        'query': action_data.get('query', action_data.get('reasoning', '')),
                        'reasoning': action_data.get('reasoning', '')
                    }
                    result = tool.call(json.dumps(func_args))
                    # Expect result to be a URL string; try to extract
                    url = result.strip()
                    # Store context
                    self.last_map_search_query = func_args.get('query', '')
                    self.last_map_search_result = result
                    # If we got a URL, emit a goto action so the env updates the page
                    return create_goto_url_action(url)
            elif action_type == 'goto_url':
                return create_goto_url_action(
                    url=action_data.get('url', '')
                )
            elif action_type == 'google_web_search':
                return create_type_action(
                    text=action_data.get('text', ''),
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    field_description=action_data.get('field_description', 'search input field'),
                    reasoning=reasoning
                )
                # Set the LLM for the web search tool
                tool = self.function_map.get(action_type)
                if tool:
                    tool.set_llm(self.tool_llm)
                    # Handle async call
                    import asyncio
                    import concurrent.futures
                    
                    # Check if event loop is already running
                    try:
                        # Try to get running loop
                        asyncio.get_running_loop()
                        # Loop is running, use ThreadPoolExecutor
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, tool.call(json.dumps(func_args)))
                            result = future.result(timeout=120)  # 2 minute timeout
                    except RuntimeError:
                        # No loop running, can use asyncio.run directly
                        result = asyncio.run(tool.call(json.dumps(func_args)))
                    # self.logger.info(f"Web search result: {result}")
                    
                    # Extract screenshot information from result
                    screenshot_paths = []
                    if result and "[Screenshot available:" in result:
                        import re
                        screenshot_matches = re.findall(r'\[Screenshot available: ([^\]]+)\]', result)
                        screenshot_paths = screenshot_matches
                        self.logger.info(f"Found screenshots: {screenshot_paths}")
                    
                    # Store the web search result and screenshots for next step context
                    self.last_web_search_result = result
                    self.last_web_search_screenshots = screenshot_paths
                    
                    # Return a wait action to allow the agent to process the search result
                    return create_wait_action(
                        seconds=1.0,
                        reasoning=f"Web search completed. Search results will be available for the next step."
                    )
            elif action_type == 'content_analyzer':
                tool = self.function_map.get(action_type)
                func_args = {
                    'query': action_data.get('query', ''),
                    'reasoning': action_data.get('reasoning', '')
                }
                if tool:
                    # Add trajectory context, page (if available), and LLM to kwargs
                    kwargs = {'page': page}
                    tool.llm = self.tool_llm

                    # Handle async call
                    import asyncio
                    import concurrent.futures

                    # Check if event loop is already running
                    try:
                        # Try to get running loop
                        asyncio.get_running_loop()
                        # Loop is running, use ThreadPoolExecutor
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, tool.call(json.dumps(func_args), **kwargs))
                            result = future.result(timeout=120)  # 2 minute timeout
                    except RuntimeError:
                        # No loop running, can use asyncio.run directly
                        result = asyncio.run(tool.call(json.dumps(func_args), **kwargs))
                    # self.logger.info(f"Content analyzer result: {result}")
                    
                    # Store the analysis result for next step context
                    # ContentAnalyzerTool returns JSON string, so store it directly
                    self.last_analysis_result = result
                    
                    # Return a wait action to allow the agent to process the analysis result
                    return create_wait_action(
                        seconds=1.0,
                        reasoning=f"Content analysis completed. Analysis results will be available for the next step."
                    )
            else:
                return create_none_action()
            
        except Exception as e:
            print(f"Error in LLM parsing: {e}")
            return create_none_action()

    
    def _generate_page_description(self, image_base64: str) -> str:
        """Generate a description of the current page using the LLM"""
        try:
            # Create a prompt for the LLM to describe the page
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that analyzes web page screenshots and provides clear, concise descriptions of what you see. Focus on the main content, interactive elements, and overall purpose of the page.'
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please describe this web page screenshot. Include the main content, any visible buttons, forms, or interactive elements, and the overall purpose of the page."
                        }
                    ]
                }
            ]
            
            # Get LLM response
            response, _, _ = self.tool_llm.chat(messages=messages, stream=False)
            if hasattr(response, 'content'):
                description = response.content
            else:
                description = str(response)
            
            description = description.replace("\"text\": \"{'role': 'assistant', 'content': '", "")
            description = description.replace("'}\"", "")
            description = description[:2000]
            return description if description else "Current page state - analyze this and decide what to do next"
            
        except Exception as e:
            self.logger.warning(f"Error generating page description: {e}")
            return "Current page state - analyze this and decide what to do next"
    
    # =========================================================================
    # Self-Evolving Graph Memory Methods
    # =========================================================================
    
    def _generate_takeaway_for_graph(
        self,
        task: str,
        actions_text: str,
        image_b64: str = None,
    ) -> str:
        """
        Generate a takeaway summary for a successful trajectory.
        
        Args:
            task: The task description
            actions_text: Formatted actions text
            image_b64: First screenshot (base64)
            
        Returns:
            Takeaway string starting with "takeaway:"
        """
        system = (
            "You extract actionable heuristics from SUCCESSFUL GUI agent trajectories.\n"
            "Return EXACTLY 1 sentence starting with 'takeaway:' in this format:\n"
            "takeaway: <ONE concise actionable heuristic>\n"
            "Constraints:\n"
            "- Focus on WHAT strategy worked (not what the agent did).\n"
            "- Must start with 'takeaway:' (exact substring).\n"
            "- Keep it under 20 words.\n"
            "- No quotes, no bullet points, no step-by-step narration."
        )
        user_text = f"Task: {task}\nActions:\n{actions_text}"
        
        if image_b64:
            # Handle both raw base64 and data URL formats
            if not image_b64.startswith("data:image"):
                image_b64 = f"data:image/png;base64,{image_b64}"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ]},
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ]
        
        resp, _, _ = self.tool_llm.chat(messages=messages, stream=False, temperature=0.0, max_tokens=128)
        summary = resp.content.strip()
        
        # Ensure it starts with "takeaway:"
        if not summary.lower().startswith("takeaway:"):
            summary = f"takeaway: {summary}"
        
        self.logger.info(f"[SelfEvolve] Generated takeaway: {summary}")
        return summary
    
    def _add_successful_trajectory_to_graph(
        self,
        trajectory_file: str,
        task_description: str,
        domain: str,
    ) -> bool:
        """
        Add a successful trajectory to the live graph memory for self-evolving.
        
        Called after task success to enable online learning.
        
        Args:
            trajectory_file: Path to saved trajectory JSONL
            task_description: The task intent
            domain: The evaluation domain (e.g., "Amazon")
        
        Returns:
            True if successfully added, False otherwise
        """
        if self.graph_memory is None or self.graph_memory_retriever is None:
            self.logger.warning("[SelfEvolve] Graph memory not initialized, skipping")
            return False
        
        try:
            from graph_memory import TaggedTrajectory
            from memory.help_functions import CLIPMultimodalSimilarity
            import re
            
            # Load the saved trajectory
            with open(trajectory_file, 'r') as f:
                traj_data = json.load(f)
            
            rounds = traj_data.get('rounds', [])
            if len(rounds) < 1:
                self.logger.info("[SelfEvolve] Empty trajectory, skipping")
                return False
            
            # Extract actions and images from rounds
            actions = []
            images = []
            
            def parse_action_from_response(response: str):
                """Parse action JSON from response string."""
                if not isinstance(response, str):
                    return None
                
                # Try ```json block
                if "```json" in response:
                    match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
                    if match:
                        try:
                            return json.loads(match.group(1))
                        except json.JSONDecodeError:
                            pass
                # Try Action: prefix
                match = re.search(r'Action:\s*(\{.*\})', response)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except json.JSONDecodeError:
                        pass
                # Try direct JSON
                try:
                    obj = json.loads(response)
                    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                        return obj
                except json.JSONDecodeError:
                    pass
                return None
            
            for r in rounds:
                response = r.get('response', '')
                if isinstance(response, list):
                    response = response[0] if response else ''
                if isinstance(response, dict) and 'content' in response:
                    response = response['content']
                
                action = parse_action_from_response(response)
                
                # Extract image from messages
                image = None
                for msg in r.get('messages', []):
                    if isinstance(msg.get('content'), list):
                        for item in msg['content']:
                            if item.get('type') == 'image_url':
                                image = item['image_url']['url']
                                break
                        if image:
                            break
                
                if action and 'name' in action:
                    actions.append(action)
                    if image:
                        images.append(image)
            
            if not actions:
                self.logger.warning("[SelfEvolve] No actions extracted from trajectory")
                return False
            
            # Format actions text for takeaway generation
            actions_text = "\n".join(
                f"- {a['name']}: {a.get('arguments', {}).get('reasoning', '')}" 
                for a in actions[:8]
            )
            
            # Get first image for embedding and takeaway
            first_image = images[0] if images else None
            
            # Generate takeaway using tool LLM
            takeaway = self._generate_takeaway_for_graph(
                task=task_description,
                actions_text=actions_text,
                image_b64=first_image,
            )
            
            # Extract tags
            tags = set()
            if self.graph_memory.tag_extractor:
                tags = self.graph_memory.tag_extractor.extract_tags(takeaway, domain)
            tags.add(f"#{domain.lower()}")
            
            # Compute embedding using CLIP
            embedding_model = CLIPMultimodalSimilarity()
            eval_type = getattr(self.args, 'evaluation_type', '')
            query_text = f"{eval_type}_{domain}: {task_description}" if eval_type else task_description
            embedding = embedding_model.get_multimodal_embeddings([query_text], [first_image])[0]
            
            # Create trajectory ID
            traj_id = f"{domain}--{Path(trajectory_file).stem}"
            
            # Create TaggedTrajectory with full_data for continuous memory
            new_traj = TaggedTrajectory(
                id=traj_id,
                takeaway=takeaway,
                tags=tags,
                embedding=embedding,
                domain=domain,
                full_data={
                    'task_description': task_description,
                    'file_path': trajectory_file,
                    'actions': actions,
                    'images': images,
                },
            )
            
            # Add to graph with deduplication
            if hasattr(self.graph_memory, 'add_trajectory_with_dedup') and self.graph_memory.llm is not None:
                result = self.graph_memory.add_trajectory_with_dedup(new_traj)
                self.logger.info(f"[SelfEvolve] Added trajectory with dedup: {result}")
            else:
                self.graph_memory.add_trajectory(new_traj)
                self.logger.info(f"[SelfEvolve] Added trajectory: {traj_id}")
            
            # Update FAISS index in retriever
            self.graph_memory_retriever.add_trajectory(new_traj)
            
            # Track additions for persistence
            if not hasattr(self, '_self_evolve_addition_count'):
                self._self_evolve_addition_count = 0
            self._self_evolve_addition_count += 1
            
            # Persist graph periodically
            persist_interval = getattr(self.args, 'graph_persist_interval', 5)
            if self._self_evolve_addition_count % persist_interval == 0:
                graph_path = getattr(self.args, 'graph_memory_index_path', None)
                if graph_path:
                    self.graph_memory.save(graph_path)
                    self.logger.info(f"[SelfEvolve] Persisted graph to {graph_path} "
                                    f"(total: {len(self.graph_memory)} trajectories)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[SelfEvolve] Failed to add trajectory: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reset(self, test_config_file: str) -> None:
        """Reset the agent for a new task"""
        self.logger.info(f"Resetting agent for config file: {test_config_file}")
        # Reset world model cache
        if self.world_model is not None:
            self.world_model.reset()
        # Reset step counter
        self.current_step = 0
        # Clear per-task memory state so retrieval/summaries refresh each task.
        self.experience_memory = None
        self.experience_texts = None
        self.experience_images = None
        self.file_id_list = None
        self.last_hybrid_exemplars = None
        self.graph_memory_block = None  # Clear graph memory block for new task

        # Reset dynamic memory update state
        self.current_raw_takeaways = None
        self.memory_update_count = 0
        self.clean_intent = None


def construct_agent(args: argparse.Namespace) -> FunctionCallAgent:
    """Construct a function call agent"""
    agent = FunctionCallAgent(args)
    return agent 