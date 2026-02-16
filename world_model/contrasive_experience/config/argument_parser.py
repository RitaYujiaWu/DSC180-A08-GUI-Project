"""Argument parser configuration for the GUI Agent"""
import argparse

def config() -> argparse.Namespace:
    """Configure and parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    
    # Browser environment arguments
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="image",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument(
        "--imgbin_dir",
        type=str,
        default="",
    ) # Not in use

    # Agent configuration
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=2,
    )
    parser.add_argument("--domain", type=str, default="shopping")
    parser.add_argument("--hist", action='store_true', default=False)
    parser.add_argument("--hist_fold", type=str, default="./cache/history/")
    parser.add_argument("--hist_num", type=int, default=1)

    parser.add_argument("--task_cnt", type=int, default=0)
    parser.add_argument("--hop_cnt", type=int, default=0)
    
    # API key
    parser.add_argument('--open_router_api_key', type=str, default='', help='OpenRouter API key')
    
    # Language model configuration
    parser.add_argument("--provider", type=str, default="custom")
    parser.add_argument("--model", type=str, default="qwen2.5-vl")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional HF repo id or local path for the continuous-memory (QFormer) model checkpoint.",
    )
    parser.add_argument("--grounding_model_name", type=str, default="ui-ins-7b", help="Model name for grounding (default: ui-ins-7b)")
    parser.add_argument("--tool_model_name", type=str, default="qwen2.5-vl", help="Model name for tool LLM (default: qwen2.5-vl)")
    parser.add_argument("--loaded_tokenizer", default=None)
    parser.add_argument("--loaded_model", default=None)
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--context_length", type=int, default=0)
    # Grounding behavior control
    parser.add_argument(
        "--grounding_mode",
        type=str,
        default="auto",
        choices=["auto", "prefer", "force", "off"],
        help="How to use grounding for pixel actions: "
             "auto (coords -> id center -> grounding), "
             "prefer (coords -> grounding -> id center), "
             "force (always call grounding, fallback to coords/id on failure), "
             "off (coords -> id center, never call grounding).",
    )
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--use_history", type=bool, default=False)
    parser.add_argument("--use_continuous_memory", type=bool, default=False)
    parser.add_argument(
        "--use_discrete_memory",
        type=bool,
        default=False,
        help="If True, retrieve similar trajectories, summarize them with a VLM, and inject the summaries into the system prompt.",
    )
    parser.add_argument(
        "--discrete_memory_cache_path",
        type=str,
        default="data/test_hybrid/trajectory_summaries.json",
        help="JSON cache file for discrete memory summaries (keyed by retrieved file_id).",
    )
    parser.add_argument(
        "--discrete_memory_max_actions",
        type=int,
        default=8,
        help="Max number of actions to show the summarizer per trajectory.",
    )
    parser.add_argument(
        "--discrete_memory_use_checkpoint",
        type=bool,
        default=False,
        help="If True, use the fine-tuned checkpoint model (main LLM) for discrete memory summarization/digestion instead of tool_llm.",
    )
    # Action Scaling Suggestion configuration
    parser.add_argument("--experience_action_suggestions", type=bool, default=False)
    parser.add_argument("--trajectory_dir", type=str, default='webvoyager_memory')
    parser.add_argument("--state_embedding_path", type=str, default='action_scaling/state_embeddings/webvoyager_memory.npy')
    # Concept memory configuration
    parser.add_argument("--use_concept_memory", type=bool, default=False)
    parser.add_argument("--concept_memory_path", type=str, default='arc_memo/output/memory.json')
    # Implicit World Model configuration
    parser.add_argument("--use_implicit_world_model", type=bool, default=False)
    parser.add_argument("--faiss_index_path", type=str, default=None)
    parser.add_argument("--similar_num", type=int, default=10)
    parser.add_argument("--bank_size", type=int, default=None)
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument("--add_history_num", type=int, default=5, 
                       help="Whether to add history actions to the prompt")
    parser.add_argument("--save_examples_memory", action='store_true', default=False, 
                       help="Whether to add example memory to the agent")
    parser.add_argument("--instruction_jsons", type=str, nargs='+', default=[], 
                       help="jsons to use for example retrieval")
    
    # Reasoning Bank configuration
    parser.add_argument("--use_reasoning_bank", type=bool, default=False,
                        help="Enable retrieval of distilled reasoning items")
    parser.add_argument("--reasoning_bank_path", type=str, default="memory/reasoning_bank.jsonl",
                        help="Path to reasoning bank JSONL")
    parser.add_argument("--reasoning_top_k", type=int, default=2,
                        help="Number of reasoning items to inject at the first turn")
    parser.add_argument("--reasoning_domain_filter", type=bool, default=True,
                        help="Filter retrieved items by current domain")
    parser.add_argument("--reasoning_index_base", type=str, default="memory_index/reasoning_bank_text",
                        help="Base path (without extension) for the FAISS index of the reasoning bank")
    parser.add_argument("--reasoning_bank_multimodal", type=bool, default=False,
                        help="Use multimodal reasoning bank with key step identification and screenshots")

    # World Model configuration
    parser.add_argument("--use_world_model", action="store_true", default=False,
                        help="Enable World Model for contrastive learning guidance")
    parser.add_argument("--world_model_data_path", type=str, default="training_data",
                        help="Path to trajectory data for World Model")
    parser.add_argument("--world_model_index_path", type=str, default=None,
                        help="Path to pre-built FAISS index for World Model")
    parser.add_argument("--world_model_multimodal", action="store_true", default=True,
                        help="Use multimodal embeddings for World Model trajectory retrieval")
    parser.add_argument("--world_model_top_k", type=int, default=3,
                        help="Number of success/failure trajectories to retrieve for contrastive analysis")
    parser.add_argument("--world_model_step_guidance", action="store_true", default=True,
                        help="Enable per-step guidance from World Model (in addition to initial guidance)")

    # Example configuration
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=10000)

    # Logging related
    parser.add_argument("--result_dir", type=str, default="")
    
    # Training data collection configuration
    parser.add_argument("--collect_training_data", action='store_true', default=False,
                        help="Enable collection of training data (prompts and responses)")
    parser.add_argument("--training_data_dir", type=str, default="training_data",
                        help="Directory to save training data files")
    parser.add_argument("--memory_data_dir", type=str, nargs='+', default=["data/trajectories"],
                        help="One or more directories containing trajectory files for memory retrieval")
    # Subtask decomposition configuration
    parser.add_argument("--subtask", action='store_true', default=False, 
                       help="Enable subtask decomposition for complex task breakdown")
    
    # Evaluation configuration
    parser.add_argument("--evaluation_type", type=str, default="mmina", 
                    #    choices=['mmina', 'supergpqa', 'webwalkerqa','expand_memory',  'visualwebarena', 'webarena', 'webvoyager', 'mind2web'],
                       help="Type of evaluation to run")
    parser.add_argument("--render_screenshot", action='store_true', 
                       help="Render screenshots during evaluation")
    
    # WebWalkerQA evaluation configuration
    parser.add_argument("--webwalkerqa_split", type=str, default="silver", 
                       choices=['main', 'silver'])
    
    # Manual Action Instruction
    parser.add_argument("--manual_action", action='store_true', default=False, 
                       help="Enable manual action instruction for complex task breakdown")
    parser.add_argument("--debug", action='store_true', default=False, 
                       help="Enable debug mode")
    
    parser.add_argument("--datetime", type=str, default=None)
    # Hybrid memory
    parser.add_argument("--use_hybrid_memory", action='store_true', default=False,
                       help="Enable hierarchical hybrid graph-latent memory")
    parser.add_argument("--hybrid_index_dir", type=str, default="hybrid_index",
                       help="Directory for hybrid memory index")
    parser.add_argument("--hybrid_k", type=int, default=3,
                       help="Top-K phase exemplars to retrieve")
    
    # Graph Memory configuration
    parser.add_argument("--use_graph_memory", type=bool, default=False,
                       help="Enable graph memory (FAISS + graph expansion for diversity)")
    parser.add_argument("--graph_memory_index_path", type=str, default=None,
                       help="Path to saved graph index (base path without extension)")
    parser.add_argument("--graph_tag_cache_path", type=str, default="graph_memory_cache/tags.json",
                       help="Path to cache extracted tags")
    parser.add_argument("--graph_expand_hops", type=int, default=1,
                       help="Number of hops to expand in graph traversal")
    parser.add_argument("--graph_initial_seeds", type=int, default=None,
                       help="Number of seeds to retrieve from FAISS in Phase 1. If None, defaults to k//2. Set to 1 to test expansion-heavy retrieval.")
    parser.add_argument("--graph_diversity_weight", type=float, default=0.3,
                       help="Weight for diversity vs similarity (0-1)")
    parser.add_argument("--graph_similar_num", type=int, default=5,
                       help="Number of trajectories to retrieve with graph memory")
    parser.add_argument("--use_dynamic_memory_update", type=bool, default=False,
                       help="Enable dynamic memory update (VLM checkpoint after each action)")
    parser.add_argument("--max_memory_updates", type=int, default=3,
                       help="Maximum number of memory updates per task (default: 3)")
    
    # Self-evolving graph memory configuration
    parser.add_argument("--use_self_evolving_memory", type=bool, default=False,
                       help="Enable self-evolving graph memory (add successful trajectories online)")
    parser.add_argument("--graph_persist_interval", type=int, default=5,
                       help="Persist graph to disk every N successful additions (default: 5)")
    
    args = parser.parse_args()
    args.instruction_path = 'agent/prompts/jsons/p_cot_ground_actree_2s.json'
    if args.use_continuous_memory and not 'full-sft' in args.model:
        args.model = 'agent-qformer'
    if args.datetime is None:
        datetime = 'test'
        args.datetime = datetime
    # Set result directory based on evaluation type
    if not args.result_dir:
        args.result_dir = f'results/{args.evaluation_type}/{args.model}/{args.domain}/{args.datetime}'
        
    # Set training data directory
    args.training_data_dir = f"training_data/{args.evaluation_type}/{args.domain}/{args.model}/{args.datetime}"
    
    return args 
