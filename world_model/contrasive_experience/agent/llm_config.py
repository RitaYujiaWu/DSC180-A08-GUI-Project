"""LLM configuration for different model types"""
import argparse
from typing import Dict, List
import base64
from openai import OpenAI
import os
from transformers import AutoProcessor
import torch
import sys
from PIL import Image
from io import BytesIO
from openai.types.chat import ChatCompletionMessage

class DirectVLLMModel:
    """Direct vLLM model wrapper that can be used without qwen_agent"""
    
    def __init__(self, model_name: str, server_url: str, api_key: str = "EMPTY", **kwargs):
        self.model_name = model_name
        self.server_url = server_url
        self.api_key = api_key
        self.client = OpenAI(
            base_url=server_url,
            api_key=api_key
        )
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_tokens = kwargs.get('max_tokens', 2048)
    
    def chat(self, messages: List[Dict], stream: bool = False, **kwargs):
        """Chat with the model using simplified message format"""
        # Prepare function calling parameters
        call_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get('temperature', self.temperature),
            "top_p": kwargs.get('top_p', self.top_p),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "n": kwargs.get('n', 1),
        }
        
        # Call the model
        response = self.client.chat.completions.create(**call_params)
        
        if stream:
            return response, None, None
        else:
            return response.choices[0].message, None, None

class DirectTransformersModel:
    """Direct Transformers model wrapper for Qwen2.5-VL with experience handling"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.temperature = kwargs.get('temperature', 0.1)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_tokens = 10**4
        self.checkpoint_path = kwargs.get('checkpoint_path', model_name)
        self.args = kwargs.get('args', None)
        # Load processor and tokenizer - use base model for Qwen3-VL checkpoints
        if '3' in self.checkpoint_path and '2_5' not in self.checkpoint_path:
            processor_path = "Qwen/Qwen3-VL-8B-Instruct"
            print(f"Using Qwen3-VL base processor from: {processor_path}")
        else:
            processor_path = self.checkpoint_path
        self.processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        
        # Import the custom model class
        train_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "CoMEM-Agent-train"))
        if not os.path.isdir(train_repo_root):
            raise FileNotFoundError(
                f"Could not find CoMEM-Agent-train at {train_repo_root}. "
                "Expected this repo layout: <root>/CoMEM-Agent-Inference and <root>/CoMEM-Agent-train."
            )
        if train_repo_root not in sys.path:
            sys.path.insert(0, train_repo_root)
        if '2_5' in self.args.checkpoint_path:
            if 'full-sft' in self.args.model:
                print('Using 2_5 full-sft model')
                from src_agent.training.qwenVL_inference_full_sft import Qwen2_5_VLForConditionalGeneration_new
            else:
                print('Using 2_5 normal model')
                from src_agent.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new
            self.model = Qwen2_5_VLForConditionalGeneration_new.from_pretrained(
                self.checkpoint_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                low_cpu_mem_usage=True
            )
        elif '3' in self.args.checkpoint_path:
            print('Using 3 VL model')
            print(f"Loading model from checkpoint: {self.checkpoint_path}")
            from src_agent.training.qwen3VL_compressor import Qwen3VLForConditionalGeneration_new
            self.model = Qwen3VLForConditionalGeneration_new.from_pretrained(
                self.checkpoint_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            if 'full-sft' in self.args.model:
                print('Using full-sft model')
                from src_agent.training.qwenVL_inference_full_sft import Qwen2_5_VLForConditionalGeneration_new
            else:
                print('Using normal model')
                from src_agent.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new
            self.model = Qwen2_5_VLForConditionalGeneration_new.from_pretrained(
                self.checkpoint_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        # Load model_inf weights from checkpoint with key remapping
        # Checkpoint has old structure: model_inf.model.layers.* and model_inf.visual.*
        # Current model expects: model_inf.model.language_model.layers.* and model_inf.model.visual.*
        self._load_model_inf_from_checkpoint()
    
    def _load_model_inf_from_checkpoint(self):
        """Load model_inf and knowledge_processor weights from checkpoint with key remapping.
        
        The checkpoint uses old key structure:
        - model_inf.model.layers.* -> model_inf.model.language_model.layers.*
        - model_inf.model.embed_tokens.* -> model_inf.model.language_model.embed_tokens.*
        - model_inf.model.norm.* -> model_inf.model.language_model.norm.*
        - model_inf.visual.* -> model_inf.model.visual.*
        - knowledge_processor.* -> knowledge_processor.* (Q-Former, no remapping needed)
        """
        import os
        from safetensors import safe_open
        
        # Find all safetensor files in checkpoint
        safetensor_files = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.safetensors')]
        if not safetensor_files:
            print("No safetensor files found in checkpoint, skipping weight loading")
            return
        
        # Collect model_inf and knowledge_processor weights from checkpoint
        checkpoint_model_inf_weights = {}
        checkpoint_knowledge_processor_weights = {}
        for sf_file in safetensor_files:
            sf_path = os.path.join(self.checkpoint_path, sf_file)
            with safe_open(sf_path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    if key.startswith('model_inf.'):
                        checkpoint_model_inf_weights[key] = f.get_tensor(key)
                    elif key.startswith('knowledge_processor.'):
                        checkpoint_knowledge_processor_weights[key] = f.get_tensor(key)
        
        print(f"Found {len(checkpoint_model_inf_weights)} model_inf weights and {len(checkpoint_knowledge_processor_weights)} knowledge_processor weights in checkpoint")
        
        if not checkpoint_model_inf_weights and not checkpoint_knowledge_processor_weights:
            print("No model_inf or knowledge_processor weights found in checkpoint")
            return
        
        # Remap model_inf keys from old structure to new structure
        remapped_weights = {}
        for old_key, tensor in checkpoint_model_inf_weights.items():
            new_key = old_key
            
            if old_key.startswith('model_inf.model.layers.'):
                new_key = old_key.replace('model_inf.model.layers.', 'model_inf.model.language_model.layers.')
            elif old_key.startswith('model_inf.model.embed_tokens'):
                new_key = old_key.replace('model_inf.model.embed_tokens', 'model_inf.model.language_model.embed_tokens')
            elif old_key.startswith('model_inf.model.norm'):
                new_key = old_key.replace('model_inf.model.norm', 'model_inf.model.language_model.norm')
            elif old_key.startswith('model_inf.visual.'):
                new_key = old_key.replace('model_inf.visual.', 'model_inf.model.visual.')
            # model_inf.lm_head stays the same
            
            remapped_weights[new_key] = tensor
        
        # Add knowledge_processor weights (no remapping needed)
        for key, tensor in checkpoint_knowledge_processor_weights.items():
            remapped_weights[key] = tensor
        
        # Load remapped weights into model
        model_state_dict = self.model.state_dict()
        loaded_count = 0
        kp_loaded_count = 0
        missing_keys = []
        
        # Debug: Check if knowledge_processor keys exist in model
        kp_keys_in_model = [k for k in model_state_dict.keys() if 'knowledge_processor' in k]
        print(f"[Debug] Model has {len(kp_keys_in_model)} knowledge_processor keys")
        if kp_keys_in_model:
            print(f"[Debug] First 3 model kp keys: {kp_keys_in_model[:3]}")
        
        for key, tensor in remapped_weights.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == tensor.shape:
                    model_state_dict[key] = tensor.to(model_state_dict[key].dtype)
                    loaded_count += 1
                    if key.startswith('knowledge_processor.'):
                        kp_loaded_count += 1
                        print(f"[Debug] Loaded Q-Former key: {key}")
                else:
                    print(f"Shape mismatch for {key}: checkpoint {tensor.shape} vs model {model_state_dict[key].shape}")
            else:
                missing_keys.append(key)
                if key.startswith('knowledge_processor.'):
                    print(f"[Debug] Q-Former key NOT FOUND in model: {key}")
        
        # Load the updated state dict
        self.model.load_state_dict(model_state_dict, strict=False)
        print(f"Loaded {loaded_count} weights from checkpoint (including {kp_loaded_count} Q-Former weights)")
        if missing_keys:
            print(f"Warning: {len(missing_keys)} keys not found in model (first 5): {missing_keys[:5]}")
        
    def process_vision_info(self, conversation):
        """Process vision information from conversation"""
        image_inputs = []
        
        for message in conversation:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if item['type'] == 'image_url':
                        image_url = item['image_url']['url']
                        image_bytes = base64.b64decode(image_url.split(',')[1])
                        image = Image.open(BytesIO(image_bytes))
                        image_inputs.append(image)
        
        return image_inputs
    
    def knowledge_processor_vlm(self, processor, inputs, texts=None, images=None, tokenizer=None, formatted_prompt=None):
        """Process experience information for VLM"""
        # Default tokens for image processing
        DEFAULT_IM_START_TOKEN = "<|im_start|>"
        DEFAULT_IM_END_TOKEN = "<|im_end|>"
        DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
        VISION_START_TOKEN = "<|vision_start|>"
        VISION_END_TOKEN = "<|vision_end|>"
        
        all_experience_input_ids = [] 
        all_experience_pixel_values = []
        all_experience_image_grid_thw = []
        for trajectory_actions, trajectory_images in zip(texts, images):
            trajectory_text = ""
            trajectory_image = []
            for action, image_base64 in zip(trajectory_actions, trajectory_images):
                if isinstance(image_base64, dict) and image_base64.get('url', '').startswith('data:image/png;base64,'):
                    image_bytes = base64.b64decode(image_base64.get('url', '').split(',')[1])
                elif isinstance(image_base64, str) and image_base64.startswith('data:image/png;base64,'):
                    image_bytes = base64.b64decode(image_base64.split(',')[1])
                else:
                    image_bytes = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_bytes))
                trajectory_image.append(image)
                trajectory_text += f"{DEFAULT_IM_START_TOKEN}user\n{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}{action}{DEFAULT_IM_END_TOKEN}\n"
            if trajectory_image:
                e_inputs = processor(text=[trajectory_text], images=trajectory_image, padding=False, return_tensors='pt')
                e_input_ids = e_inputs['input_ids'].squeeze(0)
                e_pixel_values = e_inputs['pixel_values']
                e_image_grid_thw = e_inputs['image_grid_thw']
                all_experience_pixel_values.append(e_pixel_values)
                all_experience_image_grid_thw.append(e_image_grid_thw)
            else:
                e_input_ids = processor.tokenizer(trajectory_text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
            
            all_experience_input_ids.append(e_input_ids)

        
        inputs['experience_input_ids'] = all_experience_input_ids
        inputs['experience_pixel_values'] = all_experience_pixel_values
        inputs['experience_image_grid_thw'] = all_experience_image_grid_thw
        
        return inputs
    
    def generate_response_with_experience(self, image=None, prompt=None, experience_texts=None, experience_images=None, file_id_list=None, conversation=None, experience_embedding=None):
        """Generate response with experience texts and images"""
        
        if not conversation:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ],
                }
            ]
        
        formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print('formatted_prompt:', formatted_prompt)
        image_inputs = self.process_vision_info(conversation)
        # print('image_number:', len(image_inputs))
        
        inputs = self.processor(
            text=[formatted_prompt],
            images=image_inputs,
            return_tensors="pt",
        ).to("cuda")
        file_id_list = None
        if file_id_list is not None:
            inputs['file_id_list'] = file_id_list
            inputs_with_experience = inputs
        else:
            # Process experience information
            inputs_with_experience = self.knowledge_processor_vlm(
                processor=self.processor,
                inputs=inputs,
                texts=experience_texts,
                images=experience_images,
                tokenizer=self.tokenizer,
                formatted_prompt=formatted_prompt
            ).to("cuda")
        
        generated_ids = self.model.generate(
            **inputs_with_experience, 
            max_new_tokens=self.max_tokens,
            use_cache=True, 
            temperature=self.temperature,
            top_p=self.top_p,
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_with_experience.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
        return output_text
    
    def chat(self, messages: List[Dict], stream: bool = False, 
             experience_texts=None, experience_images=None, file_id_list=None):
        """Chat with the model using transformers with experience support"""
        if stream:
            raise NotImplementedError("Streaming not yet implemented for transformers models")
        # Check if experience data is provided
        has_experience = False
        if experience_texts is not None:
            # Check if any experience text is not empty
            has_experience = any(len(text_list) > 0 for text_list in experience_texts)
        if experience_images is not None:
            # Check if any experience image list is not empty
            has_experience = any(len(img_list) > 0 for img_list in experience_images)

        if not has_experience:
            print("No experience data provided, falling back to DirectVLLMModel...")
            # Fall back to DirectVLLMModel when no experience data
            vllm_model = DirectVLLMModel(
                model_name='Qwen/Qwen2.5-VL-7B-Instruct',
                server_url='http://localhost:8000/v1',
                api_key="EMPTY",
                temperature=0.2,
                top_p=0.9,
                max_tokens=self.max_tokens
            )
            return vllm_model.chat(messages, stream=False)
        
        else:
            print("Generating response with experience...")
            # Generate response with experience
            response_text = self.generate_response_with_experience(
                experience_texts=experience_texts,
                experience_images=experience_images,
                file_id_list=file_id_list,
                conversation=messages
            )
            
            # Create OpenAI-style response
            return ChatCompletionMessage(
                role="assistant",
                content=response_text,
                function_call=None,
                tool_calls=None
            ), None, None


def create_direct_vllm_model(args: argparse.Namespace, model_name: str = None) -> DirectVLLMModel:
    """Create a direct vLLM model instance"""
    if model_name is None:
        model_name = args.model
    
    model_name_map = {
        'qwen2.5-vl': 'Qwen/Qwen2.5-VL-7B-Instruct',
        # 'qwen2.5-vl': 'qwen/qwen-2.5-vl-7b-instruct',
        'qwen2-vl': 'Qwen/Qwen2-VL-7B-Instruct',
        'qwen3-vl':  'Qwen/Qwen3-VL-8B-Instruct',
        # OpenRouter variant (served name differs from HF repo id)
        'qwen3-vl-or': 'qwen/qwen3-vl-8b-instruct',
        'ui-tars': 'ByteDance-Seed/UI-TARS-1.5-7B',
        # 'ui-tars': 'bytedance/ui-tars-1.5-7b',
        'ui-ins-7b': 'Tongyi-MiA/UI-Ins-7B',
        'ui-ins-32b': 'Tongyi-MiA/UI-Ins-32B',
        'websight': 'WenyiWU0111/websight-7B_combined',
        'cogagent': 'zai-org/cogagent-9b-20241220',
        'qwen2.5-vl-32b': 'Qwen/Qwen2.5-VL-32B-Instruct',
        # 'qwen2.5-vl-32b': 'qwen/qwen2.5-vl-32b-instruct',
        'fuyu': 'adept/fuyu-8b',
        'gemini': 'google/gemini-2.5-pro',
        'claude': 'anthropic/claude-sonnet-4',
        'gpt-4o': 'openai/gpt-4o',
    }
    model_server_map = {
        'qwen2.5-vl': 'http://localhost:8010/v1',  # Main agent model
        'qwen2-vl': 'http://localhost:8002/v1',
        'websight': 'http://localhost:8002/v1',
        'ui-ins-7b': 'http://localhost:8011/v1',  # Grounding model
        'ui-ins-32b': 'http://localhost:8005/v1',
        'ui-tars': 'http://localhost:8001/v1',
        'cogagent': 'http://localhost:8002/v1',
        'fuyu': 'http://localhost:8002/v1',
        'qwen2.5-vl-32b': 'http://localhost:8004/v1',
        # If you host Qwen3-VL locally with vLLM, use a dedicated port (e.g., 8007)
        'qwen3-vl': 'http://localhost:8007/v1',
        # OpenRouter endpoint for Qwen3-VL
        'qwen3-vl-or': 'https://openrouter.ai/api/v1',
        'gemini': 'https://openrouter.ai/api/v1',
        'claude': 'https://openrouter.ai/api/v1',
        'gpt-4o': 'https://openrouter.ai/api/v1',
    }

    model_name_ = model_name_map.get(model_name, model_name)
    server_url = model_server_map.get(model_name, 'http://localhost:8000/v1')
    # Prefer explicit arg, then environment variable, finally EMPTY
    api_key = getattr(args, 'open_router_api_key', None) or os.getenv('OPENROUTER_API_KEY', 'EMPTY')
    # Allow runtime overrides for quick experiments
    # - args.model_server_url (e.g., http://localhost:8007/v1)
    # - args.model_name_override (e.g., Qwen/Qwen3-VL-8B-Instruct)
    # - args.api_key_override (e.g., a different OpenRouter key)
    server_url = getattr(args, 'model_server_url', server_url)
    model_name_override = getattr(args, 'model_name_override', None)
    if model_name_override:
        model_name_ = model_name_override
    api_key = getattr(args, 'api_key_override', api_key)
    # server_url = 'https://openrouter.ai/api/v1'
    # api_key = ''
    print('model_name', model_name_)
    print('server_url', server_url)
    # Do not print full API keys; mask if present
    if api_key and api_key != 'EMPTY':
        print('api_key', f"{api_key[:6]}...{api_key[-4:]}")
    else:
        print('api_key', api_key)
    
    return DirectVLLMModel(
        model_name=model_name_,
        server_url=server_url,
        api_key=api_key,
        temperature=0.2,
        top_p=0.9,
        max_tokens=256,
    )


def create_direct_transformers_model(args: argparse.Namespace, model_name: str = None) -> DirectTransformersModel:
    """Create a direct Transformers model instance"""
    model_name_map = {
        'agent-qformer-full-sft': 'WenyiWU0111/agent-qformer-sft-qwen-large-memory-5000-merged',
        'agent-qformer-full-sft-rl': '/home/wenyi/rl_results/train/memory_agent_rl_v1/_actor/agent_rl_state1_60step_safetensors',
        'agent-qformer': 'WenyiWU0111/lora_qformer_test_V4-700_merged',
        'ui-tars': 'WenyiWU0111/lora_qformer_uitars_test_V1-400_merged'
    }
    if model_name is None:
        model_name = model_name_map.get(args.model, args.model)
    else:
        model_name = model_name_map.get(model_name, model_name)
    
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if checkpoint_path is None:
        checkpoint_path = model_name
    
    return DirectTransformersModel(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        temperature=0.1,
        top_p=0.001,
        args=args,
    )


def create_direct_model(args: argparse.Namespace):
    """Create a direct model instance based on model type"""
    if args.use_continuous_memory:
        return create_direct_transformers_model(args)
    else:
        # Default to vLLM
        return create_direct_vllm_model(args)


def load_grounding_model_vllm(args: argparse.Namespace):
    """
    Load grounding model using vLLM server with OpenAI client.

    Args:
        args: Arguments object

    Returns:
        Grounding model client
    """
    # Use grounding_model_name from args, default to ui-ins-7b
    grounding_model_name = getattr(args, 'grounding_model_name', 'ui-ins-7b')
    grounding_model = create_direct_vllm_model(args, model_name=grounding_model_name)
    return grounding_model

def load_tool_llm(args: argparse.Namespace, model_name=None) -> DirectVLLMModel:
    """Load tool LLM"""
    # Use tool_model_name from args if available, otherwise use provided model_name or default
    if model_name is None:
        model_name = getattr(args, 'tool_model_name', 'qwen2.5-vl')
    tool_model = create_direct_vllm_model(args, model_name=model_name)
    return tool_model

