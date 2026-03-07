"""
Trajectory Analysis System for GUI Agent Trajectories

This module implements a system to:
1. Organize trajectories into (state1→action→state2) tuples
2. Match current screenshots with historical states using CLIP
3. Analyze actions for risk/success patterns
4. Augment agents with risk/success information
"""

import json
import os
import base64
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import heapq
from PIL import Image
from io import BytesIO
from datetime import datetime
# Import the existing CLIP functionality
import sys 
sys.path.append("/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference")
from memory.help_functions import CLIPMultimodalSimilarity
from agent.llm_config import load_tool_llm, DirectVLLMModel
from action_scaling.help_functions import generate_html_content

@dataclass
class StateActionState:
    """Represents a state-action-state tuple from a trajectory."""
    state1_image: str  # Base64 encoded image
    action: str        # Action taken
    state2_image: str  # Base64 encoded image of resulting state
    trajectory_path: str # Path of the source trajectory
    round_index: int   # Round index in the trajectory


@dataclass
class ActionAnalysis:
    """Analysis of an action's risk and success patterns."""
    action: str
    success_rate: float
    risk_score: float
    success_examples: List[StateActionState]
    risk_examples: List[StateActionState]
    common_success_patterns: List[str]
    common_risk_patterns: List[str]


class TrajectoryAnalyzer:
    """Main class for analyzing GUI agent trajectories."""
    
    def __init__(self, trajectory_dir: str, state_embedding_path: str=None, tool_llm: DirectVLLMModel = None, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the trajectory analyzer.
        
        Args:
            trajectory_dir: Path to directory containing trajectory JSONL files
            state_embedding_path: Path to pre-computed state embeddings
            tool_llm: Tool LLM for analysis
            model_name: CLIP model name for image similarity
        """
        self.trajectory_dir = trajectory_dir
        self.state_embedding_path = state_embedding_path
        self.tool_llm = tool_llm
        print(f"Tool LLM model name: {self.tool_llm.model_name}")
        print(f"Tool LLM url: {self.tool_llm.server_url}")
        self.clip_similarity = CLIPMultimodalSimilarity(model_name)
        self.state_action_states: List[StateActionState] = []
        self.action_analyses: Dict[str, ActionAnalysis] = {}
        
        # Load pre-computed embeddings and metadata if available
        if self.state_embedding_path is not None:
            self.state_embeddings = np.load(self.state_embedding_path)
            # Load corresponding metadata
            metadata_path = self.state_embedding_path.replace('.npy', '.json')
            with open(metadata_path, 'r') as f:
                self.state_metadata = json.load(f)
            print(f"Loaded {len(self.state_metadata)} pre-computed embeddings and metadata")
        else:
            self.load_trajectories()
            self.compute_state_embeddings()
    
    def _reconstruct_state_action_state(self, metadata: Dict[str, Any]) -> Optional[StateActionState]:
        """
        Reconstruct a StateActionState object from metadata by loading the specific trajectory file.
        
        Args:
            metadata: Dictionary containing trajectory_id, round_index, and action
            
        Returns:
            StateActionState object or None if reconstruction fails
        """
        try:
            trajectory_path = metadata['trajectory_path']
            round_index = metadata['round_index']
            
            if not os.path.exists(trajectory_path):
                print(f"Warning: Could not find trajectory file for {trajectory_path}")
                return None
            
            # Load the specific trajectory and extract the state-action-state tuple
            with open(trajectory_path, 'r') as f:
                trajectory = json.load(f)
            rounds_data = trajectory.get('rounds', [])
            current_round_data = rounds_data[round_index]
            next_round_data = rounds_data[round_index + 1]
            messages = current_round_data.get('messages', [])
            action = current_round_data.get('response', '')
            next_messages = next_round_data.get('messages', [])
            
            # Extract state1, action, and state2
            state1_image = None
            state2_image = None
            
            for msg in reversed(messages):
                if msg.get('role') == 'user' and isinstance(msg.get('content'), list):
                    for content_item in msg['content']:
                        if isinstance(content_item, dict) and content_item.get('type') == 'image_url':
                            state1_image = content_item['image_url']['url']
                            break
                    if state1_image:
                        break
            
            for msg in next_messages:
                if msg.get('role') == 'user' and isinstance(msg.get('content'), list):
                    for content_item in msg['content']:
                        if isinstance(content_item, dict) and content_item.get('type') == 'image_url':
                            state2_image = content_item['image_url']['url']
                            break
                    if state2_image:
                        break
            
            if state1_image and state2_image:
                return StateActionState(
                    state1_image=state1_image,
                    action=action,
                    state2_image=state2_image,
                    trajectory_path=trajectory_path,
                    round_index=round_index
                )
            else:
                print(f"Warning: Could not extract state images for {trajectory_path} round {round_index}")
                return None
                
        except Exception as e:
            print(f"Error reconstructing StateActionState: {e}")
            return None
    
    def load_trajectories(self) -> None:
        """Load all trajectories from the directory and extract state-action-state tuples."""
        print("Loading trajectories...")
            
        for idx, (root, dirs, files) in enumerate(os.walk(self.trajectory_dir)):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    self._process_trajectory_file(file_path)
                    if idx % 100 == 0:
                        print(f"Processed {idx} trajectories")
        
        print(f"Loaded {len(self.state_action_states)} state-action-state tuples")
    
    def _process_trajectory_file(self, file_path: str) -> None:
        """Process a single trajectory file and extract state-action-state tuples."""
        try:
            with open(file_path, 'r') as f:
                trajectory = json.load(f)
            
            # Extract state-action-state tuples from rounds
            rounds_data = trajectory.get('rounds', [])
            if len(rounds_data) > 0:
                self._extract_state_action_states(rounds_data, file_path)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def _extract_state_action_states(self, rounds_data: Dict, trajectory_path: str) -> None:
        """Extract state-action-state tuples from a single round."""
        for idx, round_data in enumerate(rounds_data):
            messages = round_data.get('messages', [])
            action = round_data.get('response', '')
            # Find user message with image (state1)
            state1_image = None
            for msg in messages:
                if msg.get('role') == 'user' and isinstance(msg.get('content'), list):
                    for item in msg['content']:
                        if item.get('type') == 'image_url':
                            state1_image = item.get('image_url', {}).get('url', '')
        
            # Get state2 from the next round, if available
            state2_image = None
            if idx < len(rounds_data) - 1:  # Check if there's a next round
                next_round = rounds_data[idx + 1]
                next_messages = next_round.get('messages', [])
                for next_msg in next_messages:
                    if next_msg.get('role') == 'user' and isinstance(next_msg.get('content'), list):
                        for item in next_msg['content']:
                            if item.get('type') == 'image_url':
                                state2_image = item.get('image_url', {}).get('url', '')
                        break  # Found the user message, no need to continue
            
            if state1_image and action and state2_image:
                sas = StateActionState(
                    state1_image=state1_image,
                    action=action,
                    state2_image=state2_image,
                    trajectory_path=trajectory_path,
                    round_index=idx
                )
                self.state_action_states.append(sas)
    
    def compute_state_embeddings(self) -> None:
        """Compute embeddings for all states using CLIP."""
        print("Computing state embeddings...")
        
        # Extract images and texts for embedding computation
        images = []
        metadata = []
        for sas in self.state_action_states:
            try:
                # Decode base64 image
                if sas.state1_image.startswith('data:image'):
                    image_data = sas.state1_image.split(',')[1]
                else:
                    image_data = sas.state1_image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                images.append(image)
            except Exception as e:
                print(f"Error processing image: {e}")
                # Add placeholder
                images.append(Image.new('RGB', (224, 224), color='white'))
            
            metadata.append({
                    'trajectory_path': sas.trajectory_path,
                    'round_index': sas.round_index
                })

        # Compute embeddings in chunks
        if images:
            chunk_size = 100  # Process 100 images at a time
            print(f"Computing embeddings for {len(images)} states in chunks of {chunk_size}")
            
            all_embeddings = []
            for chunk_idx in range(0, len(images), chunk_size):
                chunk_images = images[chunk_idx:chunk_idx + chunk_size]
                print(f"Processing embedding chunk {chunk_idx//chunk_size + 1}/{(len(images) + chunk_size - 1)//chunk_size} ({len(chunk_images)} images)")
                
                chunk_embeddings = self.clip_similarity.get_image_embeddings(chunk_images)
                all_embeddings.append(chunk_embeddings)
            
            # Concatenate all embeddings
            self.state_embeddings = np.concatenate(all_embeddings, axis=0)
            embedding_shape = self.state_embeddings.shape
            date_time = datetime.now().strftime("%Y%m%d_%H%M")
            embedding_save_path = f"/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/action_scaling/state_embeddings/{date_time}_{embedding_shape[0]}.npy"
            os.makedirs(os.path.dirname(embedding_save_path), exist_ok=True)
            np.save(embedding_save_path, self.state_embeddings)
            print(f"Saved embeddings to {embedding_save_path}")
            with open(embedding_save_path.replace('.npy', '.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {embedding_save_path.replace('.npy', '.json')}")
        else:
            print("No images found for embedding computation")
    
    def find_similar_states(self, query_image: str, top_k: int = 5) -> List[Tuple[StateActionState, float]]:
        """
        Find states similar to the query image using pre-computed embeddings.
        
        Args:
            query_image: Base64 encoded query image
            top_k: Number of similar states to return
            
        Returns:
            List of (StateActionState, similarity_score) tuples
        """
        if self.state_embeddings is None:
            raise ValueError("State embeddings not computed. Call compute_state_embeddings() first.")
        
        if not self.state_metadata:
            raise ValueError("State metadata not loaded. Cannot reconstruct StateActionState objects.")
        
        # Process query image
        try:
            if query_image.startswith('data:image'):
                image_data = query_image.split(',')[1]
            else:
                image_data = query_image
            
            image_bytes = base64.b64decode(image_data)
            query_pil_image = Image.open(BytesIO(image_bytes))
            
            # Get query embedding
            query_embedding = self.clip_similarity.get_image_embeddings([query_pil_image])
            
            # Calculate similarities
            similarities = self.clip_similarity.calculate_similarity(
                query_embedding, self.state_embeddings
            )
            
            # Get top-k similar states
            top_indices = heapq.nlargest(top_k, range(len(similarities)), 
                                       key=lambda i: similarities[i])
            
            results = []
            for idx in top_indices:
                # Reconstruct StateActionState from metadata
                meta_data = self.state_metadata[idx]
                sas = self._reconstruct_state_action_state(meta_data)
                
                if sas:  # Only add if reconstruction was successful
                    similarity = similarities[idx]
                    results.append((sas, similarity))
                else:
                    print(f"Warning: Failed to reconstruct StateActionState for index {idx}")
            
            return results
            
        except Exception as e:
            print(f"Error finding similar states: {e}")
            return []
    
    def get_action_recommendations(self, current_state_image: str) -> str:
        """
        Get action recommendations based on current state using LLM analysis.
        
        Args:
            current_state_image: Base64 encoded current state image
            
        Returns:
            Dictionary with structured action recommendations and analysis
        """
        # Find similar states
        similar_states = self.find_similar_states(current_state_image, top_k=10)
        if not similar_states:
            return {"error": "No similar states found"}
        
        if not self.tool_llm:
            return {"error": "Tool LLM not available for analysis"}
        
        individual_evaluations = []
        # Evaluate each action individually
        for sas, similarity in similar_states:
            evaluation = self._evaluate_single_action(sas, current_state_image)
            individual_evaluations.append((evaluation, similarity))
        self.save_similar_states(current_state_image, similar_states, individual_evaluations)
        
        # Generate overall structured analysis based on individual evaluations
        overall_analysis = self._generate_overall_analysis(individual_evaluations, current_state_image)
        
        return overall_analysis
    
    def _evaluate_single_action(self, sas: StateActionState, current_state_image: str) -> Dict[str, Any]:
        """
        Evaluate a single action instance using LLM by comparing state1 and state2 images.
        
        Args:
            sas: StateActionState instance
            similarity: Similarity score with current state
            current_state_image: Base64 encoded current state image
            
        Returns:
            Dictionary with evaluation results
        """
        SYSTEM_PROMPT = open("/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/action_scaling/prompts/evaluate_single_action.txt", "r").read()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "The current state is: "},
                {"type": "image_url", "image_url": {"url": current_state_image}}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "The reference starting state is: "},
                {"type": "image_url", "image_url": {"url": sas.state1_image}}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "The action to evaluate is: "},
                {"type": "text", "text": sas.action}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "The reference resulting state is: "},
                {"type": "image_url", "image_url": {"url": sas.state2_image}}
            ]},
            {"role": "user", "content": "Please check the current state, and the reference (starting state, action, resulting state) and provide evaluation according to the system prompt."}
        ]
        
        # Send both state images to LLM for comparison
        # Note: You'll need to implement this based on your LLM's image handling capabilities
        response, _, _ = self.tool_llm.chat(messages, stream=False)
        response = response.content
        # Parse the response
        success_match = re.search(r'"success":\s*(true|false)', response)
        if success_match:
            success = success_match.group(1).lower() == 'true'
        else:
            success = False
        # Add metadata
        evaluation = {
            'success': success,
            'evaluation': response,
            'raw_sas': {
            'trajectory_path': sas.trajectory_path,
            'round_index': sas.round_index,
            'action': sas.action
                        }
            }
        
        return evaluation
        
    def _generate_overall_analysis(self, individual_evaluations: List[Dict], current_state_image: str) -> str:
        """
        Generate overall structured analysis based on individual action evaluations.
        
        Args:
            individual_evaluations: List of individual action evaluations
            current_state_image: Base64 encoded current state image
            
        Returns:
            Structured overall analysis response
        """
        # Extract patterns from evaluations
        total_actions = len(individual_evaluations)
        successful_actions = []
        failed_actions = []
        
        for eval_data, similarity in individual_evaluations:
            raw_action = eval_data.get('raw_sas', {}).get('action', 'unknown')
            if eval_data['success']:
                # Extract patterns from the evaluation text if available
                if 'evaluation' in eval_data:
                    successful_actions.append((raw_action, eval_data['evaluation'], similarity))
            else:
                failed_actions.append((raw_action, eval_data['evaluation'], similarity))
                
        prompt = f"""You are an expert GUI automation analyst. Your task is to extract high-level experience and suggestions from {total_actions} individual action evaluations to help an agent make better decisions. You will be given the successful and failed actions, and the evaluation of each action. 
        We extract top-k similar states and corresponding actions from the current state, and evaluate the actions success/failure. The similarity score is the similarity between the current state and the reference state.
        The following are the successful actions:"""
        for action, evaluation, similarity in successful_actions:
            prompt += f"Action: {action}\nEvaluation: {evaluation}\nSimilarity: {similarity:.3f}\n"        
        prompt += "The following are the failed actions:"
        for action, evaluation, similarity in failed_actions:
            prompt += f"Action: {action}\nEvaluation: {evaluation}\nSimilarity: {similarity:.3f}\n"
        prompt += """Extract high-level experience and provide actionable guidance in this JSON format:
{
    "key_insights": [
      "Most important insights from the evaluations",
      "Patterns that lead to success/failure"
    ],
  "action_guidance": {
    "recommended_actions": [
      "Specific actions that typically succeed",
      "Action patterns to follow"
    ],
    "risky_actions": [
      "Actions that typically fail",
      "Patterns to avoid"
    ]
  },
  "practical_suggestions": [
    "Concrete suggestions for the agent",
    "Specific strategies to improve success"
  ]
}
Focus on extracting actionable experience that directly helps the agent make better decisions."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        response, _, _ = self.tool_llm.chat(messages, stream=False)
        return response.content
    
    
    def save_similar_states(self, current_state_image: str, similar_states: List[Tuple], individual_evaluations: List[Tuple]) -> None:
        """
        Save similar states and evaluations to HTML files for visualization.
        
        Args:
            current_state_image: Base64 encoded current state image
            similar_states: List of (StateActionState, similarity) tuples
            individual_evaluations: List of (evaluation, similarity) tuples
        """
        try:
            # Create output directory
            output_dir = "/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/action_scaling/check_similar_states"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            html_file = os.path.join(output_dir, f"similar_states_{timestamp}.html")
            
            # Validate inputs
            if not similar_states or not individual_evaluations:
                print("Warning: No similar states or evaluations to save")
                return
            
            if len(similar_states) != len(individual_evaluations):
                print(f"Warning: Mismatch between similar_states ({len(similar_states)}) and individual_evaluations ({len(individual_evaluations)})")
                return
            
            # Create HTML content
            html_content = generate_html_content(current_state_image, similar_states, individual_evaluations)
            
            # Validate HTML content
            if not html_content or len(html_content) < 100:
                print("Warning: Generated HTML content is too short or empty")
                return
            
            # Save HTML file
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"Similar states saved to: {html_file}")
            
        except Exception as e:
            print(f"Error saving similar states: {e}")
            import traceback
            traceback.print_exc()
    


def main():
    """Example usage of the TrajectoryAnalyzer."""
    # Initialize analyzer
    trajectory_dir = "/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/expand_memory_organized"
    state_embedding_path = None
    tool_llm = DirectVLLMModel(
        model_name="qwen2.5-vl",
        server_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
    analyzer = TrajectoryAnalyzer(trajectory_dir, state_embedding_path, tool_llm)
    
    # # Save analysis
    # analyzer.save_analysis("/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/action_analysis.json")
    
    # Example: Get recommendations for a state
    # (You would provide an actual base64 image here)
    # recommendations = analyzer.get_action_recommendations("base64_image_data")
    # print(recommendations)


if __name__ == "__main__":
    main()
