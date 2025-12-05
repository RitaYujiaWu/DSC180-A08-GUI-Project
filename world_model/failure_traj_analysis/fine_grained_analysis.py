#!/usr/bin/env python3
"""
Phase 1: Error Type Detection System V5
Using latest error_type_definition.md with V3 loader
"""

import json
import os
import asyncio
import aiohttp
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pathlib
import sys
root_dir = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{root_dir}/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference")
from agent.llm_config import DirectVLLMModel
from error_detector.error_definitions import ErrorDefinitionsLoader

@dataclass
class ModuleError:
    """Error detection for a single module"""
    module_name: str
    error_type: str
    error_detected: bool
    evidence: str
    reasoning: str


@dataclass
class StepAnalysis:
    """Analysis for a single step"""
    step: int
    memory_error: Optional[ModuleError]
    reflection_error: Optional[ModuleError]
    planning_error: Optional[ModuleError]
    action_error: Optional[ModuleError]
    step_summary: str


class ErrorTypeDetector:
    """Detects error types without scoring"""
    
    def __init__(self, api_config: Dict[str, Any]=None):
        if api_config is None:
            self.config = {
                'model': 'Qwen/Qwen2.5-VL-7B-Instruct',
                'base_url': 'http://localhost:8000/v1',
                'api_key': 'EMPTY'
            }
        else:
            self.config = api_config
        self.llm = DirectVLLMModel(
            model_name=self.config['model'],
            server_url=self.config['base_url'],
            api_key=self.config['api_key']
        )
        # Load error definitions
        self.error_loader = ErrorDefinitionsLoader()
        
        # Define error types based on latest error_type_definition.md
        self.error_types = {
            'memory': self.error_loader.get_valid_error_types('memory'),
            'reflection': self.error_loader.get_valid_error_types('reflection'),
            'planning': self.error_loader.get_valid_error_types('planning'),
            'action': self.error_loader.get_valid_error_types('action'),
            'system': self.error_loader.get_valid_error_types('system'),
            'others': self.error_loader.get_valid_error_types('others')
        }
    
    def parse_trajectory(self, file_path: str) -> Dict[str, Any]:
        """Parse trajectory file"""
        # open html file
        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        # parse html
        soup = BeautifulSoup(html, 'html.parser')
        # find task id : match task_id: 
        pattern = r'task_id: (\d+)'
        match = re.search(pattern, html)
        if match:
            task_id = match.group(1)
        else:
            task_id = 'unknown'
        # find task description: match intent: 
        pattern = r'intent: (.*)'
        match = re.search(pattern, html)
        if match:
            task_description = match.group(1)
        else:
            task_description = 'unknown'
        # find environment: match site: 
        pattern = r'site: (.*)'
        match = re.search(pattern, html)
        if match:
            environment = match.group(1)
        else:
            environment = 'unknown'
        # find success
        score_div = soup.find('div', class_='final_score')
        if score_div and score_div.text.strip().isdigit():
            score = int(score_div.text.strip())
            success = score == 1
        else:
            success = False
        # find images
        image_nodes = []
        for state_div in soup.find_all('div', class_='state_obv'):
            img_tag = state_div.find('img')
            if img_tag and img_tag.get('src'):
                image_nodes.append(img_tag)
        images = [image['src'].split(',')[-1].strip() for image in image_nodes]
        # find all the <div class="step">
        responses = soup.find_all('div', class_='response_history')
        total_steps = len(responses)
        
        # Extract steps
        steps = []
        
        for i, response in enumerate(responses):
            # Keep inner HTML so module tags like <memory>, <reflection>, etc. remain available
            content = response.decode_contents()
            steps.append({
                'step': i + 1,
                'content': content,
                'current_image': images[i],
                'next_image': images[i+1] if i+1 < len(images) else None,
                # 'env_response': env_response,
                # 'current_input': current_input
            })
        
        return {
            'task_id': task_id,
            'task_description': task_description,
            'success': success,
            'steps': steps,
            'total_steps': total_steps,
            'environment': environment
        }
    
    def extract_module_content_from_step(self, content: str, module_name: str, env: str) -> str:
        """Extract specific module content from step"""
        modules = self.extract_modules_from_content(content, env)
        return modules.get(module_name, "")
    
    def extract_modules_from_content(self, content: str, env: str) -> Dict[str, str]:
        """Extract module content based on environment and prompt format"""
        modules = {}

        # Both ALFWorld and WebShop use tag-based format
        # Memory analysis
        memory_match = re.search(r'<memory>(.*?)</memory>', content, re.DOTALL)
        if memory_match:
            modules['memory'] = memory_match.group(1).strip()
        else:
            modules['memory'] = ""
            print(f"No memory found")

        # Reflection
        reflection_match = re.search(r'<reflection>(.*?)</reflection>', content, re.DOTALL)
        if reflection_match:
            modules['reflection'] = reflection_match.group(1).strip()
        else:
            modules['reflection'] = ""
            print(f"No reflection found")

        # Plan
        plan_match = re.search(r'<plan>(.*?)</plan>', content, re.DOTALL)
        if plan_match:
            modules['planning'] = plan_match.group(1).strip()
        else:
            modules['planning'] = ""
            print(f"No planning found")

        # Action - most critical part to fix
        action_match = re.search(r'<action>(.*?)</action>', content, re.DOTALL)
        if action_match:
            action_content = action_match.group(1).strip()
            modules['action'] = action_content
        else:
            # Fallback for cases without action tags (shouldn't happen in standard format)
            modules['action'] = ""
            print(f"No action found")

        # Special handling for WebShop: extract the actual action command if present
        if env == 'webshop' and modules['action']:
            # For WebShop, the action content might contain the actual command
            # like "search[query]" or "click[button]" or "buy[item]"
            action_patterns = [
                r'search\[[^\]]*\]',
                r'click\[[^\]]*\]',
                r'buy\[[^\]]*\]'
            ]
            for pattern in action_patterns:
                match = re.search(pattern, modules['action'], re.IGNORECASE)
                if match:
                    # Keep the full action command
                    modules['action'] = match.group(0)
                    break

        # Special handling for GAIA environment
        if env == 'gaia':
            # GAIA uses slightly different tags
            # Memory recall instead of memory
            memory_recall_match = re.search(r'<memory_recall>(.*?)</memory_recall>', content, re.DOTALL)
            if memory_recall_match and not modules['memory']:
                modules['memory'] = memory_recall_match.group(1).strip()

            # Tool calls in GAIA
            tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            if tool_call_match:
                # Extract the tool and parameters
                tool_content = tool_call_match.group(1).strip()
                if 'tool:' in tool_content:
                    modules['action'] = tool_content

            # Answer tags for final answers in GAIA
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if answer_match:
                # This is a final answer, not an action
                modules['action'] = f"answer: {answer_match.group(1).strip()[:100]}"

        # Special handling for WebShop's think tag
        if env == 'webshop':
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if think_match and not modules['planning']:
                modules['planning'] = think_match.group(1).strip()

        return modules
    
    def detect_module_errors(
        self,
        module_name: str,
        module_content: str,
        step_num: int,
        step_data: Dict,
        task_description: str,
        previous_steps: List[Dict],
        environment: str
    ) -> ModuleError:
        """Detect errors in a specific module"""
        
        # Build detection prompt
        prompt = self._build_error_detection_prompt(
            module_name,
            module_content,
            step_num,
            step_data,
            task_description,
            previous_steps,
            environment
        )
        
        # Call LLM for error detection
        response, _, _ = self.llm.chat(prompt, stream=False)
        response = response.content
        
        # Parse detection result
        error = self._parse_error_detection(response, module_name)
        
        return error
    
    def _build_error_detection_prompt(
        self,
        module_name: str,
        module_content: str,
        step_num: int,
        step_data: Dict,
        task_description: str,
        previous_steps: List[Dict],
        environment: str
    ) -> list:
        """Build prompt for error detection"""
        
        # Get detailed error definitions for this module
        error_definitions = self.error_loader.format_for_phase1_prompt(module_name)
        # get current step observation
        observation_at_this_step = step_data['current_image']
        # get next step observation
        observation_after_this_step = step_data['next_image']
        
        # # Build context - ALL modules need current step input (user message which includes history)
        # context = f"Current Step Input (user message with history):\n{current_step_input}\n\n"
        context = ''
        # Add previous module outputs from THIS SAME STEP for evaluation
        if module_name == 'memory':
            # Memory only needs the current step input
            pass
        elif module_name == 'reflection':
            # Reflection needs to see Memory output from this step
            memory_content = self.extract_module_content_from_step(step_data['content'], 'memory', environment)
            if memory_content:
                context += f"Memory Module Output (from this step):\n{memory_content}\n\n"
        elif module_name == 'planning':
            # Planning needs to see Memory and Reflection outputs from this step
            memory_content = self.extract_module_content_from_step(step_data['content'], 'memory', environment)
            reflection_content = self.extract_module_content_from_step(step_data['content'], 'reflection', environment)
            if memory_content:
                context += f"Memory Module Output (from this step):\n{memory_content}\n\n"
            if reflection_content:
                context += f"Reflection Module Output (from this step):\n{reflection_content}\n\n"
        elif module_name == 'action':
            # Action needs to see Planning output from this step
            planning_content = self.extract_module_content_from_step(step_data['content'], 'planning', environment)
            if planning_content:
                context += f"Planning Module Output (from this step):\n{planning_content}\n\n"
        
        prompt_part1 = f"""
You are an expert at detecting errors in agent trajectories.

TASK: {task_description}
ENVIRONMENT: {environment}
CURRENT STEP: {step_num}

INPUT AND CONTEXT:
{context}

MODULE TO ANALYZE: {module_name}
MODULE OUTPUT (What the agent produced for this module):
{module_content if module_content else "No content found for this module"}

{error_definitions}
"""

        prompt_part2 = """
Based on the SPECIFIC error definitions provided above:
1. Identify if there is an error in this module by checking if the output matches any error definition
2. If yes, specify which exact error type based on the definitions
3. Provide evidence from the content that directly relates to the definition
4. Explain your reasoning showing how it matches the specific definition criteria

SPECIAL RULES:
- Please compare the observation at this step and the observation after this step to check if there is any error.
- Evaluation criteria for each module:
  * Memory: Should correctly summarize/recall from the current step input only
  * Reflection: Should correctly reflect based on current input + this step's Memory output
  * Planning: Should plan reasonably based on current input + this step's Memory & Reflection outputs
  * Action: Should execute correctly based on current input + this step's Planning output
- Each module builds on previous modules' outputs FROM THE SAME STEP
- System errors (step_limit, tool_execution_error, etc.) should be identified separately

REQUIRED OUTPUT FORMAT (JSON):
{{
    "error_detected": true/false,
    "error_type": "specific_error_type or no_error",
    "evidence": "Quote or description from module content supporting the detection",
    "reasoning": "Explanation of why this is (or isn't) an error based on the definition"
}}

Be precise and base your detection on the actual content and error definitions.
Output must be a single JSON object with no additional commentary or text outside the JSON.
"""
        prompt = [
            {'role': 'system', 'content': prompt_part1},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'OBSERVATION AT THIS STEP:'},
                    {'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{observation_at_this_step}"}}
                ]
            }
        ]
        if observation_after_this_step:
            prompt.append({
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'OBSERVATION AFTER THIS STEP:'},
                    {'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{observation_after_this_step}"}}
                ]
            })
        else:
            prompt.append({'role': 'user', 'content': 'This is the last step, so there is no observation after this step'})
        prompt.append({'role': 'user', 'content': prompt_part2})
        return prompt
    
    def _check_system_errors(
        self,
        step_num: int,
        step_data: Dict,
        task_description: str,
        env_response: str,
        environment: str
    ) -> Optional[ModuleError]:
        """Check for system-level errors"""
        
        # Check for step limit
        if step_num >= 30:  # Common step limit in ALFWorld
            return ModuleError(
                module_name='system',
                error_type='step_limit',
                error_detected=True,
                evidence=f"Reached step {step_num}, hitting system step limit",
                reasoning="Task failed due to exceeding maximum allowed steps"
            )
        
        # Check for environment errors in response
        if env_response and any(keyword in env_response.lower() for keyword in ['error', 'crashed', 'exception', 'timeout']):
            # Call LLM to verify if this is a system error
            prompt = f"""
Analyze if this environment response indicates a system error:

Environment Response: {env_response[:500]}

System error types:
- tool_execution_error: External tool or API returned error
- llm_limit: LLM response limitations (timeout, max tokens)
- environment_error: Environment bug or crash

Is this a system error? If yes, which type?

REQUIRED OUTPUT FORMAT (JSON):
{{
    "error_detected": true/false,
    "error_type": "specific_system_error_type or no_error",
    "evidence": "Quote from response",
    "reasoning": "Why this is/isn't a system error"
}}
"""
            response, _, _ = self.llm.chat(prompt, stream=False)
            response = response.content
            return self._parse_error_detection(response, 'system')
        
        return None
    
    def _parse_error_detection(self, response: str, module_name: str) -> ModuleError:
        """Parse LLM error detection response"""

        def _strip_code_fences(text: str) -> str:
            if text.strip().startswith("```"):
                lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
                return "\n".join(lines)
            return text

        def _extract_json_candidates(text: str) -> List[str]:
            candidates: List[str] = []
            start = text.find('{')
            while start != -1:
                brace_level = 0
                end = start
                for idx in range(start, len(text)):
                    char = text[idx]
                    if char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level -= 1
                        if brace_level == 0:
                            end = idx
                            candidates.append(text[start:end + 1])
                            break
                start = text.find('{', end + 1)
            return candidates

        response = _strip_code_fences(response.strip())

        try:
            candidates = _extract_json_candidates(response)
            if not candidates:
                raise ValueError("No JSON found in response")

            last_error: Optional[Exception] = None
            detection = None

            for json_str in candidates:
                # Primary attempt
                try:
                    detection = json.loads(json_str)
                    break
                except json.JSONDecodeError as primary_error:
                    try:
                        import ast

                        pythonish = re.sub(r'\btrue\b', 'True', json_str, flags=re.IGNORECASE)
                        pythonish = re.sub(r'\bfalse\b', 'False', pythonish, flags=re.IGNORECASE)
                        pythonish = re.sub(r'\bnull\b', 'None', pythonish, flags=re.IGNORECASE)
                        detection = ast.literal_eval(pythonish)
                        break
                    except Exception as secondary_error:
                        last_error = primary_error if isinstance(primary_error, json.JSONDecodeError) else secondary_error
                        continue

            if detection is None:
                raise last_error or ValueError("Failed to parse any JSON candidate")
            
            return ModuleError(
                module_name=module_name,
                error_type=detection.get('error_type', 'unknown'),
                error_detected=detection.get('error_detected', False),
                evidence=detection.get('evidence', 'No evidence provided'),
                reasoning=detection.get('reasoning', 'No reasoning provided')
            )

        except Exception as e:
            logger.warning(
                "Failed to parse error detection for %s: %s | raw=<%s>",
                module_name,
                e,
                response.strip().replace('\n', ' ')[:500]
            )

            # Regex fallback to salvage fields from semi-structured output
            error_detected_match = re.search(
                r'error[_\s-]*detected[^:]*:\s*([A-Za-z]+)',
                response,
                re.IGNORECASE
            )
            error_type_match = re.search(
                r'error[_\s-]*type[^:]*:\s*["\']?([^"\'\n\r,]+)',
                response,
                re.IGNORECASE
            )
            evidence_match = re.search(
                r'evidence[^:]*:\s*(.+?)(?:reasoning[^:]*:|$)',
                response,
                re.IGNORECASE | re.DOTALL
            )
            reasoning_match = re.search(
                r'reasoning[^:]*:\s*(.+)$',
                response,
                re.IGNORECASE | re.DOTALL
            )

            if error_detected_match or error_type_match:
                detected_str = error_detected_match.group(1).strip().lower() if error_detected_match else 'false'
                detected_val = detected_str in {'true', 'yes', '1'}
                error_type_val = error_type_match.group(1).strip().lower() if error_type_match else 'unknown'
                evidence_val = evidence_match.group(1).strip().rstrip(',') if evidence_match else 'Extracted via regex fallback'
                reasoning_val = reasoning_match.group(1).strip() if reasoning_match else 'Extracted via regex fallback'

                return ModuleError(
                    module_name=module_name,
                    error_type=error_type_val,
                    error_detected=detected_val,
                    evidence=evidence_val,
                    reasoning=reasoning_val
                )

            return ModuleError(
                module_name=module_name,
                error_type='parse_error',
                error_detected=False,
                evidence=f"Parse error: {str(e)}",
                reasoning="Failed to parse LLM response"
            )
    
    def analyze_step(
        self,
        step_data: Dict,
        task_description: str,
        previous_steps: List[Dict],
        task_success: bool,
        environment: str
    ) -> StepAnalysis:
        """Analyze all modules in a step for errors"""
        
        step_num = step_data['step']
        content = step_data['content']
        # env_response = step_data['env_response']
        env_response = content 
        
        # Extract module content
        modules = self.extract_modules_from_content(content, environment)
        
        # Detect errors in each module
        module_errors = {}
        
        # Get current step input (user message) for Memory module
        current_step_input = ""
        if 'current_input' in step_data:
            current_step_input = step_data['current_input']
        
        # Check for system errors first (these apply to the whole step, not specific modules)
        system_error = self._check_system_errors(
            step_num, 
            step_data, 
            task_description,
            env_response,
            environment
        )
        if system_error and system_error.error_detected:
            module_errors['system'] = system_error
        
        for module_name in ['memory', 'reflection', 'planning', 'action']:
            # print(f"Analyzing module: {module_name}")
            module_content = modules.get(module_name, "")

            # Skip memory/reflection for step 1 (no history to remember or reflect on)
            if step_num == 1 and module_name in ['memory', 'reflection']:
                module_errors[module_name] = None
                continue

            # Skip memory/reflection for WebShop if not present
            if environment == 'webshop' and module_name in ['memory', 'reflection'] and not module_content:
                module_errors[module_name] = None
                continue
            
            error = self.detect_module_errors(
                module_name,
                module_content,
                step_num,
                step_data,
                task_description,
                previous_steps,
                environment
            )
            # print(f"Error detected for module {module_name}: {error}")
            module_errors[module_name] = error
        
        # Generate step summary
        errors_found = [
            f"{err.module_name}:{err.error_type}" 
            for err in module_errors.values() 
            if err and err.error_detected
        ]
        
        step_summary = f"Step {step_num}: "
        if errors_found:
            step_summary += f"Errors detected - {', '.join(errors_found)}"
        else:
            step_summary += "No errors detected"
        
        return StepAnalysis(
            step=step_num,
            memory_error=module_errors.get('memory'),
            reflection_error=module_errors.get('reflection'),
            planning_error=module_errors.get('planning'),
            action_error=module_errors.get('action'),
            step_summary=step_summary
        )
    
    def analyze_trajectory(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze complete trajectory for error types"""
        
        task_description = trajectory_data['task_description']
        task_success = trajectory_data['success']
        steps = trajectory_data['steps']
        environment = trajectory_data['environment']
        
        # Analyze each step
        step_analyses = []
        previous_steps = []
        
        for step_data in steps:
            analysis = self.analyze_step(
                step_data,
                task_description,
                previous_steps,
                task_success,
                environment
            )
            step_analyses.append(analysis)
            previous_steps.append(step_data)
        
        # Convert to serializable format
        analyses_dict = []
        for analysis in step_analyses:
            step_dict = {
                'step': analysis.step,
                'errors': {},
                'summary': analysis.step_summary
            }
            
            for module_name in ['memory', 'reflection', 'planning', 'action']:
                error = getattr(analysis, f"{module_name}_error")
                if error:
                    step_dict['errors'][module_name] = {
                        'error_type': error.error_type,
                        'error_detected': error.error_detected,
                        'evidence': error.evidence,
                        'reasoning': error.reasoning
                    }
            
            analyses_dict.append(step_dict)
        
        return {
            'task_id': trajectory_data['task_id'],
            'task_description': task_description,
            'task_success': task_success,
            'environment': environment,
            'total_steps': len(steps),
            'step_analyses': analyses_dict
        }
    
    def process_file(self, file_path: str, output_dir: str, output_filename: str = None) -> Dict[str, Any]:
        """Process a single trajectory file"""
        try:
            # Parse trajectory
            trajectory_data = self.parse_trajectory(file_path)
            
            # Analyze for errors
            analysis = self.analyze_trajectory(trajectory_data)
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            if output_filename:
                output_file = Path(output_dir) / f"{output_filename}_error_detection.json"
            else:
                output_file = Path(output_dir) / f"{Path(file_path).stem}_error_detection.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed {analysis['task_id']}: {analysis['total_steps']} steps analyzed")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
def _main():
    detector = ErrorTypeDetector()
    detector.process_file(
        "html/render_Amazon--3.html",
        "error_detection_results"
    )


if __name__ == "__main__":
    _main()
