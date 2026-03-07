"""Test runner for the GUI Agent"""
import argparse
import json
import logging
import os
import pickle
import re
import shutil
from pathlib import Path
import base64
import io
import time
from typing import List, Optional, Dict, Any
import pathlib
root_dir = pathlib.Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(f"{root_dir}/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference")
from error_detector.fine_grained_analysis import ErrorTypeDetector
from error_detector.critical_error_detection import CriticalErrorAnalyzer
from browser_env import (
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from agent.agent import Action
from utils.early_stop import early_stop
from utils.help_functions import save_scores_to_json
from agent.llm_config import load_tool_llm
import shutil
from webvoyager_evaluation.evaluator import LLMEvaluator
from memory.reasoning_bank import ReasoningBank, distill_reasoning_items


class TestRunner:
    """Handles the main test execution loop"""
    
    def __init__(self, args: argparse.Namespace, agent):
        self.args = args
        self.agent = agent
        self.logger = logging.getLogger("logger")
        # Initialize environment
        self.env = ScriptBrowserEnv(
            headless=True,
            slow_mo=args.slow_mo,
            viewport_size={
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            save_trace_enabled=args.save_trace_enabled,
            sleep_after_execution=args.sleep_after_execution,
            args=args,  # Pass args to the environment
        )
        self.evaluate_model = load_tool_llm(self.args, model_name='qwen2.5-vl')
        self.evaluator = LLMEvaluator(vllm_client=self.evaluate_model)
        if self.args.use_error_detection:
            self.error_detect_detector = ErrorTypeDetector()
            self.critical_error_detector = CriticalErrorAnalyzer()
        else:
            self.error_detect_detector = None
            self.critical_error_detector = None

    def _compute_and_display_success_rate(self):
        """Compute and display success rate after all tests complete."""
        from pathlib import Path
        import re

        result_dir = Path(self.args.result_dir)

        # Find all final render HTML files
        render_files = list(result_dir.glob("render_*.html"))
        final_render_files = [f for f in render_files if '_attempt' not in f.name]

        if not final_render_files:
            final_render_files = render_files

        if not final_render_files:
            self.logger.warning("No render files found for success rate computation")
            return

        scores = {}
        successful = 0
        failed = 0

        for render_file in sorted(final_render_files):
            try:
                with open(render_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Look for score in HTML div structure: <div class='final_score'><pre>X.X</pre></div>
                    # 0.0 = failed, 1.0 = success
                    score_match = re.search(r"<div class='final_score'><pre>([0-9.]+)</pre></div>", content)
                    if score_match:
                        score = float(score_match.group(1))
                    else:
                        continue

                    # Extract task ID
                    filename = render_file.name
                    task_id_match = re.search(r'render_(\d+)(?:_attempt\d+)?\.html', filename)
                    task_id = task_id_match.group(1) if task_id_match else filename

                    scores[task_id] = score
                    if score >= 0.5:
                        successful += 1
                    else:
                        failed += 1
            except Exception as e:
                self.logger.warning(f"Error parsing {render_file}: {e}")

        total = successful + failed
        success_rate = (successful / total * 100) if total > 0 else 0.0

        # Save summary to JSON
        summary = {
            "total_tasks": total,
            "successful_tasks": successful,
            "failed_tasks": failed,
            "success_rate": success_rate,
            "scores": scores
        }

        summary_file = result_dir / "score_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

        # Display summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Tasks:       {total}")
        self.logger.info(f"Successful:        {successful}")
        self.logger.info(f"Failed:            {failed}")
        self.logger.info(f"Success Rate:      {success_rate:.2f}%")
        self.logger.info("=" * 60)
        self.logger.info(f"Summary saved to: {summary_file}")

        # Also print to console
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Tasks:       {total}")
        print(f"Successful:        {successful}")
        print(f"Failed:            {failed}")
        print(f"Success Rate:      {success_rate:.2f}%")
        print("=" * 60)

    def _build_feedback_prompt(self, feedback_history: List[str]) -> str:
        """Format accumulated feedback into a prompt fragment."""
        if not feedback_history:
            return ""
        lines = [f"{idx + 1}. {text}" for idx, text in enumerate(feedback_history)]
        return "Address these critical feedback points from previous attempts:\n" + "\n".join(lines)
    
    def _generate_iterative_feedback(
        self,
        render_path: Path,
        episode_id: str,
        attempt_idx: int
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        """Run error analysis and return targeted feedback for the next attempt."""
        if not self.error_detect_detector or not self.critical_error_detector:
            return None
        
        analysis_dir = Path(self.args.result_dir) / "error_detection" / f"attempt_{attempt_idx + 1}"
        os.makedirs(analysis_dir, exist_ok=True)
        phase1_prefix = f"{episode_id}_attempt{attempt_idx + 1}"
        
        phase1_result = self.error_detect_detector.process_file(
            str(render_path),
            str(analysis_dir),
            output_filename=phase1_prefix
        )
        if not phase1_result or phase1_result.get("task_success"):
            self.logger.info("Error detection found no issues or task succeeded; skipping feedback generation.")
            return None
        
        phase1_file = analysis_dir / f"{phase1_prefix}_error_detection.json"
        critical_result = self.critical_error_detector.process_trajectory(
            str(phase1_file),
            str(render_path),
            str(analysis_dir)
        )
        if not critical_result or not critical_result.get("critical_error"):
            self.logger.warning("Critical error analysis failed to produce feedback.")
            return None
        
        critical = critical_result["critical_error"]
        feedback_lines = [
            f"Critical failure at step {critical['critical_step']} in module {critical['critical_module']} "
            f"(error type: {critical['error_type']}).",
            f"Root cause: {critical['root_cause']}."
        ]
        guidance = critical.get("correction_guidance", "").strip()
        if guidance:
            feedback_lines.append(f"Follow this corrective guidance: {guidance}.")
        cascading = critical.get("cascading_effects", [])
        if cascading:
            casc_text = "; ".join(
                f"Step {effect.get('step')}: {effect.get('effect')}"
                for effect in cascading
                if effect.get('step') is not None and effect.get('effect')
            )
            if casc_text:
                feedback_lines.append(f"Cascading impact to monitor: {casc_text}.")
        
        feedback_text = " ".join(feedback_lines).strip()
        self.logger.info(f"Generated feedback for next attempt: {feedback_text}")
        return feedback_text, critical
    
    def _run_single_attempt(
        self,
        config_file: str,
        base_intent: str,
        attempt_idx: int,
        feedback_history: List[str],
        task_meta: Dict[str, Any],
        resume_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute one full attempt of the task, optionally under feedback guidance."""
        self.logger.info(f"===== Attempt {attempt_idx + 1} starts for {config_file} =====")
        render_helper = RenderHelper(config_file, self.args.result_dir, attempt_idx + 1)
        
        score_for_render = 0.0
        try:
            self.agent.reset(config_file)
            self.agent.current_step = 0
            
            trajectory: Trajectory = []
            obs, info = self.env.reset(options={"config_file": config_file})
            current_url = info["page"].url
            state_info: StateInfo = {"observation": obs, "info": info, "current_url": current_url}
            trajectory.append(state_info)
            
            meta_data: Dict[str, Any] = {"action_history": [], "response_history": [], "page": self.env.page}
            resumed_actions: List[Action] = []
            if resume_info:
                critical_step = max(1, int(resume_info.get("critical_step", 1)))
                prev_traj = resume_info.get("trajectory", [])
                for step_idx in range(max(0, critical_step - 1)):
                    action_idx = 2 * step_idx + 1
                    if action_idx >= len(prev_traj):
                        break
                    prev_action = prev_traj[action_idx]
                    if not isinstance(prev_action, dict):
                        continue
                    action_type = prev_action.get("action_type")
                    if action_type == ActionTypes.STOP:
                        break
                    resumed_actions.append(prev_action)
                if resumed_actions:
                    self.logger.info(
                        f"[Attempt {attempt_idx + 1}] Replaying {len(resumed_actions)} actions to resume before step {critical_step}"
                    )
            
            feedback_prompt = self._build_feedback_prompt(feedback_history)
            if feedback_prompt:
                self.logger.info(f"[Attempt {attempt_idx + 1}] Applying feedback:\n{feedback_prompt}")
            
            enhanced_intent = (
                f"{base_intent} Once you find the result, please directly yield a stop action, "
                f"and give a brief explanation in your answer!"
            )
            if feedback_prompt:
                enhanced_intent = (
                    f"{enhanced_intent}\n\n"
                    f"Critical feedback from prior attempts:\n{feedback_prompt}\n"
                    f"Ensure your next steps explicitly address this guidance."
                )

            # Replay actions up to the critical failure point
            if resumed_actions:
                for replay_idx, replay_action in enumerate(resumed_actions, start=1):
                    trajectory.append(replay_action)
                    action_str = get_action_description(replay_action)
                    meta_data["action_history"].append(action_str)
                    meta_data["page"] = self.env.page
                    
                    obs, reasoning, terminated, done, info, current_url = self.env.step(
                        replay_action, observation=obs, old_info=info, tool_llm=self.evaluate_model
                    )
                    if not done and reasoning:
                        meta_data['error_message'] = reasoning
                    else:
                        meta_data.pop('error_message', None)
                    state_info = {"observation": obs, "info": info, "current_url": current_url}
                    trajectory.append(state_info)
                    try:
                        render_helper.render(
                            replay_action, state_info, meta_data, self.args.render_screenshot
                        )
                    except Exception as e:
                        self.logger.error(f"Error rendering replayed screenshot: {e}")
                    # if terminated or not done:
                    #     self.logger.warning(
                    #         f"Replay action at index {replay_idx} ended unexpectedly (terminated={terminated}, done={done})"
                    #     )
                    #     break
            
            while True:
                current_url = current_url.lower()
                early_stop_flag, stop_info = early_stop(
                    trajectory,
                    self.args.max_steps,
                    {
                        "parsing_failure": self.args.parsing_failure_th,
                        "repeating_action": self.args.repeating_action_failure_th,
                    },
                )
                
                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    intent_for_step = enhanced_intent
                    if 'error_message' in meta_data:
                        intent_for_step += (
                            f" The error message from the previous action is: {meta_data['error_message']}, "
                            "please try another action."
                        )
                    action, meta_data = self.agent.next_action_custom(
                        trajectory,
                        intent_for_step,
                        meta_data=meta_data,
                    )
                
                trajectory.append(action)
                
                action_str = get_action_description(action)
                try:
                    render_helper.render(
                        action, state_info, meta_data, self.args.render_screenshot
                    )
                except Exception as e:
                    self.logger.error(f"Error rendering screenshot: {e}")
                meta_data["action_history"].append(action_str)
                meta_data["page"] = self.env.page
                
                if isinstance(action, list):
                    last_action_type = action[-1]["action_type"]
                else:
                    last_action_type = action["action_type"]
                if last_action_type in [ActionTypes.STOP, 'finished']:
                    self.logger.info(f"[Attempt {attempt_idx + 1}] Completed")
                    break
                try:
                    done = False
                    terminated = False
                    max_retries = 3
                    for i in range(max_retries):
                        obs, reasoning, terminated, done, info, current_url = self.env.step(action, observation=obs, old_info=info, tool_llm=self.evaluate_model)
                        if done:
                            break
                        time.sleep(0.1)
                    if not done:
                        meta_data['error_message'] = reasoning
                except Exception as e:
                    self.logger.error(f"Error in step: {e}")
                    terminated = False
                    done = False
                    # Use the last known values from trajectory if available
                    if trajectory and len(trajectory) > 0:
                        last_state = trajectory[-1]
                        if isinstance(last_state, dict):
                            if 'observation' in last_state:
                                obs = last_state['observation']
                            if 'info' in last_state:
                                info = last_state['info']
                                if 'page' in info and hasattr(info['page'], 'url'):
                                    current_url = info['page'].url
                # observation, 0.0, done, truncated, info
                print("CURRENT: ", current_url)

                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)
                
                if terminated:
                    trajectory.append(create_stop_action(""))
                    self.logger.info(f"[Attempt {attempt_idx + 1}] Terminated")
                    break

            # Ensure render file is copied to aggregate path before evaluation
            # Evaluator expects render_{task_id}.html, but RenderHelper creates render_{task_id}_attempt{N}.html
            try:
                aggregate_path = Path(self.args.result_dir) / f"render_{task_meta['task_id']}.html"
                # Flush the render file before copying
                render_helper.render_file.flush()
                shutil.copyfile(render_helper.render_path, aggregate_path)
            except Exception as copy_err:
                self.logger.warning(f"Failed to pre-copy render file for evaluation: {copy_err}")

            try:
                score, answer_text, ori_answer = self.evaluator(config_file, self.args.result_dir)
            except Exception as e:
                self.logger.error(f"Error in evaluator: {e}")
                score, answer_text, ori_answer = 0.0, "Error in evaluator", "Error in evaluator"
            score_for_render = score
            
            last_action = trajectory[-1] if trajectory else {}
            pred = last_action.get("answer", "") if isinstance(last_action, dict) else ""
            reasoning = last_action.get("reasoning", "") if isinstance(last_action, dict) else ""
            self.logger.info(f"[Attempt {attempt_idx + 1}] Predicted answer: {pred}\nReasoning: {reasoning}")
            
            result = "PASS" if score == 1 else "FAIL"
            self.logger.info(f"[Attempt {attempt_idx + 1}] ({result}) {config_file}")
            
            return {
                "score": score,
                "answer_text": answer_text,
                "ori_answer": ori_answer,
                "trajectory": trajectory,
                "current_url": info["page"].url if info and "page" in info else "",
                "pred": pred,
                "reasoning": reasoning,
                "render_path": render_helper.render_path
            }
        finally:
            # Close render file and add final score to HTML
            render_helper.close(score_for_render)
            try:
                # Update aggregate file with the final scored version
                aggregate_path = Path(self.args.result_dir) / f"render_{task_meta['task_id']}.html"
                shutil.copyfile(render_helper.render_path, aggregate_path)
            except Exception as copy_err:
                self.logger.warning(f"Failed to update render file with final score: {copy_err}")
    
    def run(self, config_file_list: list[str]):
        """Run the main test loop"""
        # Process each config file
        for config_file in config_file_list:
            self._process_config_file(config_file)
        # Close environment
        self.env.close()

        # Compute and display success rate
        self._compute_and_display_success_rate()
    
    def _process_config_file(self, config_file: str):
        """Process a single config file"""
        with open(config_file) as f:
            config_data = json.load(f)
            base_intent = config_data["intent"]
            task_id = config_data["task_id"]
            site = config_data["site"]
        
        episode_id = f"{site}_{task_id}"
        numbers = re.findall(r'\d+', config_file)
        self.args.task_cnt = int(numbers[0]) if numbers else None
        self.args.hop_cnt = 0
        
        self.logger.info(f"[Config file]: {config_file}")
        self.logger.info(f"[Intent]: {base_intent}")

        collector = None
        # Note: render_helper is created and closed within _run_one_attempt

        # TODO: Reasoning bank code moved to after score is computed
        # End-of-episode distillation into Reasoning Bank (optional)
        # FIXME: This block was misplaced - it tries to use 'score' before it's defined
        # Commenting out to allow basic evaluation to work
        if False:  # Disabled - was trying to use undefined variables
            try:
                if getattr(self.args, 'use_reasoning_bank', False):
                    is_success = bool(score == 1.0)
                
                # Get trajectory from training collector if available
                trajectory_obj = None
                if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
                    collector = self.agent.training_collector
                    if collector and collector.enabled and hasattr(collector, 'conversation_history'):
                        # Build trajectory object from collector data
                        trajectory_obj = {
                            'task_description': intent,
                            'rounds': collector.conversation_history
                        }
                
                # Load bank and distill
                bank = ReasoningBank(
                    bank_path=getattr(self.args, 'reasoning_bank_path', 'memory/reasoning_bank.jsonl'),
                    index_base_path=getattr(self.args, 'reasoning_index_base', 'memory_index/reasoning_bank_text'),
                    use_multimodal=getattr(self.args, 'reasoning_bank_multimodal', False)
                )
                prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent", "prompts")
                if not os.path.exists(prompts_dir):
                    # fallback relative to project root
                    prompts_dir = "agent/prompts"
                
                items = []
                # Use multimodal distillation if trajectory object is available
                if trajectory_obj and getattr(self.args, 'reasoning_bank_multimodal', False):
                    self.logger.info("[ReasoningBank] Using multimodal distillation")
                    items = distill_multimodal_reasoning_items(
                        tool_llm=self.evaluate_model,
                        prompts_dir=prompts_dir,
                        trajectory_obj=trajectory_obj,
                        is_success=is_success,
                        dataset="webvoyager",
                        domain=site,
                        task_id=str(task_id),
                        source_path=config_file,
                        max_items=2,
                        use_visual_stage1=True
                    )
                else:
                    # Fallback to text-only distillation
                    self.logger.info("[ReasoningBank] Using text-only distillation (no trajectory object)")
                    response_history = meta_data.get('response_history', [])
                    if isinstance(response_history, list) and response_history:
                        parts = [str(resp)[:400] for resp in response_history[:30] if resp]
                        trajectory_text = "\n---\n".join(parts)
                    else:
                        trajectory_text = ""
                    
                    from memory.reasoning_bank import distill_reasoning_items
                    items = distill_reasoning_items(
                        tool_llm=self.evaluate_model,
                        prompts_dir=prompts_dir,
                        is_success=is_success,
                        query=intent,
                        trajectory_text=trajectory_text,
                        dataset="webvoyager",
                        domain=site,
                        task_id=str(task_id),
                        source_path=config_file,
                        max_items=3
                    )
                
                if items:
                    # Log distilled items summary before writing
                    try:
                        if 'key_takeaway' in items[0]:
                            takeaways = [it.get("key_takeaway", "")[:80] for it in items]
                            self.logger.info(f"[ReasoningBank] distilled {len(items)} multimodal items "
                                             f"(label={'success' if is_success else 'failure'}) "
                                             f"for task_id={task_id}: {takeaways}")
                        else:
                            titles = [it.get("title", "") for it in items]
                            self.logger.info(f"[ReasoningBank] distilled {len(items)} text items "
                                             f"(label={'success' if is_success else 'failure'}) "
                                             f"for task_id={task_id}: {titles}")
                    except Exception:
                        pass
                    bank.add_items(items, persist=True, update_index=True)
                    # Log persistence and index update details
                    try:
                        self.logger.info(f"[ReasoningBank] bank updated: path={bank.bank_path}, "
                                         f"index={bank.index_path}, total_items={len(bank.items)}")
                    except Exception:
                        pass
                else:
                    self.logger.info("[ReasoningBank] no items distilled for this episode")
            except Exception as e:
                self.logger.error(f"Reasoning bank distillation failed: {e}")
                import traceback
                traceback.print_exc()

        # End conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            collector = self.agent.training_collector
            if collector and collector.enabled:
                conversation_id = f"{site}_{config_file.split('/')[-1].split('.')[0]}".replace(' ', '_')
                collector.start_conversation(
                    conversation_id=conversation_id,
                    task_description=base_intent
                )
                self.logger.info(f"Started conversation collection for task: {conversation_id}")
        
        max_attempts = 1
        if self.args.use_error_detection:
            max_attempts = getattr(self.args, "iterative_debug_attempts", 3)
        feedback_history: List[str] = []
        resume_info: Optional[Dict[str, Any]] = None
        final_result: Optional[Dict[str, Any]] = None
        
        for attempt_idx in range(max_attempts):
            attempt_result = self._run_single_attempt(
                config_file=config_file,
                base_intent=base_intent,
                attempt_idx=attempt_idx,
                feedback_history=feedback_history,
                task_meta={"task_id": task_id, "site": site},
                resume_info=resume_info
            )
            final_result = attempt_result
            
            if attempt_result["score"] == 1:
                self.logger.info(f"Task succeeded on attempt {attempt_idx + 1}")
                break
            
            if not self.args.use_error_detection:
                break
            
            if attempt_idx == max_attempts - 1:
                break
            
            feedback_result = self._generate_iterative_feedback(
                attempt_result["render_path"],
                episode_id,
                attempt_idx
            )
            if not feedback_result:
                self.logger.info("No feedback generated; stopping iterative debugging.")
                break
            feedback_text, critical = feedback_result
            feedback_history.append(feedback_text)
            resume_info = {
                "critical_step": critical.get("critical_step", 1),
                "trajectory": attempt_result["trajectory"],
            }
        
        if not final_result:
            self.logger.warning(f"No result generated for {config_file}")
            return
        
        current_url = final_result.get("current_url", "")
        score = final_result["score"]
        
        if collector and collector.enabled and collector.current_conversation_id:
            conversation_summary = {
                "task_id": config_file.split('/')[-1].split('.')[0],
                "site": site,
                "sub_domain": '',
                "success": score,
                "final_url": current_url,
                "task_completed": True,
                "task_description": base_intent
            }
            
            if self.args.save_examples_memory:
                saved_file = collector.end_conversation(conversation_summary, score)
                if saved_file:
                    self.logger.info(f"Conversation saved: {saved_file}")
                else:
                    self.logger.info("Conversation not saved")
            else:
                self.logger.info("not save_examples_memory")
    
    