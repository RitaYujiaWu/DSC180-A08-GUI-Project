import sys
import pathlib
path = pathlib.Path(__file__).parent.parent
sys.path.append(str(path))
from agent.llm_config import create_direct_vllm_model
from evaluator import LLMEvaluator
import os
import json
import argparse
import asyncio


class LLMEvaluation:
    def __init__(self, html_folder: str, args: argparse.Namespace):
        self.llm_client = create_direct_vllm_model(args, model_name=args.model)
        self.html_folder = html_folder
        self.config_folder = args.config_folder
        self.evaluator = LLMEvaluator(self.llm_client)
        if not os.path.exists(os.path.join(self.html_folder, f"llm_evaluation.json")):
            with open(os.path.join(self.html_folder, f"llm_evaluation.json"), 'r') as f:
                results = json.load(f)
                self.seen_task_ids = [item['task_id'] for item in results][:-1] # exclude the last one
                self.seen_task_ids = set(self.seen_task_ids)
        else:
            self.seen_task_ids = set()
        
    async def evaluate(self):
        if len(self.seen_task_ids) > 0:
            with open(os.path.join(self.html_folder, f"llm_evaluation.json"), 'r') as f:
                results = json.load(f)
                all_scores = [item['score'] for item in results]
            with open(os.path.join(self.html_folder, f"locate_error_steps.json"), 'w') as f:
                locate_error_steps = json.load(f)
            with open(os.path.join(self.html_folder, f"generate_related_queries_template.json"), 'w') as f:
                generate_related_queries = json.load(f)
        else:
            results = []
            all_scores = []
            error_reasonings = []
            locate_error_steps = []
            generate_related_queries = {}
        for config_file in os.listdir(self.config_folder):
            config_path = os.path.join(self.config_folder, config_file)
            with open(config_path, 'r') as f:
                config = json.load(f)
                task_id = config.get('task_id', '')
                if task_id in self.seen_task_ids:
                    print(f"Task {task_id} already evaluated")
                    continue
                self.seen_task_ids.add(task_id)
                html_file = os.path.join(self.html_folder, f"render_{task_id}.html")
            print(f"Evaluating task {task_id} with html file {html_file}")
            if not os.path.exists(html_file):
                continue
            score, answer_text, ori_answer = self.evaluator(config_path, self.html_folder)
            all_scores.append(score)
            results.append({'task_id': task_id, 'score': score, 'answer_text': answer_text, 'ori_answer': ori_answer})
            print(f"Task {task_id} score: {score}")
            with open(os.path.join(self.html_folder, f"llm_evaluation.json"), 'w') as f:
                json.dump(results, f, indent=4)
            # if score == 0:
            #     tag, error_reason = self.evaluator.analyze_error_reasoning(config_path, self.html_folder)
            #     error_reasonings.append({'task_id': task_id, 'tag': tag, 'error_reason': error_reason})
            #     with open(os.path.join(self.html_folder, f"error_reasonings.json"), 'w') as f:
            #         json.dump(error_reasonings, f, indent=4)
            if score == 0:
                tag, step, correct_action, response, related_queries = await self.evaluator.locate_error_step(config_path, self.html_folder)
                locate_error_steps.append({'task_id': task_id, 'tag': tag, 'step': step, 'correct_action': correct_action, 'response': response})
                with open(os.path.join(self.html_folder, f"locate_error_steps.json"), 'w') as f:
                    json.dump(locate_error_steps, f, indent=4)
                if len(related_queries) > 0:
                    generate_related_queries[task_id] = related_queries
                    with open(os.path.join(self.html_folder, f"generate_related_queries_template.json"), 'w') as f:
                        json.dump(generate_related_queries, f, indent=4)
        print('html_folder: ', self.html_folder)
        print(f"Average score: {sum(all_scores) / len(all_scores)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--html_folder', type=str, default='/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/results/webvoyager/qwen2.5-vl/test/test')
    parser.add_argument('--model', type=str, default='qwen2.5-vl-32b')
    parser.add_argument('--config_folder', type=str, default='/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/webvoyager_evaluation/data/test')
    args = parser.parse_args()
    llm_evaluation = LLMEvaluation(args.html_folder, args)
    asyncio.run(llm_evaluation.evaluate())