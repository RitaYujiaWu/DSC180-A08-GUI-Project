"""Simplified evaluation system for GUI Agent"""
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Union
import re
from PIL import Image
from PIL import ImageDraw
from io import BytesIO
import requests
from beartype import beartype
from playwright.sync_api import CDPSession, Page
from openai import OpenAI
from crawl4ai import AsyncWebCrawler
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from browser_env import Action, Trajectory, StateInfo

class Evaluator:
    """Base class for evaluation"""
    
    def __init__(self, eval_tag: str = "") -> None:
        self.eval_tag = eval_tag

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page,
        client: CDPSession,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        """Get the last action from trajectory"""
        if not trajectory or not isinstance(trajectory[-1], dict):
            raise ValueError("The last element of trajectory should be an action")
        return trajectory[-1]

    @staticmethod
    def get_last_state(trajectory: Trajectory) -> StateInfo:
        """Get the last state from trajectory"""
        if len(trajectory) < 2 or not isinstance(trajectory[-2], dict):
            raise ValueError("The second last element of trajectory should be a state")
        return trajectory[-2]


class LLMEvaluator(Evaluator):
    """Check whether the answer is correct with exact match, must include, and fuzzy match"""
    
    def __init__(self, vllm_client=None):
        super().__init__()
        self.vllm_client = vllm_client

    def __call__(
        self,
        # trajectory: Trajectory,
        config_file: Path | str,
        html_folder: Path | str,
        # page: Page | None = None,
        # client: CDPSession | None = None,
    ) -> float:

        with open(config_file, "r") as f:
            configs = json.load(f)
            task_id = configs.get("task_id", "")
            intent = configs.get("intent", "")
        print(f"task_id: {task_id}, intent: {intent}")
        html_file = os.path.join(html_folder, f"render_{task_id}.html")
        with open(html_file, "r") as f:
            html_content = f.read()
        image_bs64s = self.extract_and_validate_images(html_content)
        answer = self.extract_answer(html_content)
        print(f"answer: {answer}")
        SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the last 5 screens showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'.
You should provide the 'SUCCESS' or 'NOT SUCCESS' between <result> and </result>."""

        USER_PROMPT = """TASK: {task}
        Result Response: {answer}"""

        user_prompt = USER_PROMPT.format(task=intent, answer=answer)
        print(f"user_prompt: {user_prompt}")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt}]
            }]
        image_contents = []
        for image_bs64 in image_bs64s[-5:]:
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_bs64['base64_data']}"}})
        messages.append({'role': 'user', 'content': image_contents})
        response, _, _ = self.vllm_client.chat(
            messages=messages,
            temperature=0.8,
            max_tokens=1024
        )
        print('response', response)
        answer_text = response.content.strip().lower()
        try:
            extracted_answer = re.search(r'<result>(.*?)</result>', answer_text).group(1)
            if not extracted_answer:
                extracted_answer = answer_text
        except:
            extracted_answer = answer_text
        if 'not success' in extracted_answer:
            return 0.0, answer_text, answer
        elif 'success' in extracted_answer:
            return 1.0, answer_text, answer
        else:
            print(f"Could not parse success/not success from response: '{answer_text}'")
            return 0.0, answer_text, answer
    
    def analyze_error_reasoning(self, config_file, html_folder):
        with open(config_file, "r") as f:
            configs = json.load(f)
            task_id = configs.get("task_id", "")
            intent = configs.get("intent", "")
        html_file = os.path.join(html_folder, f"render_{task_id}.html")
        with open(html_file, "r") as f:
            html_content = f.read()
        image_bs64s = self.extract_and_validate_images(html_content)
        actions = self.extract_action(html_content)
        SYSTEM_PROMPT = """You are an expert GUI agent analyst. Your task is to analyze a failed interaction trajectory and identify the most likely cause of failure based on visual evidence and action history.

### Input:
Below is a sequence of screenshots and actions taken by an agent while trying to complete a GUI task. Texts in the screenshots are the action taken by the agent at that step. The task ultimately failed.

- For each step, you will be provided:
  - Screenshot (Step N)
  - Action taken (e.g., "click button at index 5", "input text", etc.)

### Instructions:
1. Carefully examine the screenshots and the actions taken.
2. Identify the most likely reason for failure.
3. Be precise and concise in your diagnosis. Use GUI terminology.
4. If the agent clicked the wrong element, missed a visible button, or was blocked by an overlay, mention it.
5. If the failure is due to UI layout issues, state what was missing or misaligned.

### Output format:
<Failure Tag>Tag the failure type, such as "Popup Block", "Click Error", "Missed Button", "Overlay Block", "Repeated Click", "etc."</Failure Tag>
<Failure Reason>Short, clear description of the issue</Failure Reason>
<Supporting Evidence><Explain briefly what in the screenshots or actions indicates this</Supporting Evidence>

### Example Output:
<Failure Tag>Popup Block</Failure Tag>
<Failure Reason>Pop-up dialog blocked the screen, and the agent failed to dismiss it.</Failure Reason>
<Supporting Evidence>At Step 3, a visible pop-up with a close icon (index 9) appears. The agent tried to click a background element, which was unclickable due to the overlay.</Supporting Evidence>

---
"""
        user_prompt = """The task description is: {task}
        The actions taken by the agent are: {actions}
        The screenshots are as follows:"""
        actions_str = '\n'.join([f"Step {i+1}: {action}" for i, action in enumerate(actions)])
        user_prompt = user_prompt.format(task=intent, actions=actions_str)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt}]
            }]
        image_contents = []
        for image_bs64, action in zip(image_bs64s, actions):
            image_data = base64.b64decode(image_bs64['base64_data'])
            img = Image.open(BytesIO(image_data))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), action, fill='black')
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"}})
        messages.append({'role': 'user', 'content': image_contents})
        response, _, _ = self.vllm_client.chat(
            messages=messages,
            temperature=1,
            max_tokens=1024
        )
        response = response.content.strip()
        try:
            tag = re.search(r'<Failure Tag>(.*?)</Failure Tag>', response).group(1)
        except:
            tag = ''
        print('tag', tag)
        print('response', response)
        return tag, response
    
    async def locate_error_step(self, config_file, html_folder):
        with open(config_file, "r") as f:
            configs = json.load(f)
            task_id = configs.get("task_id", "")
            intent = configs.get("intent", "")
        html_file = os.path.join(html_folder, f"render_{task_id}.html")
        with open(html_file, "r") as f:
            html_content = f.read()
        image_bs64s = self.extract_and_validate_images(html_content)
        actions = self.extract_action(html_content)
        urls = self.extract_url(html_content)
        SYSTEM_PROMPT = """You are an expert GUI agent analyst. Your task is to analyze a failed interaction trajectory and identify the most likely cause of failure based on visual evidence and action history.

### Input:
Below is a sequence of screenshots and actions taken by an agent while trying to complete a GUI task. Texts in the screenshots are the action taken by the agent at that step. The task ultimately failed.

- For each step, you will be provided:
  - Screenshot (Step N)
  - Action taken (e.g., "click button at index 5", "input text", etc.)

### Instructions:
1. Carefully examine the screenshots and the actions taken.
2. Identify if the failure falls into one of these critical categories:
   - Click Failure: Agent failed to click on a necessary element
   - Type Failure: Agent failed to correctly type text into an input field
   - Selection Failure: Agent failed to select an option from dropdown/menu
   - Popup Handling Failure: Agent failed to close or interact with a pop-up window
   - Sort Failure: Agent failed to properly sort results/items
   - Filter Failure: Agent failed to apply appropriate filters

3. Be precise about the step number where the failure occurred.
4. Explain exactly what action should have been taken instead.
5. If the failure doesn't match any of the categories above, use "Other" and explain.

### Output format:
<Failure Tag>Tag the failure type, such as "Click Failure", "Type Failure", "Selection Failure", "Popup Handling Failure", "Sort Failure", "Filter Failure", or "Other"</Failure Tag>
<Failure Step>Step number where the failure occurred, only include numerical step number and no other text. If there are multiple steps, return step numbers separated by commas.</Failure Step>
<Failure Reason>Short, clear description of the issue</Failure Reason>
<Supporting Evidence>Explain briefly what in the screenshots or actions indicates this</Supporting Evidence>
<Correct Action>What should have been done instead</Correct Action>

### Example Output:
<Failure Tag>Type Failure</Failure Tag>
<Failure Step>4</Failure Step>
<Failure Reason>Agent failed to type "Ohio" in the destination field</Failure Reason>
<Supporting Evidence>In Step 4 screenshot, we can see the misspelled text "Ohoi" in the destination field, which returned no results in Step 5</Supporting Evidence>
<Correct Action>Agent should have typed "Ohio" correctly in the destination field</Correct Action>

---"""
        user_prompt = """The task description is: {task}
        The actions taken by the agent are: {actions}
        The screenshots are as follows:"""
        actions_str = '\n'.join([f"Step {i+1}: {action}" for i, action in enumerate(actions)])
        user_prompt = user_prompt.format(task=intent, actions=actions_str)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt}]
            }]
        image_contents = []
        for image_bs64, action in zip(image_bs64s, actions):
            image_data = base64.b64decode(image_bs64['base64_data'])
            img = Image.open(BytesIO(image_data))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), action, fill='black')
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"}})
        messages.append({'role': 'user', 'content': image_contents})
        response, _, _ = self.vllm_client.chat(
            messages=messages,
            temperature=1,
            max_tokens=1024
        )
        response = response.content.strip()
        def extract_tag(text, tag_name, default=''):
            pattern = f'<{tag_name}>(.*?)</{tag_name}>'
            match = re.search(pattern, text)
            return match.group(1) if match else default

        # Extract all needed tags with a single pattern approach
        tag = extract_tag(response, 'Failure Tag')
        step = extract_tag(response, 'Failure Step')  
        correct_action = extract_tag(response, 'Correct Action')
        print('tag', tag)
        print('step', step)
        print('correct_action', correct_action)
        # generate related queries for the error step
        related_queries = []
        ori_url = ''
        if ',' in step:
            steps = step.split(',')
            for step in steps:
                try:
                    step = int(step.strip())
                    screenshot_bs64 = image_bs64s[step-1]
                    url = urls[step-1]
                    if url == ori_url:
                        continue
                    ori_url = url
                    failed_action = actions[step-1]
                except:
                    continue
                # new_related_queries = self.generate_error_step_queries(intent, failed_action, screenshot_bs64)
                # new_related_queries = await self.filter_related_queries(screenshot_bs64, url, new_related_queries)
                rewritten_task, template, queries = await self.generate_grounded_template_queries(failed_action, screenshot_bs64, url)
                related_queries.append({'step': step, 'url': url, 'rewritten_task': rewritten_task, 'template': template, 'queries': queries})
        else:
            try:
                step = int(step.strip())
                screenshot_bs64 = image_bs64s[step-1]
                url = urls[step-1]
                failed_action = actions[step-1]
                # new_related_queries = self.generate_error_step_queries(intent, failed_action, screenshot_bs64)
                # new_related_queries = await self.filter_related_queries(screenshot_bs64, url, new_related_queries)
                rewritten_task, template, queries = await self.generate_grounded_template_queries(failed_action, screenshot_bs64, url)
                related_queries.append({'step': step, 'url': url, 'rewritten_task': rewritten_task, 'template': template, 'queries': queries})
            except Exception as e:
                print(f"Error generating related queries: {e}")
                pass
        return tag, step, correct_action, response, related_queries 
    
    def generate_error_step_queries(self, intent, failed_action, screenshot_bs64):
        SYSTEM_PROMPT = """You are an expert in GUI testing and training data generation. Your task is to generate diverse, related queries that could help train an agent to avoid specific types of GUI interaction failures.
### Input:
You will be given:
1. A description of a failed task
2. The failed action
3. A screenshot of the current step
### Instructions:
1. Generate 5-10 alternative queries that focus on the same type of interaction that failed
2. Ensure the queries:
   - Target the same failure mode (clicking, typing, selection, popup handling, sorting, filtering)
   - Are specific, atomic and actionable. Can be finished in one to two steps.
   - Cover cases related to the failure

### Output format:
Put one query per line.
<Related Queries>
...
</Related Queries>

### Example:
If the original failure was:
- Task: "Book a hotel in Ohio from Oct 1 to Oct 3"
- Failed Action: Agent failed to type "Ohio" correctly

<Related Queries>
Type "New York City" in the destination field
Type "Las Vegas, Nevada" in the destination field
Type "Toronto, Canada" in the destination search
......
</Related Queries>
---"""
        user_prompt = """The task description is: {task}
        The failed action is: {failed_action}
        The screenshot is as follows:"""
        user_prompt = user_prompt.format(task=intent, failed_action=failed_action)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_bs64['base64_data']}"}},
                {"type": "text", "text": "What are some alternative queries that could help train an agent to avoid this failure?"}
            ]}]
        response, _, _ = self.vllm_client.chat(
            messages=messages,
            temperature=1,
            max_tokens=1024
        )
        response = response.content.strip()
        print('response', response)
        related_queries = re.search(r'<Related Queries>(.*?)</Related Queries>', response, re.DOTALL)
        if related_queries:
            related_queries = related_queries.group(1)
            # Filter out empty lines and numbered prefixes
            related_queries = [re.sub(r'^\d+\.\s*', '', query.strip()) for query in related_queries.split('\n') if query.strip()]
            # Remove any lines that don't look like queries (e.g., empty lines)
            related_queries = [query for query in related_queries if query]
            print('related_queries', related_queries)
            return related_queries
        else:
            return []
        
    async def filter_related_queries(self, screenshot_bs64, url, related_queries):
        SYSTEM_PROMPT = """You are an expert in GUI testing and training data generation. Your task is to filter related queries based on the url.
### Input:
You will be given:
1. The screenshot of the current step
2. The text based content of the screenshot
3. Generated query
### Instructions:
Carefully examine the screenshot and the text based content. Determine if the generated query can be solved in this website. If yes, return "Yes". If no, return "No".---"""
        user_prompt = """The screenshot is as follows:
        The text based content is as follows: {text_content}
        """
        text_content = await self.get_page_content(url)
        user_prompt = user_prompt.format(text_content=text_content[:5000])
        for query in related_queries:
            messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": user_prompt},
                                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_bs64['base64_data']}"}},
                                         {"type": "text", "text": f"The generated query is: {query}. Determine if the generated query can be solved in this website. If yes, return 'Yes'. If no, return 'No'."}]}]
            response, _, _ = self.vllm_client.chat(
                messages=messages,
                temperature=1,
                max_tokens=1024
            )
            response = response.content.strip()
            if not 'yes' in response.lower():
                related_queries.remove(query)
        return related_queries
    
    async def generate_grounded_template_queries(self, failed_action: str, screenshot_bs64: Dict[str, Any], url: str, max_queries: int = 8) -> Dict[str, Any]:
        """
        Generate solvable, grounded related queries by:
        1) Rewriting the failed action into a positive, solvable task.
        2) Converting the task into a parameterized template with explicit slots.
        3) Imputing parameter values strictly from the page text content and the screenshot.
        4) Returning grounded queries that are actually solvable on the current page.

        Args:
            failed_action: The textual description of the failed action.
            screenshot_bs64: A dict with key 'base64_data' for the current step screenshot.
            url: Page URL to extract text content for grounding and solvability.
            max_queries: Maximum number of grounded queries to return.

        Returns:
            Dict with keys:
              - 'rewritten_task': str
              - 'template': str
              - 'queries': List[str] (filtered to be solvable)
        """
        text_content = await self.get_page_content(url)
        if text_content is None:
            text_content = ""

        SYSTEM_PROMPT = """You are an expert in GUI task synthesis for RL training. Your job is to derive solvable, grounded queries from a failed action on a specific webpage.

Strict rules:
- You must ground parameter values ONLY in the provided page text. Do not invent values.
- Prefer values that clearly appear in the text_content (exact strings, visible price ranges, categories, dates, options, labels).
- If the page does not provide enough information to fill parameters, mark the output as unsolvable by returning an empty <Grounded Queries> block.
- Each grounded query must be realistically solvable in 1-2 steps on this page.
- Use the screenshot as a secondary hint; decisions must be verifiable from text_content.

Steps:
1) Rewrite the failed action into a clear, solvable task command (imperative voice).
2) Generalize that task into a parameterized template with explicit slots and slot options. Example: "please filter the price for the product to {under|above|between} {range}".
3) From text_content, extract valid, concrete parameter fillings (e.g., exact price points/ranges, categories, sort keys) that appear verbatim.
4) Instantiate 3-10 grounded queries by filling the template with those values. Keep them atomic and solvable.

Output format:
<Rewritten Task>...</Rewritten Task>
<Template>...</Template>
<Grounded Queries>
query 1
query 2
...
</Grounded Queries>
"""

        USER_PROMPT = """Failed action: {failed_action}
Page URL: {url}
Page text_content (truncated):
"""

        # Keep prompt size reasonable
        max_text_len = 8000
        truncated_text = text_content[:max_text_len]
        user_prompt = USER_PROMPT.format(failed_action=failed_action, url=url) + truncated_text

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_bs64['base64_data']}"}},
                ],
            },
        ]

        response, _, _ = self.vllm_client.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        content = (response.content or "").strip()

        try:
            rewritten_task_match = re.search(r"<Rewritten Task>(.*?)</Rewritten Task>", content, re.DOTALL)
            template_match = re.search(r"<Template>(.*?)</Template>", content, re.DOTALL)
            grounded_block_match = re.search(r"<Grounded Queries>(.*?)</Grounded Queries>", content, re.DOTALL)
            rewritten_task = rewritten_task_match.group(1).strip() if rewritten_task_match else ""
            template = template_match.group(1).strip() if template_match else ""
            grounded_block = grounded_block_match.group(1) if grounded_block_match else ""
            queries = [re.sub(r"^\d+\.[\s\-]*", "", q.strip()) for q in grounded_block.split("\n") if q.strip()]
        except Exception:
            rewritten_task, template, queries = "", "", []

        # # Trim and deduplicate
        # unique_queries = []
        # seen = set()
        # for q in queries:
        #     if q and q not in seen:
        #         unique_queries.append(q)
        #         seen.add(q)
        # if max_queries and len(unique_queries) > max_queries:
        #     unique_queries = unique_queries[:max_queries]

        # Validate solvability via existing filter
        try:
            queries = await self.filter_related_queries(screenshot_bs64, url, queries)
        except Exception:
            pass

        return rewritten_task, template, queries
    
    def extract_and_validate_images(self, html_content):
        # Find all image src attributes containing base64 data
        match_pattern = r"<img\s+[^>]*src=['\"]data:image/[^;]+;base64,([^'\"]+)['\"][^>]*>"
        matches = re.findall(match_pattern, html_content)
        
        valid_images = []
        
        for base64_data in matches:
            try:
                # Decode base64 string
                image_data = base64.b64decode(base64_data)
                
                # Try to open as an image to validate
                img = Image.open(BytesIO(image_data))
                
                # If we get here, it's a valid image
                # You can add additional validation if needed (e.g., check dimensions)
                img_info = {
                    "base64_data": base64_data,
                    "format": img.format,
                    "size": img.size,
                    "mode": img.mode
                }
                
                valid_images.append(img_info)
                
                # Optional: save the image to disk
                # img.save(f"image_{len(valid_images)}.{img.format.lower()}")
                
            except Exception as e:
                print(f"Invalid or corrupted image data: {str(e)}")
                continue
        
        return valid_images
    
    def extract_action(self, html_content):
        match_pattern = r"<div class='parsed_action'.*?><pre>(.*?)</pre></div>"
        valid_actions = re.findall(match_pattern, html_content)
        return valid_actions
    
    def extract_url(self, html_content):
        href_pattern = r'<a href=(https?://[^>]+)>'
        href_match = re.findall(href_pattern, html_content)
        return href_match
    
    def extract_answer(self, html_content):
        match_pattern = r"finished\(answer=(.*?)\)"
        answer = re.search(match_pattern, html_content)
        if answer:
            answer = answer.group(1)
        else:
            answer = ''
        return answer
        
    async def get_page_content(self, url: str):
        """
        Get page content using PageParserTool and check for blocking
        
        Args:
            url: URL to parse
            screenshot: Base64 screenshot (not used in this implementation)
            
        Returns:
            Tuple of (content, is_blocked)
        """
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
            return result.markdown

        except Exception as e:
            return None

if __name__ == "__main__":
    from config.argument_parser import config
    from agent.llm_config import load_tool_llm
    args = config()
    evaluate_model = load_tool_llm(args, 'qwen2.5-vl-32b')
    evaluator = LLMEvaluator(vllm_client=evaluate_model)
    config_file = '/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/webvoyager_evaluation/data/test/test_ESPN--4.json'
    html_folder = '/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/results/webvoyager/test/qwen2.5-vl/20251017_052201'
    score, answer_text, ori_answer = evaluator(config_file, html_folder)
    print(score, answer_text, ori_answer)
    