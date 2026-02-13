# src/policy_hf.py
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional, Union
import io
import json
import re

import torch
import torch.nn as nn
from PIL import Image

from transformers import AutoProcessor

try:
    from transformers import AutoModelForVision2Seq  # type: ignore
except Exception:
    from transformers.models.auto.modeling_auto import AutoModelForVision2Seq  # type: ignore


# ----------------------------
# Helpers
# ----------------------------
def _extract_image_slots(messages: List[Dict[str, Any]]) -> int:
    """
    Count image placeholders in a HF chat-template message list.
    We count:
      - {"type":"image"}      (preferred: placeholder)
      - {"type":"image_url"}  (not preferred for training, can bloat prompt if base64)
    """
    n = 0
    for m in messages:
        content = m.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                    n += 1
    return n


def _decode_screenshot(obs: Any) -> Optional[Image.Image]:
    if not isinstance(obs, dict):
        return None
    ss = obs.get("screenshot", None)
    if ss is None:
        return None
    if isinstance(ss, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(ss)).convert("RGB")
        except Exception:
            return None
    if isinstance(ss, Image.Image):
        return ss.convert("RGB")
    return None


def _obs_text_compact(obs: Any, max_chars: int = 1200) -> str:
    if isinstance(obs, dict):
        o = dict(obs)
        if "screenshot" in o:
            o["screenshot"] = "<image>"
        return str(o)[:max_chars]
    return str(obs)[:max_chars]


def sanitize_action_fallback(model_text: str) -> str:
    if not model_text:
        return "pyautogui.sleep(0.5)"

    t = model_text.strip()
    t = t.replace("```python", "").replace("```py", "").replace("```", "")
    t = t.replace("\r\n", "\n").strip()
    t = re.sub(r"^(action|assistant)\s*:\s*", "", t, flags=re.IGNORECASE).strip()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in lines:
        ln = ln.strip("`").strip()
        ln = ln.split(";")[0].strip()
        if "pyautogui." in ln:
            m = re.search(r"(pyautogui\.[^\n]+)$", ln)
            ln2 = m.group(1).strip() if m else ln
            ln2 = ln2.split("#")[0].strip()
            return ln2
    return "pyautogui.sleep(0.5)"


def _extract_tool_call_json(response: str) -> Optional[str]:
    if not response:
        return None

    lines = response.splitlines()
    inside = False
    buf: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<tool_call>"):
            inside = True
            buf = []
            continue
        if line.startswith("</tool_call>"):
            inside = False
            s = "\n".join(buf).strip()
            return s if s else None
        if inside:
            buf.append(line)

    # tolerate single-line JSON without tags
    for raw in lines:
        line = raw.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if "name" in obj and "arguments" in obj:
                    return line
            except Exception:
                pass
    return None


def _pyautogui_from_tool_call(tool_call_json: str) -> Tuple[str, Dict[str, Any]]:
    obj = json.loads(tool_call_json)
    args = obj.get("arguments", {}) or {}
    action = args.get("action", "")

    coord = args.get("coordinate", None)
    wait_time = args.get("time", None)

    if action == "left_click":
        if isinstance(coord, list) and len(coord) == 2:
            x, y = coord
            return f"pyautogui.click({int(x)}, {int(y)})", obj
        return "pyautogui.click()", obj

    if action == "right_click":
        if isinstance(coord, list) and len(coord) == 2:
            x, y = coord
            return f"pyautogui.rightClick({int(x)}, {int(y)})", obj
        return "pyautogui.rightClick()", obj

    if action == "middle_click":
        if isinstance(coord, list) and len(coord) == 2:
            x, y = coord
            return f"pyautogui.middleClick({int(x)}, {int(y)})", obj
        return "pyautogui.middleClick()", obj

    if action == "double_click":
        if isinstance(coord, list) and len(coord) == 2:
            x, y = coord
            return f"pyautogui.doubleClick({int(x)}, {int(y)})", obj
        return "pyautogui.doubleClick()", obj

    if action == "mouse_move":
        if isinstance(coord, list) and len(coord) == 2:
            x, y = coord
            return f"pyautogui.moveTo({int(x)}, {int(y)})", obj
        return "pyautogui.moveTo(0, 0)", obj

    if action == "left_click_drag":
        if isinstance(coord, list) and len(coord) == 2:
            x, y = coord
            duration = args.get("duration", 0.5)
            try:
                dur = float(duration)
            except Exception:
                dur = 0.5
            return f"pyautogui.dragTo({int(x)}, {int(y)}, duration={dur})", obj
        return "pyautogui.dragTo(0, 0)", obj

    if action == "scroll":
        pixels = args.get("pixels", 0)
        try:
            px = int(pixels)
        except Exception:
            px = 0
        return f"pyautogui.scroll({px})", obj

    if action == "type":
        text = args.get("text", "")
        return f"pyautogui.typewrite({repr(str(text))})", obj

    if action == "key":
        keys = args.get("keys", [])
        if not isinstance(keys, list):
            keys = [keys]
        cleaned: List[str] = []
        for k in keys:
            if isinstance(k, str):
                cleaned.append(k.strip())
        if len(cleaned) >= 2:
            joined = ", ".join([repr(k) for k in cleaned])
            return f"pyautogui.hotkey({joined})", obj
        if len(cleaned) == 1:
            return f"pyautogui.press({repr(cleaned[0])})", obj
        return "pyautogui.sleep(0.2)", obj

    if action == "wait":
        try:
            t = float(wait_time) if wait_time is not None else 1.0
        except Exception:
            t = 1.0
        t = max(0.05, min(10.0, t))
        return f"pyautogui.sleep({t})", obj

    if action == "terminate":
        return "DONE", obj

    return "pyautogui.sleep(0.5)", obj


# ----------------------------
# Config
# ----------------------------
@dataclass
class PolicyConfig:
    model_name_or_path: str
    device: str
    max_new_tokens: int
    temperature: float
    top_p: float


def _get_hidden_size(model) -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size") and cfg.hidden_size is not None:
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise ValueError(f"Cannot find hidden_size in config: {cfg.__class__.__name__}")


# ----------------------------
# Actor-Critic VLM
# ----------------------------
class ActorCriticVLM(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Memory-friendly defaults for training
        try:
            self.model.config.use_cache = False
        except Exception:
            pass

        hidden = _get_hidden_size(self.model)
        self.v_head = nn.Linear(hidden, 1).to(dtype=self.model.dtype)

    def forward(self, **model_inputs):
        """
        IMPORTANT MEMORY FIX:
        - Avoid output_hidden_states=True (stores all layers).
        - Prefer last_hidden_state if present; otherwise fall back to hidden_states[-1].
        """
        out = self.model(
            **model_inputs,
            output_hidden_states=False,
            use_cache=False,
        )

        # Some models expose last_hidden_state; some don't.
        last_h = getattr(out, "last_hidden_state", None)
        if last_h is None:
            # Fall back: request hidden states ONLY if needed.
            out2 = self.model(
                **model_inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            if not hasattr(out2, "hidden_states") or out2.hidden_states is None:
                raise RuntimeError("Model output does not contain hidden states.")
            last_h = out2.hidden_states[-1]
            logits = out2.logits
        else:
            logits = out.logits

        last_h = last_h.to(dtype=self.v_head.weight.dtype)
        values = self.v_head(last_h).squeeze(-1)
        return logits, values


# ----------------------------
# HFPolicy
# ----------------------------
class HFPolicy:
    """
    OSWorld-style agent calling, but running locally with HuggingFace VLM.

    FIX:
    - DO NOT embed base64 data URLs into messages. Use {"type":"image"} placeholders.
      Then pass actual PIL images via processor(images=...).
    - PPO replay: we store prompts as messages with image placeholders, and separately
      store the exact images_list per step for evaluate().
    """

    def __init__(self, cfg: PolicyConfig, lr: float, history_n: int = 4):
        self.cfg = cfg
        self.history_n = int(history_n)
        self.accelerator = None

        self.net = ActorCriticVLM(cfg.model_name_or_path).to(cfg.device)
        self.processor = self.net.processor
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("Processor does not expose a tokenizer; cannot decode generations.")

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr)

        self._last_task_text: Optional[str] = None
        self._hist_responses: List[str] = []
        self._hist_imgs_png: List[bytes] = []

    def set_accelerator(self, accelerator):
        self.accelerator = accelerator

    def _unwrap_net(self) -> ActorCriticVLM:
        net = self.net
        if hasattr(net, "module"):
            return net.module  # DDP
        return net

    def reset_episode(self):
        self._hist_responses = []
        self._hist_imgs_png = []
        self._last_task_text = None

    # ---------- system prompt ----------
    def _osworld_system_prompt(self, screen_w: int, screen_h: int) -> str:
        description_prompt_lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            f"* The screen's resolution is {screen_w}x{screen_h}. Coordinates MUST be absolute pixels in this resolution.",
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* Make sure to click buttons/links/icons with the cursor tip in the center of the element.",
        ]
        description_prompt = "\n".join(description_prompt_lines)

        action_description_prompt = (
            "* `key`: Performs key presses.\n"
            "* `type`: Type a string.\n"
            "* `mouse_move`: Move cursor to (x,y).\n"
            "* `left_click`: Left click at (x,y).\n"
            "* `left_click_drag`: Drag to (x,y).\n"
            "* `right_click`: Right click.\n"
            "* `middle_click`: Middle click.\n"
            "* `double_click`: Double click.\n"
            "* `scroll`: Scroll pixels.\n"
            "* `wait`: Wait seconds.\n"
            "* `terminate`: Terminate task.\n"
        )

        tools_def = {
            "type": "function",
            "function": {
                "name_for_human": "computer_use",
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "properties": {
                        "action": {
                            "description": action_description_prompt,
                            "enum": [
                                "key", "type", "mouse_move",
                                "left_click", "left_click_drag",
                                "right_click", "middle_click",
                                "double_click", "scroll", "wait", "terminate",
                            ],
                            "type": "string",
                        },
                        "keys": {"description": "Required only by `action=key`.", "type": "array"},
                        "text": {"description": "Required only by `action=type`.", "type": "string"},
                        "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                        "pixels": {"description": "The amount of scrolling.", "type": "number"},
                        "time": {"description": "Seconds to wait.", "type": "number"},
                        "status": {"description": "Task status.", "type": "string", "enum": ["success", "failure"]},
                    },
                    "required": ["action"],
                    "type": "object",
                },
                "args_format": "Format the arguments as a JSON object.",
            },
        }

        system_prompt = (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{json.dumps(tools_def)}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block containing only the JSON: "
            '{"name": <function-name>, "arguments": <args-json-object>}.\n\n'
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing, use action=terminate in the tool call."
        )
        return system_prompt

    # ---------- image processing ----------
    def _process_image(self, img: Image.Image) -> Image.Image:
        """
        Return a resized PIL image (no base64).
        Keeping PIL images avoids prompt text bloat and is what processors expect.
        """
        max_pixels = 1600 * 1600
        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / float(w * h)) ** 0.5
            nw = max(32, (int(w * scale) // 32) * 32)
            nh = max(32, (int(h * scale) // 32) * 32)
            img = img.resize((nw, nh))
        return img.convert("RGB")

    def _pil_to_png_bytes(self, img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _png_bytes_to_pil(self, png_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")

    # ---------- message building (PLACEHOLDERS ONLY) ----------
    def _build_messages(
        self,
        obs: Any,
        task_text: str,
        history_text: str,
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """
        Returns:
          messages: list with {"type":"image"} placeholders only (NO base64 urls)
          images_list: PIL images aligned 1:1 with image placeholders in messages
        """
        img = _decode_screenshot(obs)
        if img is None:
            raise RuntimeError("VLM-only policy requires obs['screenshot'] bytes.")

        img = self._process_image(img)
        screen_w, screen_h = img.size

        # Auto-reset when task changes
        if self._last_task_text is None or task_text != self._last_task_text:
            self._hist_responses = []
            self._hist_imgs_png = []
            self._last_task_text = task_text

        obs_text = _obs_text_compact(obs)

        instruction_prompt = (
            "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {task_text}\n\n"
            "Previous actions / results:\n"
            f"{history_text if history_text.strip() else 'None'}\n\n"
            f"Observation (text summary):\n{obs_text}\n"
        )

        system_prompt = self._osworld_system_prompt(screen_w=screen_w, screen_h=screen_h)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]

        images_list: List[Image.Image] = []

        hist_len = min(self.history_n, len(self._hist_responses))
        if hist_len > 0:
            hist_resps = self._hist_responses[-hist_len:]
            hist_pngs = self._hist_imgs_png[-hist_len:]

            for i in range(hist_len):
                pil_i = self._process_image(self._png_bytes_to_pil(hist_pngs[i]))

                if i == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},  # placeholder only
                                {"type": "text", "text": instruction_prompt},
                            ],
                        }
                    )
                else:
                    messages.append(
                        {"role": "user", "content": [{"type": "image"}]}  # placeholder only
                    )

                images_list.append(pil_i)

                messages.append(
                    {"role": "assistant", "content": [{"type": "text", "text": hist_resps[i]}]}
                )

            # current image only
            messages.append({"role": "user", "content": [{"type": "image"}]})
            images_list.append(img)

        else:
            # no history: current image + instruction text
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # placeholder only
                        {"type": "text", "text": instruction_prompt},
                    ],
                }
            )
            images_list.append(img)

        n_slots = _extract_image_slots(messages)
        if n_slots != len(images_list):
            raise RuntimeError(f"[HFPolicy] image slot mismatch: slots={n_slots}, images={len(images_list)}")

        return messages, images_list

    # ---------- encoding ----------
    def _encode(
        self,
        messages: List[Dict[str, Any]],
        images: Union[None, Image.Image, List[Image.Image]],
    ) -> Dict[str, torch.Tensor]:
        """
        IMPORTANT:
        - messages contain only placeholders, so apply_chat_template produces compact text.
        - images passed separately to the processor.
        """
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        if images is None:
            image_list: List[Image.Image] = []
        elif isinstance(images, list):
            image_list = images
        else:
            image_list = [images]

        n_slots = _extract_image_slots(messages)

        if n_slots == 0:
            enc = self.processor(text=prompt, return_tensors="pt")
        else:
            if len(image_list) != n_slots:
                raise RuntimeError(f"[HFPolicy] expects {n_slots} images, but got {len(image_list)}.")
            enc = self.processor(text=prompt, images=image_list, return_tensors="pt")

        enc = {k: v.to(self.cfg.device) for k, v in enc.items() if torch.is_tensor(v)}
        return enc

    # ---------- scoring ----------
    def _score_action_text(
        self,
        messages: List[Dict[str, Any]],
        images_list: List[Image.Image],
        action_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_prompt = self._encode(messages, images_list)
        prompt_len = int(enc_prompt["input_ids"].shape[1])

        full_msgs = messages + [{"role": "assistant", "content": [{"type": "text", "text": action_text + "\n"}]}]
        enc_full = self._encode(full_msgs, images_list)

        logits, values = self.net(**enc_full)

        prompt_last_idx = max(0, prompt_len - 1)
        v = values[0, prompt_last_idx]

        input_ids = enc_full["input_ids"][0]
        attn = enc_full["attention_mask"][0]
        full_len = int(attn.sum().item())

        start_tok = prompt_len
        end_tok = full_len

        if end_tok - start_tok <= 0 or start_tok == 0:
            lp = torch.zeros((), device=self.cfg.device)
            ent = torch.zeros((), device=self.cfg.device)
            return lp, v, ent

        logprobs_all = torch.log_softmax(logits[0], dim=-1)
        tgt = input_ids[start_tok:end_tok]
        pred_pos = torch.arange(start_tok - 1, end_tok - 1, device=self.cfg.device)

        lp = logprobs_all[pred_pos, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()

        probs = torch.softmax(logits[0, pred_pos, :], dim=-1)
        ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()

        return lp, v, ent

    # ---------- main API ----------
    @torch.no_grad()
    def act(self, obs: Any, task_text: str, history_text: str) -> Tuple[str, float, float, Optional[bytes]]:
        messages, images_list = self._build_messages(obs, task_text, history_text)

        enc_prompt = self._encode(messages, images_list)

        # value at prompt end (forward ok through wrapped net)
        _, values = self.net(**enc_prompt)
        v_prompt_last = float(values[0, -1].item())

        # DDP-safe access to underlying HF model for generate()
        net_unwrapped = self._unwrap_net()
        model_for_gen = net_unwrapped.model

        gen_ids = model_for_gen.generate(
            **enc_prompt,
            max_new_tokens=int(self.cfg.max_new_tokens),
            do_sample=True,
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        prompt_len = enc_prompt["input_ids"].shape[1]
        gen_only = gen_ids[0, prompt_len:]
        response_text = self.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

        env_action = None
        tool_json = _extract_tool_call_json(response_text)
        if tool_json is not None:
            try:
                env_action, _obj = _pyautogui_from_tool_call(tool_json)
            except Exception:
                env_action = None
        if env_action is None:
            env_action = sanitize_action_fallback(response_text)

        lp, _v, _ent = self._score_action_text(messages, images_list, env_action)
        old_logprob = float(lp.item())

        # update history with CURRENT screenshot + raw response
        try:
            cur_img = _decode_screenshot(obs)
            if cur_img is not None:
                cur_img = self._process_image(cur_img)
                self._hist_imgs_png.append(self._pil_to_png_bytes(cur_img))
                self._hist_responses.append(
                    response_text if response_text else "Action: (empty)\n<tool_call>{}</tool_call>"
                )
                if len(self._hist_imgs_png) > self.history_n:
                    self._hist_imgs_png = self._hist_imgs_png[-self.history_n:]
                if len(self._hist_responses) > self.history_n:
                    self._hist_responses = self._hist_responses[-self.history_n:]
        except Exception:
            pass

        # optional: raw screenshot bytes for external logging/debug
        ss_bytes = None
        if isinstance(obs, dict) and isinstance(obs.get("screenshot", None), (bytes, bytearray)):
            ss_bytes = bytes(obs["screenshot"])

        return str(env_action), float(old_logprob), float(v_prompt_last), ss_bytes

    def update(self, loss: torch.Tensor, grad_clip_norm: float) -> Dict[str, float]:
        self.opt.zero_grad(set_to_none=True)

        if self.accelerator is not None:
            self.accelerator.backward(loss)
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                self.accelerator.clip_grad_norm_(self.net.parameters(), float(grad_clip_norm))
        else:
            loss.backward()
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(grad_clip_norm))

        self.opt.step()
        return {"loss": float(loss.item())}

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)

        net_unwrapped = self._unwrap_net()
        net_unwrapped.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        torch.save(net_unwrapped.v_head.state_dict(), os.path.join(path, "value_head.pt"))

    def evaluate(
        self,
        prompts: List[List[Dict[str, Any]]],
        actions: List[str],
        screenshots: List[Optional[bytes]],
        images_list_per_prompt: Optional[List[List[Image.Image]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO re-eval with gradients:
          - logprob(action | prompt,images) [B]
          - value(prompt_last_token)        [B]
          - entropy over action tokens      [B]

        IMPORTANT CHANGE:
        - Since prompts now contain ONLY image placeholders (no base64),
          we cannot recover images from prompts.
        - Therefore, pass `images_list_per_prompt` from your training loop.

        Backward compatibility:
        - If images_list_per_prompt is None, we fall back to screenshots[i] as a single image.
          This is imperfect when history images exist, but avoids hard crashes.
        """
        assert len(prompts) == len(actions) == len(screenshots)
        B = len(prompts)

        lp_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []
        ent_list: List[torch.Tensor] = []

        for i in range(B):
            msgs = prompts[i]

            if images_list_per_prompt is not None:
                imgs = images_list_per_prompt[i]
            else:
                imgs = []

            # Fallback: if no provided image list, try single screenshot
            if len(imgs) == 0:
                ss = screenshots[i]
                if ss is not None:
                    try:
                        imgs = [Image.open(io.BytesIO(ss)).convert("RGB")]
                    except Exception:
                        imgs = []

            if len(imgs) == 0:
                lp_list.append(torch.zeros((), device=self.cfg.device))
                v_list.append(torch.zeros((), device=self.cfg.device))
                ent_list.append(torch.zeros((), device=self.cfg.device))
                continue

            n_slots = _extract_image_slots(msgs)
            if n_slots != len(imgs):
                lp_list.append(torch.zeros((), device=self.cfg.device))
                v_list.append(torch.zeros((), device=self.cfg.device))
                ent_list.append(torch.zeros((), device=self.cfg.device))
                continue

            lp, v, ent = self._score_action_text(msgs, imgs, actions[i])
            lp_list.append(lp)
            v_list.append(v)
            ent_list.append(ent)

        new_logprob = torch.stack(lp_list)
        v = torch.stack(v_list)
        entropy = torch.stack(ent_list)
        return new_logprob, v, entropy