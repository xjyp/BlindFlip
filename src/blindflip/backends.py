"""Lazy local inference and an environment-keyed chat-completions client."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib import error, parse, request


def non_thinking_options(model):
    name = model.lower()
    if name.startswith("deepseek"):
        return {"reasoning": {"enabled": False}}
    if name.startswith("gemini"):
        return {"thinking": False}
    if name.startswith("qwen"):
        return {"enable_thinking": False}
    if name.startswith("gpt-"):
        return {"reasoning_effort": "none"}
    return {}


class MockJudge:
    """Synthetic plumbing check only; these responses are not research results."""

    def generate(self, messages, on_attempt):
        results = []
        for message in messages:
            user = message[-1]["content"]
            note_label = os.environ.get("BLINDFLIP_NOTE_LABEL", "Source Note")
            note = user.rsplit(f"[{note_label}]\n", 1)[-1]
            flip = note != "(empty)" and hashlib.sha256(user.encode()).digest()[0] < 160
            results.append({"raw": "Output (b)" if flip else "Output (a)", "error": None})
        return results


class VLLMJudge:
    def __init__(self, config):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
        if os.environ.get("OMP_NUM_THREADS", "") == "0":
            os.environ["OMP_NUM_THREADS"] = "1"
        from vllm import LLM, SamplingParams
        kwargs = dict(model=config["model"], dtype=config["dtype"],
                      max_model_len=config["max_model_len"],
                      gpu_memory_utilization=config["gpu_memory_utilization"],
                      tensor_parallel_size=config["tensor_parallel_size"],
                      trust_remote_code=config["trust_remote_code"],
                      enforce_eager=True, seed=config["inference_seed"], swap_space=4)
        if config.get("revision"):
            kwargs["revision"] = config["revision"]
        self.llm = LLM(**kwargs)
        self.sampling = SamplingParams(temperature=0.0, top_p=1.0,
                                       max_tokens=config["max_tokens"])
        self.chat_kwargs = ({"enable_thinking": False}
                            if "qwen3" in config["model"].lower() else None)

    def generate(self, messages, on_attempt):
        for index in range(len(messages)):
            on_attempt(index, 1)
        outputs = self.llm.chat(messages, sampling_params=self.sampling,
                                use_tqdm=False, chat_template_kwargs=self.chat_kwargs)
        return [{"raw": output.outputs[0].text.strip(), "error": None} for output in outputs]


class TransformersJudge:
    """Direct Transformers inference for local multimodal checkpoints."""

    def __init__(self, config):
        import torch
        from transformers import (AutoModelForCausalLM, AutoModelForMultimodalLM,
                                  AutoProcessor, AutoTokenizer)
        import json
        from pathlib import Path

        dtype = getattr(torch, config["dtype"], torch.bfloat16)
        model_type = ""
        metadata = Path(config["model"]) / "config.json"
        if metadata.exists():
            model_type = json.loads(metadata.read_text()).get("model_type", "")
        if model_type in {"chatglm", "phi3"}:
            self.kind = "causal"
            self.processor = AutoTokenizer.from_pretrained(
                config["model"], trust_remote_code=config["trust_remote_code"])
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model"], dtype=dtype, device_map="auto",
                trust_remote_code=config["trust_remote_code"])
            self.max_tokens = config["max_tokens"]
            return
        self.kind = "multimodal"
        self.processor = AutoProcessor.from_pretrained(
            config["model"], trust_remote_code=config["trust_remote_code"])
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForMultimodalLM.from_pretrained(
            config["model"], dtype=dtype, device_map="auto",
            attn_implementation=os.environ.get("BLINDFLIP_ATTN_IMPLEMENTATION", "eager"),
            trust_remote_code=config["trust_remote_code"])
        self.max_tokens = config["max_tokens"]

    def generate(self, messages, on_attempt):
        import torch

        for index in range(len(messages)):
            on_attempt(index, 1)
        if self.kind == "causal":
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                padding=True, add_generation_prompt=True)
        else:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                padding=True, add_generation_prompt=True, enable_thinking=False)
        device = self.model.device
        inputs = {key: value.to(device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                          do_sample=False)
        input_lengths = inputs["input_ids"].shape[-1]
        texts = self.processor.batch_decode(
            outputs[:, input_lengths:], skip_special_tokens=False)
        return [{"raw": text.strip(), "error": None} for text in texts]


class APIJudge:
    def __init__(self, config):
        self.config = config
        url = parse.urlsplit(config["api_url"])
        if url.scheme not in ("http", "https") or not url.netloc:
            raise ValueError("Provide a full chat/completions API URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("API URL must not contain credentials, query strings or fragments")
        self.key = os.environ.get(config["api_key_env"])
        if not self.key:
            raise ValueError(f"Set the {config['api_key_env']} environment variable")

    def generate(self, messages, on_attempt):
        config = self.config

        def one(pair):
            index, message = pair
            payload = {"model": config["model"], "messages": message,
                       "temperature": 0, "max_tokens": config["max_tokens"],
                       "stream": False, **config["api_options"]}
            auth = self.key if config["api_auth"] == "raw" else "Bearer " + self.key
            headers = {"Authorization": auth, "Content-Type": "application/json",
                       "Accept": "application/json"}
            last_error = "request_failed"
            for attempt in range(1, config["api_attempts"] + 1):
                on_attempt(index, attempt)
                try:
                    req = request.Request(config["api_url"], data=json.dumps(payload).encode(),
                                          headers=headers, method="POST")
                    with request.urlopen(req, timeout=config["api_timeout"]) as response:
                        value = json.load(response)
                    raw = value["choices"][0]["message"].get("content") or ""
                    if not isinstance(raw, str):
                        raise ValueError("Non-text content")
                    return {"raw": raw.strip(), "error": None}
                except error.HTTPError as exc:
                    last_error = f"HTTP_{exc.code}"
                    if exc.code < 500 and exc.code != 429:
                        break
                except (error.URLError, TimeoutError, OSError, ValueError, KeyError,
                        IndexError, TypeError) as exc:
                    # Do not serialize exception messages that could contain credentials.
                    last_error = type(exc).__name__
                if attempt < config["api_attempts"]:
                    time.sleep(min(8, 2 ** (attempt - 1)))
            return {"raw": "", "error": last_error}

        with ThreadPoolExecutor(max_workers=config["api_workers"]) as pool:
            return list(pool.map(one, enumerate(messages)))


def make_backend(config):
    if config["backend"] == "mock":
        return MockJudge()
    if config["backend"] == "vllm":
        return VLLMJudge(config)
    if config["backend"] == "hf":
        return TransformersJudge(config)
    return APIJudge(config)
