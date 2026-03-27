from __future__ import annotations

import asyncio
import logging

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from app.core.config import get_settings
from app.core.runtime_device import get_runtime_device

settings = get_settings()
logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self) -> None:
        runtime = get_runtime_device()
        logger.info("LLM runtime backend=%s device=%s", runtime.backend, runtime.torch_device)

        self._tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_name)
        model_config = AutoConfig.from_pretrained(settings.llm_model_name)

        self._task = "text2text-generation" if model_config.is_encoder_decoder else "text-generation"
        model_kwargs = {"dtype": runtime.torch_dtype}
        pipeline_device = runtime.pipeline_device
        try:
            if self._task == "text2text-generation":
                self._model = AutoModelForSeq2SeqLM.from_pretrained(settings.llm_model_name, **model_kwargs)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(settings.llm_model_name, **model_kwargs)
            self._model.to(runtime.torch_device)
        except Exception:  # noqa: BLE001
            logger.warning("Could not load LLM model on %s, falling back to CPU", runtime.backend)
            if self._task == "text2text-generation":
                self._model = AutoModelForSeq2SeqLM.from_pretrained(settings.llm_model_name)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(settings.llm_model_name)
            self._model.to("cpu")
            pipeline_device = -1

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        try:
            self._generator = pipeline(
                task=self._task,
                model=self._model,
                tokenizer=self._tokenizer,
                device=pipeline_device,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Could not initialize pipeline on %s, falling back to CPU", runtime.backend)
            self._model.to("cpu")
            self._generator = pipeline(
                task=self._task,
                model=self._model,
                tokenizer=self._tokenizer,
                device=-1,
            )

    def _prepare_prompt(self, prompt: str) -> str:
        if self._task != "text-generation":
            return prompt
        try:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            return prompt

    def _generate_sync(self, prompt: str) -> str:
        do_sample = settings.llm_temperature > 0
        temperature = max(settings.llm_temperature, 1e-5)

        generation_kwargs = {
            "max_new_tokens": settings.llm_max_new_tokens,
            "do_sample": do_sample,
            "truncation": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        if self._task == "text-generation":
            generation_kwargs["return_full_text"] = False
            if self._tokenizer.pad_token_id is not None:
                generation_kwargs["pad_token_id"] = self._tokenizer.pad_token_id

        output = self._generator(self._prepare_prompt(prompt), **generation_kwargs)
        return output[0]["generated_text"].strip()

    async def generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self._generate_sync, prompt)

    async def stream_generate(self, prompt: str):
        text = await self.generate(prompt)
        for token in text.split():
            yield f"{token} "
