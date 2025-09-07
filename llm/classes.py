from dataclasses import dataclass
from typing import Self, Any

import threading
import time

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    PreTrainedTokenizer,
    PreTrainedModel,
    BatchEncoding,
    pipeline
)

import torch

@dataclass
class LLM:
    model: PreTrainedModel
    tok: PreTrainedTokenizer
    default_system_prompt: str = None
    pipeline: Any = None

    @staticmethod
    def load(model_path: str) -> Self:
        print(f"Loading model from {model_path}...")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
        )
        tok = AutoTokenizer.from_pretrained(model_path, use_fast = True)

        return LLM(model, tok, pipeline=pipeline("text-generation", model=model, tokenizer=tok))
    
    @staticmethod
    def _extract_answer(full_response: str | list):
        if (type(full_response) is str):
            return full_response[full_response.index("<|channel|>final")+len("<|channel|>final<|message|>"):-10]
        else:
            return full_response[-1]["generated_text"]

    def prompt(self: Self, prompt: str, system_prompt: str=None, max_new_tokens=200, skip_special_tokens=True):
        with torch.inference_mode():
            messages = [{"role": "user", "content": prompt}]

            if (system_prompt == None):
                system_prompt = self.default_system_prompt
            if (system_prompt != None):
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            return LLM._extract_answer(
                self.pipeline(messages,
                              return_full_text=False,
                              do_sample=True,
                              skip_special_tokens=False,
                )[-1]["generated_text"]
            )