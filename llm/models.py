import os
from llm.classes import LLM

llm: LLM = LLM.load(os.getenv("MODEL_PATH"))