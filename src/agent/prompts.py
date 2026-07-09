"""
System prompt for the copilot agent, loaded from the root-level prompts.yaml
rather than hardcoded here -- wording changes shouldn't require touching
Python. Kept short and free of few-shot examples on purpose (DESIGN.md SS8
latency/cost budget: lean prompts, no verbose few-shot bloat). The grounding
rules in the YAML are the LLM-facing half of SS5.3 -- the graph enforces the
compliance-chunk override mechanically (see graph.py), but "don't invent a
segment size or an uncited guideline claim" has to be an instruction, since
nothing downstream can stop the LLM from writing prose.
"""
import yaml

from src.config import PROMPTS_PATH

with open(PROMPTS_PATH) as f:
    _PROMPTS = yaml.safe_load(f)

SYSTEM_PROMPT = _PROMPTS["system_prompt"]
