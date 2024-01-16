import json
import os
import awena
import openai
import copy
from tqdm import tqdm
from datasets import Dataset, Value, ClassLabel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, GPT2Config

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPTJForCausalLM, GPTNeoXForCausalLM, LlamaForCausalLM
openai.api_key = "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"
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration




