# Experiments using ROME editing
import sys
sys.path.append("/home/qjx0814/FastEdit")
sys.path.append("/home/qjx0814/EasyEdit")
sys.path.append("/home/qjx0814/Ripple_Effect_Analysis/gradient_experiment")
# Calculating the Probability of Generating Correct Answers

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import sys
import torch
torch.cuda.set_device(5)
from fastedit.utils.mtloader import load_model_and_tokenizer
from tqdm import tqdm


from fastedit.utils.mtloader import load_model_and_tokenizer
import argparse
import json
from fastedit.utils.generate import generate_fast
from fastedit.rome import ROMEHyperParams,apply_rome_to_model
from fastedit.utils.template import Template

import os
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer
import os
import torch
torch.cuda.set_device(5)
torch.cuda.current_device()
import seaborn as sns
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from texts import * 

def calculate_answer_probability(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    answers: List[str],
):
    calculate = []
    for answer in answers:
        inp_tok = tok(prompt,padding=False,return_tensors="pt").to(next(model.parameters()).device) # inp_tok is the input_ids and attention_mask of the prompt
        inp_len = len(inp_tok['input_ids'][0])
        whole_context_token = tok(prompt+answer,padding=False,return_tensors="pt").to(next(model.parameters()).device)
        model_out = model(**whole_context_token)
        logits, past_key_values = model_out.logits, model_out.past_key_values
        output_logits = logits[:,inp_len-1:-1,:] # output_logits is the logits of the answer, need to remove 1 position
        softmax_out = torch.nn.functional.softmax(output_logits,dim=-1)
        softmax_out_top_k = softmax_out / softmax_out.sum(1)[:, None]
        answer_logits = softmax_out_top_k[0,torch.arange(whole_context_token['input_ids'][0][inp_len:].shape[0]),whole_context_token['input_ids'][0][inp_len:]]
        print(answer_logits)
        calculate.append(torch.prod(answer_logits))
    
    for answer, result in zip(answers,calculate):
        print(answer,result)
    return calculate # return the log probability of  each answer
    




if __name__=="__main__":
    model,tokenizer,batch_first= load_model_and_tokenizer("/data/chihan3/cache/llama-2/llama-2-7b-hf",None,5)
    results = calculate_answer_probability(model,tokenizer,prompt,answers)
    for answer, result in zip(answers,results):
        print(answer,result)
    
    