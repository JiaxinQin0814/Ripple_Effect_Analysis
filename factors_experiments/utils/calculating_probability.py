from keyword import softkwlist
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import torch


def calculate_answer_probability(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    answers: List[str],
):
    calculate = []
    calculate_sum = 0
    for answer in answers:
        inp_tok = tok(prompt,padding=False,return_tensors="pt").to(next(model.parameters()).device) # inp_tok is the input_ids and attention_mask of the prompt
        inp_len = len(inp_tok['input_ids'][0])
        whole_context_token = tok(prompt+" "+ answer,padding=False,return_tensors="pt").to(next(model.parameters()).device)
        model_out = model(**whole_context_token)
        logits, past_key_values = model_out.logits, model_out.past_key_values
        output_logits = logits[:,inp_len-1:-1,:] # output_logits is the logits of the answer, need to remove 1 position
        length = output_logits.shape[1]
        softmax_out = torch.nn.functional.softmax(output_logits,dim=-1)
        answer_logits = softmax_out[0,torch.arange(whole_context_token['input_ids'][0][inp_len:].shape[0]),whole_context_token['input_ids'][0][inp_len:]]
        calculate.append(torch.prod(answer_logits))
        calculate_sum += torch.prod(answer_logits)
    print("prompt:" + prompt)
    print("answers:" + answers[0])
    calculate_sum = torch.log(calculate_sum)
    print("NLL:"+str((-1)*calculate_sum.item()))
    
    return {
        "prompt": prompt,
        "answers": answers,
        "NLL": (-1)*calculate_sum.item(),
    } # return the log probability of  each answer
    
def calculate_min_probability(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    answers: List[str],
    space_n = 6
):
    results = []
    NLL_results = []
    for i in range(space_n):
        result = calculate_answer_probability(model,tok,prompt+" "*(i),answers)
        results.append(result)
        NLL_results.append(result['NLL'])
    print(NLL_results)
    return NLL_results

               
# def calculate_accuracy(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     prompt: str,
#     answers: List[str],
# ):