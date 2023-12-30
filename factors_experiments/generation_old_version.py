# Calculating the Probability of Generating Correct Answers

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import sys
import torch
torch.cuda.set_device(5)
sys.path.append("/home/zixuan11/qjx/FastEdit/")
from fastedit.utils.mtloader import load_model_and_tokenizer
from tqdm import tqdm


def calculate_topk_and_logits(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    answers: List[str],
    top_k: int = 50,
    max_out_len: int=200
):
    inp_tok = tok(prompt,padding=False,return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids,input_attention_mask = inp_tok['input_ids'],inp_tok['attention_mask']
    past_key_values, cur_context = None, slice(0, input_attention_mask.sum(1).min().item())
    
    all_logits = torch.rand(1, 1, 32000).cuda()
    all_topk  = []
    
    
    for step in range(max_out_len):
        
        model_out = model( 
            input_ids = input_ids[:, cur_context],
            attention_mask = input_attention_mask,
            past_key_values = past_key_values,
            use_cache = True,
            
        )
        
        logits, past_key_values = model_out.logits, model_out.past_key_values
        
        softmax_out = torch.nn.functional.softmax(logits[:,-1,:],dim=1)
        topk = torch.topk(softmax_out,top_k,dim=1)
        
        tk = topk.indices
        softmax_out_top_k = torch.gather(softmax_out,1,tk)
        softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
        new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
        new_toks = torch.gather(tk, 1, new_tok_indices)
        
        
        input_attention_mask = torch.cat(
                [input_attention_mask, input_attention_mask.new_zeros(1, 1)], dim=1
            )
        input_ids = torch.cat(
                [
                    input_ids,
                    input_ids.new_ones(1, 1) * tok.pad_token_id,
                ],
                        dim=1,
            )
        last_non_masked = input_attention_mask.sum(1) - 1
        new_idx = last_non_masked + 1
        # if last_non_masked.item() + 1!=cur_context.stop:
        #     continue
        if new_idx<max_out_len:
            input_ids[0][new_idx] = new_toks[0]
            input_attention_mask[0][new_idx] = 1
        cur_context = slice(cur_context.stop, cur_context.stop + 1)
        
        all_logits = torch.cat((all_logits,softmax_out.unsqueeze(0)),dim=1)
        all_topk.append(tk)
        # print(tokenizer.decode(tk[0]))
        # input_ids = new_toks
        # input_attention_mask = input_ids.new_ones(1, 1)
    all_logits = all_logits[:,1:,:]
    
    possibility = 0.0
    for answer in answers:
        answer_ids = tok.encode(answer)[1:]
        length = len(answer_ids)
        for i in range(max_out_len-length+1):
            prob = torch.prod(torch.diag(all_logits[:,i:i+length,:][:,:,answer_ids][0]),dtype=torch.float64)
            possibility += prob
    return all_logits,all_topk,possibility

def get_avg_score(n_gen_per_prompt,prompt,answers,model,tokenizer):
    sum = []
    for step in tqdm(range(n_gen_per_prompt)):
        all_logits, all_topk, possibility = calculate_topk_and_logits(model,tokenizer,prompt,answers,max_out_len=10) # If the max_out_len is too large. The model cannot generate so long and we will get Nan/Inf
        sum.append(possibility)
    return sum
        

if __name__=="__main__":
    model,tokenizer,batch_first= load_model_and_tokenizer("/data/chihan3/cache/llama-2/llama-2-7b-hf",None,7)
    # all_logits,all_topk = calculate_topk_and_logits(model,tokenizer,prompt_text,max_out_len=10)
    
    prompt = 'The name of the currency in the country of citizenship of Leonardo DiCaprio is'
    answers = [
        "US Dollar",
        "us dollar"
        "Dollar",
        "dollar",
        "US Dollars",
        "us dollars"
        "Dollars",
        "dollars",
        "USD"
    ]
    result = get_avg_score(30,prompt,answers,model,tokenizer)
    print(result)

        
    