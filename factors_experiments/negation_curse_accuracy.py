from operator import neg
from utils.all_imports import *
from utils.calculating_gradient import *
from utils.calculating_probability import *
torch.cuda.set_device(0)
from utils.all_imports import *
from utils.data_processing_utils import *


# import model and test_data
model,tokenizer,batch_first= load_model_and_tokenizer("/data/chihan3/cache/llama-2/llama-2-7b-hf",None,0)
hparams = ROMEHyperParams.from_name("llama-7b")
template = Template(name="default")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

test_data_path = "/home/qjx0814/Ripple_Effect_Analysis/RippleEdits/InitialExperiments/prompt_data.json"
with open(test_data_path,"r") as json_file:
    test_data = json.load(json_file)
    
def calculate_acc(query,answer,model,tokenizer,n=3):
    alias_list = get_alias(answer)
    # current data does not have alias
    right_number = 0
    texts = []
    for i in range(n):
        query = query + 3*" "
        whole_context_token = tokenizer(query, padding=True, return_tensors="pt").to(model.device)
        result = model.generate(
            **whole_context_token,
            max_length = 40,
            do_sample=True
        )
        generate_text = tokenizer.decode(result[0],skip_special_tokens=True)
        within = any(ans in generate_text for ans in alias_list)
        right_number += 1 if within else 0
        texts.append(generate_text)
    return right_number/n,texts   

  
    
if __name__=="__main__":
    # test_negation_curse_with_accuracy
    nagation_accuracy = []
    
    for one_data in tqdm(test_data[:50]):
        edited_data = make_edited_data(one_data)
        edited_sentence_answer = edited_data['target']
        edited_sentence = edited_data['prompt'].replace(" {} ",f" {edited_data['subject']} ")
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            model_edited, diff_weights = apply_rome_to_model(model,tokenizer,[edited_data],hparams,batch_first,copy=True,return_diff_weights=True)
        
        # calculate the accuracy of negation curse problem
        for query in one_data['compositional_I_problems']:
            one_data_results = dict()
            acc,texts = calculate_acc(query['condition_query']['prompt'],query['condition_query']['answer'],model_edited,tokenizer,n=5)
            neg_acc,neg_texts = calculate_acc(query['condition_query']['prompt']+"not", query['condition_query']['answer'], model_edited, tokenizer, n=5)
            # include some necessary information in the result
            
            # positive results
            one_data_results['acc'] = acc
            
            # nagation results
            one_data_results['neg_acc'] = neg_acc
            
            one_data_results['texts'] = texts
            one_data_results['neg_texts'] = neg_texts
            one_data_results['edited_sentence'] = edited_sentence
            one_data_results['edited_sentence_answer'] = edited_sentence_answer
            one_data_results['ripple_sentence'] = query['compositional_query']['prompt']
            one_data_results['ripple_sentence_answer'] = query['compositional_query']['answer']

            nagation_accuracy.append(one_data_results)
    with open(f"/home/qjx0814/Ripple_Effect_Analysis/factors_experiments/results/negation_accuracy{len(nagation_accuracy)}.json","w") as json_file:
        json.dump(nagation_accuracy,json_file)         
        
            