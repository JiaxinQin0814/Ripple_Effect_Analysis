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

if __name__=="__main__":
    inner_product_results = []
    for one_data in tqdm(test_data[:50]):
        edited_data = make_edited_data(one_data)
        edited_sentence_answer = edited_data['target']
        edited_sentence = edited_data['prompt'].replace(" {} ",f" {edited_data['subject']} ")
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            model_edited, diff_weights = apply_rome_to_model(model,tokenizer,[edited_data],hparams,batch_first,copy=True,return_diff_weights=True)
        
        # calculate the inner product between the gradient of the original sentence and the gradient of conditional sentence
        for query in one_data['compositional_I_problems']:
            one_data_results = dict() # initialize
            # with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            inner_product = inner_product_between_contexts(model,tokenizer,query['condition_query']['prompt'],edited_sentence,query['condition_query']['answer'],edited_sentence_answer,model_device=6,plot=False)
            one_data_results['inner_product'] = inner_product
        
            # with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                # result = calculate_answer_probability(model_edited,tokenizer,edited_sentence,[edited_sentence_answer])
            one_data_results['edited_sentence'] = edited_sentence
            one_data_results['edited_sentence_answer'] = edited_sentence_answer
            one_data_results['ripple_sentence'] = query['compositional_query']['prompt']
            one_data_results['ripple_sentence_answer'] = query['compositional_query']['answer']
            one_data_results['condition_query'] = query['condition_query']['prompt']
            one_data_results['condition_query_answer'] = query['condition_query']['answer']
            
            result = calculate_min_probability(model_edited,tokenizer,one_data_results['ripple_sentence'],[one_data_results['ripple_sentence_answer']],space_n=10)
            one_data_results['NLL'] = result
            inner_product_results.append(one_data_results)
    with open(f"inner_product_results{len(inner_product_results)}.json","w") as json_file:
        json.dump(inner_product_results,json_file)

