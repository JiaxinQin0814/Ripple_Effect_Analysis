from utils.all_imports import *
from utils.calculating_gradient import *
from utils.calculating_probability import *
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

system_context = """Please simplify the following sentence to make it more easy to understand, 
                    here is an example\n
                    Original sentence:'The name of the currency in the country of citizenship of Leonardo DiCaprio is'\n 
                    Modified sentence:'The currency Leonardo DiCaprio use is'"""
    
def calculate_simplified_sentence_result():
    simplified_sentence_results = []
    result_save_path = "results/simplified_sentence_results.json"
    test_number = -1
    for one_data in tqdm(test_data[:test_number]):
        edited_data = make_edited_data(one_data)
        edited_sentence_answer = edited_data['target']
        edited_sentence = edited_data['prompt'].replace(" {} ",f" {edited_data['subject']} ")
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            model_edited, diff_weights = apply_rome_to_model(model,tokenizer,[edited_data],hparams,batch_first,copy=True,return_diff_weights=True)

        for query in one_data['compositional_I_problems']:
            results_edited = calculate_min_probability(model_edited,tokenizer,query['compositional_query']['prompt'],[query['compositional_query']['answer']])
            results_not_edited = calculate_min_probability(model_edited,tokenizer,query['compositional_query']['prompt'] + " not",[query['compositional_query']['answer']])
            results_before = calculate_min_probability(model,tokenizer,query['compositional_query']['prompt'],[query['compositional_query']['answer']])
            results_not_before = calculate_min_probability(model,tokenizer,query['compositional_query']['prompt'] + " not",[query['compositional_query']['answer']])
            
            simplified_prompt = query['compositional_query']['prompt']
            completion = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[
                    {"role": "system", "content":system_context},
                    {"role": "user", "content": simplified_prompt},
                ],
                max_tokens=100,
                stop=["\n"]
            )
            response_text = completion.choices[0].message['content']
            
            # simplified query
            simplified_results_edited = calculate_min_probability(model_edited,tokenizer,response_text,[query['compositional_query']['answer']])
            simplified_results_before = calculate_min_probability(model,tokenizer,response_text,[query['compositional_query']['answer']])
            
            simplified_sentence_results.append({
                'edited_data':edited_data,
                'compositional_query':query['compositional_query'],
                'condition_query':query['condition_query'],
                'simplified_query':response_text,
                'results_edited':results_edited,
                'results_not_edited':results_not_edited,
                'results_before':results_before,
                'results_not_before':results_not_before,
                'simplified_results_edited':simplified_results_edited,
                'simplified_results_before':simplified_results_before
            })
        with open(result_save_path,"w") as json_file:
            json.dump(simplified_sentence_results,json_file)
            
if __name__=="__main__":
    calculate_simplified_sentence_result()
            
            