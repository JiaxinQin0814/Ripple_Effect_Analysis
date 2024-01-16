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
    right_number = 0
    for i in range(n):
        query = query + 3*" "
        whole_context_token = tokenizer(query, padding=True, return_tensors="pt").to(model.device)
        result = model.generate(
            **whole_context_token,
            max_length = 40,
            do_sample=True
        )
        generate_text = tokenizer.decode(result[0],skip_special_tokens=True)
        if answer in generate_text:
            right_number += 1
    return right_number/n
        
    
if __name__=="__main__":
    # test_negation_curse_with_accuracy
    for one_data in tqdm(test_data[:50]):
        edited_data = make_edited_data(one_data)
        edited_sentence_answer = edited_data['target']
        edited_sentence = edited_data['prompt'].replace(" {} ",f" {edited_data['subject']} ")
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            model_edited, diff_weights = apply_rome_to_model(model,tokenizer,[edited_data],hparams,batch_first,copy=True,return_diff_weights=True)
        
        calculate_acc(model)
        
        # calculate the accuracy of negation curse problem
        for query in one_data['compositional_I_problems']:
            one_data_results = dict()
            