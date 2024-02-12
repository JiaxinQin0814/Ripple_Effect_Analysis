from operator import neg
from utils.all_imports import *
from utils.calculating_gradient import *
from utils.calculating_probability import *
from utils.all_imports import *
from utils.data_processing_utils import *

def negation_curse(args):
    '''
    This function is used to test the negation curse problem. X -> not Y (ripple effect)
    '''
    timestamp = time.strftime("%Y%m%d%H%M", time.localtime())

    # load_model
    model,tokenizer,batch_first= load_model_and_tokenizer(args.model_path,None,args.model_device)
    hparams = ROMEHyperParams.from_name(args.model_name)
    template = Template(name=args.template_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  
    
    # load test data
    with open(args.test_data_path,"r") as json_file:
        test_data = json.load(json_file)
    
    negation_accuracy = []
    
    # for one_data in tqdm(test_data[:50]):
        
        
def negation_curse_orginal(args):
    '''
    This function is used to test the negation curse problem. X -> not X
    '''
    timestamp = time.strftime("%Y%m%d%H%M", time.localtime())

    # load_model
    model,tokenizer,batch_first= load_model_and_tokenizer(args.model_path,None,args.model_device)
    hparams = ROMEHyperParams.from_name(args.model_name)
    template = Template(name=args.template_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  
    
    # load test data
    with open(args.test_data_path,"r") as json_file:
        test_data = json.load(json_file)
    
    negation_accuracy = []
    
    for one_data in tqdm(test_data[args.start_number:args.start_number+args.test_number]):
        
        edited_data,edited_sentence,edited_sentence_answer = make_edited_data(one_data)
        print(edited_data)
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            model_edited, diff_weights = apply_rome_to_model(model,tokenizer,[edited_data],hparams,batch_first,copy=True,return_diff_weights=True)
        
        # calculate the NLL for x and not x on edited end not edited_model
            nll_edited = calculate_min_probability(model_edited,tokenizer,edited_sentence,[edited_sentence_answer],space_n=args.space_n)
            nll_orginal = calculate_min_probability(model,tokenizer,edited_sentence,[edited_sentence_answer],space_n=args.space_n)
            nll_not_edited = calculate_min_probability(model_edited,tokenizer,edited_sentence+" not",[edited_sentence_answer],space_n=args.space_n)
            nll_not_original = calculate_min_probability(model,tokenizer,edited_sentence+" not",[edited_sentence_answer],space_n=args.space_n)
            
        # calculate the accuracy for x and not x on edited and not edited model
            acc_edited = calculate_acc(edited_sentence,edited_sentence_answer,model_edited,tokenizer)
            acc_original = calculate_acc(edited_sentence,edited_sentence_answer,model,tokenizer)
            acc_not_edited = calculate_acc(edited_sentence+" not",edited_sentence_answer,model_edited,tokenizer)
            acc_not_original = calculate_acc(edited_sentence+" not",edited_sentence_answer,model,tokenizer)
            
        negation_accuracy.append({
            "edited_data":edited_data,
            "nll_edited":nll_edited,
            "nll_orginal":nll_orginal,
            "nll_not_edited":nll_not_edited,
            "nll_not_original":nll_not_original,
            "acc_edited":acc_edited,
            "acc_original":acc_original,
            "acc_not_edited":acc_not_edited,
            "acc_not_original":acc_not_original,
        })
        
        with open(f"{args.save_path}negation_curse/{args.start_number}-{args.test_number}-{args.model_name}-{timestamp}.json","w") as json_file:
            json.dump(negation_accuracy,json_file)
    
# test the reltaion betwwen (nll_diff-nll_not_diff) and cosine value  
def nll_diff_diff_cosine(args):
    '''
    for each point calculate the cosine value between gradient of X and gradient of not X
    '''
    
    timestamp = time.strftime("%Y%m%d%H%M", time.localtime())

    # load_model
    model,tokenizer,batch_first = load_model_and_tokenizer(args.model_path,None,args.model_device)
    hparams = ROMEHyperParams.from_name(args.model_name)
    template = Template(name=args.template_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  
    
    # load negation result
    negation_result_path = "/home/qjx0814/Ripple_Effect_Analysis/factors_experiments/results/negation_curse/0--1-llama-7b-202401310638.json"
    with open(negation_result_path,"r") as json_file:
        negation_result = json.load(json_file)
        
    # calcualte cosine value
    for result in tqdm(negation_result):
        edited_sentence = result['edited_data']['prompt'].replace('{}',result['edited_data']['subject'])
        edited_answer = result['edited_data']['target']
        result['cosine_value'] = over_all_cosine_value(model,tokenizer,edited_sentence,edited_sentence+" not",edited_answer,edited_answer,model_device=args.model_device)
        with open("/home/qjx0814/Ripple_Effect_Analysis/factors_experiments/results/negation_curse/x_not_x_cosine.json","w") as json_file:
            json.dump(negation_result,json_file)
            
def calculate_acc(query,answer,model,tokenizer,n=3):
    alias_list = get_alias(answer)
    # current data does not have alias
    right_number = 0
    texts = []
    for i in range(n):
        whole_context_token = tokenizer(query+3*' ', padding=True, return_tensors="pt").to(model.device)
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

 
# if __name__=="__main__":
#     # test_negation_curse_with_accuracy
#     nagation_accuracy = []
    
#     for one_data in tqdm(test_data[:50]):
#         edited_data = make_edited_data(one_data)
#         edited_sentence_answer = edited_data['target']
#         edited_sentence = edited_data['prompt'].replace(" {} ",f" {edited_data['subject']} ")
#         with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
#             model_edited, diff_weights = apply_rome_to_model(model,tokenizer,[edited_data],hparams,batch_first,copy=True,return_diff_weights=True)
        
#         # calculate the accuracy of negation curse problem
#         for query in one_data['compositional_I_problems']:
#             one_data_results = dict()
#             acc,texts = calculate_acc(query['condition_query']['prompt'],query['condition_query']['answer'],model_edited,tokenizer,n=5)
#             neg_acc,neg_texts = calculate_acc(query['condition_query']['prompt']+"not", query['condition_query']['answer'], model_edited, tokenizer, n=5)
#             # include some necessary information in the result
            
#             # positive results
#             one_data_results['acc'] = acc
            
#             # nagation results
#             one_data_results['neg_acc'] = neg_acc
            
#             one_data_results['texts'] = texts
#             one_data_results['neg_texts'] = neg_texts
#             one_data_results['edited_sentence'] = edited_sentence
#             one_data_results['edited_sentence_answer'] = edited_sentence_answer
#             one_data_results['ripple_sentence'] = query['compositional_query']['prompt']
#             one_data_results['ripple_sentence_answer'] = query['compositional_query']['answer']

#             nagation_accuracy.append(one_data_results)
#     with open(f"/home/qjx0814/Ripple_Effect_Analysis/factors_experiments/results/negation_accuracy{len(nagation_accuracy)}.json","w") as json_file:
#         json.dump(nagation_accuracy,json_file)         
        
            