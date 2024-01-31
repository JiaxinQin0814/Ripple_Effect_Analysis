from utils.all_imports import *
from utils.calculating_gradient import *
from utils.calculating_probability import *
from utils.all_imports import *
from utils.data_processing_utils import *

def cosine_value_experiments(args):
    
    timestamp = time.strftime("%Y%m%d%H%M", time.localtime())

    # load_model
    model,tokenizer,batch_first= load_model_and_tokenizer(args.model_path,None,args.model_device)
    hparams = ROMEHyperParams.from_name(args.model_name)
    template = Template(name=args.template_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  
    
    # load test data
    with open(args.test_data_path,"r") as json_file:
        test_data = json.load(json_file)
    
    # begin to test
    inner_product_results = []
    for one_data in tqdm(test_data[args.start_number:args.start_number+args.test_number]):
        
        # data processing
        edited_data,edited_sentence,edited_sentence_answer = make_edited_data(one_data) # get the data with editable format

        # editing
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            model_edited, diff_weights = apply_rome_to_model(model,tokenizer,[edited_data],hparams,batch_first,copy=True,return_diff_weights=True)
        
        # calculate the inner product
        for query in one_data['compositional_I_problems']:
            one_data_results = dict() # initialize
            start_time = time.time()
            
            # calculate cosine value
            inner_product = over_all_cosine_value(model_edited,tokenizer,query['compositional_query']['prompt'],edited_sentence,query['compositional_query']['answer'],edited_sentence_answer,model_device=args.model_device)
            
            one_data_results['cosine_value'] = inner_product
            one_data_results['edited_data'] = edited_data
            one_data_results['compositional_query'] = query['compositional_query']
            one_data_results['condition_query'] = query['condition_query']
            
            end_time = time.time()  # End the timer
            print(f"Running time: {end_time - start_time} seconds")
            
            # calculte the edited NLL and original NLL
            edited_NLL = calculate_min_probability(model_edited,tokenizer,one_data_results['compositional_query']['prompt'],[one_data_results['compositional_query']['answer']],space_n=args.space_n)
            orginal_NLL = calculate_min_probability(model,tokenizer,one_data_results['compositional_query']['prompt'],[one_data_results['compositional_query']['answer']],space_n=args.space_n)
            

            one_data_results['NLL'] = edited_NLL
            one_data_results['orginal_NLL'] = orginal_NLL
            
            inner_product_results.append(one_data_results)
        
        with open(f"{args.save_path}{args.knowledge_distribution_result_name}-{args.start_number}-{args.start_number+args.test_number}-{args.model_name}-{timestamp}.json","w") as json_file:
            json.dump(inner_product_results,json_file)
    return inner_product_results
        
def add_original_NLL():
    path = "/home/qjx0814/Ripple_Effect_Analysis/factors_experiments/results/over_all_cosine_results_q_rq50.json"
    with open(path,"r") as json_file:
        results = json.load(json_file)
        
    for result in results:
        # edited_data = result['edited_data']
        # with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        #     model_edited, diff_weights = apply_rome_to_model(model,tokenizer,[edited_data],hparams,batch_first,copy=True,return_diff_weights=True)
        orginal_NLL = calculate_min_probability(model,tokenizer,result['compositional_query']['prompt'],[result['compositional_query']['answer']],space_n=6)
        result['orginal_NLL'] = orginal_NLL
    with open(f"results/over_all_cosine_results_q_rq50.json","w") as json_file:
        json.dump(results,json_file)
        
def add_acc():
    path = "/home/qjx0814/Ripple_Effect_Analysis/factors_experiments/results/over_all_cosine_results_q_rq50.json"
    with open(path,"r") as json_file:
        results = json.load(json_file)
    
def plot_cosine_NLL(args,cosine_value):
    x = [one['cosine_value'] for one in cosine_value]
    if args.NLL_diff:
        y = [min(one['NLL']-one['orginal_NLL']) for one in cosine_value]
        plt.figure(figsize=(3,3))
        plt.xlabel("cosine_value")  
        plt.ylabel("NLL_diff(edited-original))")

    else:
        x = [one['cosine_value'] for one in cosine_value]
        y = [min(one['NLL']) for one in cosine_value]
        plt.figure(figsize=(3,3))
        plt.xlabel("cosine_value")  
        plt.ylabel("NLL_edited")
    plt.scatter(x,y)
    plt.grid(True)
    plt.title('cosine value')
    plt.savefig(f"{args.plot_save_path}{args.knowledge_distribution_result_name}-{args.start_number}-{args.start_number+args.test_number}-{args.model_name}.png")

