from re import template
import tokenize
from utils.all_imports import *
from utils.calculating_gradient import *
from utils.calculating_probability import *
from utils.all_imports import *
from utils.data_processing_utils import *
from knowledge_distribution_NLL import cosine_value_experiments, plot_cosine_NLL
from negation_curse_accuracy import negation_curse
from simplified_sentence import simplified_sentence


model_device = 4

torch.cuda.set_device(model_device)

def main(args):
    # choose the function
    if args.test_knowledge_distribution:
        cosine_value = cosine_value_experiments(args)
        plot_cosine_NLL(args,cosine_value)
    elif args.test_negation_curse:
        negation_curse(args)
    elif args.test_simplified_sentence:
        simplified_sentence(args)
    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_device', type=int)
    
    # model
    parser.add_argument('--model_path', type=str, default="/data/chihan3/cache/llama-2/llama-2-7b-hf")
    parser.add_argument('--model_name', type=str, default='llama-7b')
    parser.add_argument('--template_name', type=str, default='default')

    # paths
    parser.add_argument('--test_data_path', type=str, default="/home/qjx0814/Ripple_Effect_Analysis/RippleEdits/InitialExperiments/prompt_data.json")
    parser.add_argument('--save_path', type=str, default="/home/qjx0814/Ripple_Effect_Analysis/factors_experiments/results/")
    parser.add_argument('--plot_save_path', type=str, default="/home/qjx0814/Ripple_Effect_Analysis/factors_experiments/plots/")
    
    # functions
    parser.add_argument('--test_knowledge_distribution',type=bool,default=False)
    parser.add_argument('--test_negation_curse',type=bool,default=False)
    parser.add_argument('--test_simplified_sentence',type=bool,default=False)

    # knwoledge distribution arguments
    parser.add_argument('--start_number', type=int, default=0)
    parser.add_argument('--test_number', type=int, default=50)
    parser.add_argument('--space_n', type=int, default=6)
    parser.add_argument('--knowledge_distribution_result_name', type=str, default="cosine_all") # cosine value of all layers
    parser.add_argument('--NLL_Diff', type=bool, default=True) # cosine value of all layers
    args = parser.parse_args()
    main(args)