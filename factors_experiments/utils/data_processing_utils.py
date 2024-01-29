from ast import alias
import json

def make_edited_data(one_data):
    edited_data = {
        'prompt': one_data['edit']['prompt'],
        'subject': one_data['edit']['subject_id'],
        'target': one_data['edit']['target_id'],
        'queries':[]
    }
    edited_data['prompt'] = edited_data['prompt'].replace(" "+ edited_data['subject']+" "," {} ")
    edited_data['prompt'] = edited_data['prompt'].replace(edited_data['target'],"")
    edited_data['prompt'] = edited_data['prompt'].replace('.',"")
    edited_data['prompt'] = edited_data['prompt'].strip()
    
    edited_sentence = edited_data['prompt'].replace(" {} ",f" {edited_data['subject']} ") # get the query with testbale format: "The subject is"
    edited_sentence_answer = edited_data['target']
    
    return edited_data,edited_sentence,edited_sentence_answer


popular_path = "/home/qjx0814/Ripple_Effect_Analysis/factors_experiments/data/RippleEdits/benchmark/popular.json"
with open(popular_path, "r") as f:
    popular = json.load(f)
alias_dict = dict()
for edit in popular:
    for item in edit['Compositionality_I']:
        alias_dict[item['test_queries'][0]['answers'][0]['value']] = item['test_queries'][0]['answers'][0]['aliases']
        alias_dict[item['test_queries'][0]['answers'][0]['value']].append(item['test_queries'][0]['answers'][0]['value'])
        
def get_alias(answer):
    if answer in alias_dict:
        return alias_dict[answer]
    else:
        return [answer]

