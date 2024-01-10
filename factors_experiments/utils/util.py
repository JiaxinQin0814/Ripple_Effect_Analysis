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
    return edited_data