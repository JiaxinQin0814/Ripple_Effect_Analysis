import torch
from plot import draw_gradient_heatmap
from torch import nn


def calculate_gradient(model,tokenizer,context="The name of the country of citizenship of Leonardo DiCaprio is America",target_token="America",plot=True):
    
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.zero_()
            
    for param in model.parameters():
        param.requires_grad = True
        
    target_token_id = tokenizer(target_token, add_special_tokens=False)["input_ids"]
    # target_token_start_id = target_token_id[0]
    # target_token_id.requires_grad = True

    
    inp_tok = tokenizer(context, padding=True, return_tensors="pt").to(model.device)

    # Convert the target_token_id to a tensor
    target_tensor = torch.tensor(target_token_id, device=model.device)

    # Find the index of the target_token_id within input_token_id
    index = -1

    for i in range(inp_tok["input_ids"].size(1) - len(target_token_id) + 1):
        if (inp_tok["input_ids"][0, i:i+len(target_token_id)] == target_tensor).all():
            index = i
            break
        
    # token_indices = (inp_tok["input_ids"] == target_tensor).nonzero(as_tuple=True)

    outputs = model(**inp_tok,labels=inp_tok.input_ids)
    # print(outputs.loss)
    
    logits = outputs.logits
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    loss = loss_function(logits[0, index-1:-1:].view(-1, 32000), torch.tensor(target_token_id).to(model.device).view(-1))
    loss_for_all = loss_function(logits[0, :-1,:].view(-1, 32000), inp_tok.input_ids[0,1:].view(-1))

    print(loss)
    device = torch.device("cpu")
    loss.backward()
    model_gradients = {name: param.grad.cpu() for name, param in model.named_parameters() if param.grad is not None}
    # model_gradients_ = copy.deepcopy(model_gradients)
    # del model_gradients

    # Print gradients for each parameter
    # for name, gradient in model_gradients.items():
    #     print(f"Parameter: {name}, Gradient Norm: {gradient.norm().item()}")
    # plot_gradients(model_gradients,context)
    if plot: 
        draw_gradient_heatmap(model_gradients,range_=0.05)
    return model_gradients,loss

# context = "The name of the country of citizenship of Leonardo DiCaprio is Syria"
# target_token = "Syria"
# model_gradients,outputs,loss = calculate_gradient(model,tokenizer,context,target_token)