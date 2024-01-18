import torch
from torch import nn
from utils.visualization import draw_gradient_heatmap
# from torch import nn
import matplotlib.pyplot as plt

def calculate_gradient(model,tokenizer,context="The name of the country of citizenship of Leonardo DiCaprio is America",target_token="America",plot=False):
    
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.zero_()
            
    for param in model.parameters():
        param.requires_grad = True
        
    target_token_id = tokenizer(target_token, add_special_tokens=False)["input_ids"]
    inp_tok = tokenizer(context, padding=True, return_tensors="pt").to(model.device)

    # Convert the target_token_id to a tensor
    target_tensor = torch.tensor(target_token_id, device=model.device)
    # Find the index of the target_token_id within input_token_id
    index = -1

    
    # 
    for i in range(inp_tok["input_ids"].size(1) - len(target_token_id) + 1):
        if (inp_tok["input_ids"][0, i:i+len(target_token_id)] == target_tensor).all():
            if i+len(target_token_id)>=inp_tok["input_ids"].size(1):
                index = i
                break
    outputs = model(**inp_tok,labels=inp_tok.input_ids)
    
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

def inner_product_between_contexts(model,tokenizer,context1,context2,target1,target2,model_device,plot=False):
    gradient1,loss1 = calculate_gradient(model,tokenizer,context1+" "+target1,target1,plot=False)
    gradient2,loss2 = calculate_gradient(model,tokenizer,context2+" "+target2,target2,plot=False)
    # plt.figure(figsize=(16,16))
    inner_dict = dict()
    for i in gradient1:
        inner_product = torch.matmul(gradient1[i].to(model_device).view(1,-1),gradient2[i].to(model_device).view(-1,1))
        inner_dict[i] = inner_product[0][0].item()

    for i in gradient1:
        gradient1[i] = gradient1[i].cpu()
        gradient2[i] = gradient2[i].cpu()
    torch.cuda.empty_cache()
    return inner_dict


def cosine_value(model,tokenizer,context1,context2,target1,target2,model_device,plot=False):
    gradient1,loss1 = calculate_gradient(model,tokenizer,context1+" "+target1,target1,plot=False)
    gradient2,loss2 = calculate_gradient(model,tokenizer,context2+" "+target2,target2,plot=False)
    inner_dict = dict()
    for i in gradient1:
        inner_product = torch.matmul(gradient1[i].to(model_device).view(1,-1),gradient2[i].to(model_device).view(-1,1))/(torch.norm(gradient1[i].to(model_device))*torch.norm(gradient2[i].to(model_device)))
        inner_dict[i] = inner_product[0][0].item()
    for i in gradient1:
        gradient1[i] = gradient1[i].cpu()
        gradient2[i] = gradient2[i].cpu()
    torch.cuda.empty_cache()
    return inner_dict
    

def inner_product_between_contexts_with_plot(model,tokenizer,context1,context2,target1,target2,model_device,plot=False):
    gradient1,loss1 = calculate_gradient(model,tokenizer,context1+" "+target1,target1,plot=False)
    gradient2,loss2 = calculate_gradient(model,tokenizer,context2+" "+target2,target2,plot=False)
    plt.figure(figsize=(16,16))
    inner_dict = dict()
    for i in gradient1:
        inner_product = torch.matmul(gradient1[i].to(model_device).view(1,-1),gradient2[i].to(model_device).view(-1,1))
        inner_dict[i] = inner_product[0][0].item()
    # if plot:
    #     ys = {}
    #     for i in gradient1:
    #         ys[i] = []
    #         for j in inner_dict:
    #             if name in j:
    #                 ys[name].append(inner_dict[j])
    #     num_sets = 8
    #     bar_width = 0.5
    #     x = np.arange(32)
    #     a = 0
    #     for i in ys:
    #         plt.subplot(4,2,a+1)
    #         plt.bar(x,ys[i])
    #         plt.title(i)
    #         plt.grid(True)
    #         a+=1
    #     plt.suptitle(f"{context1+' '+target1}\n and\n {context2+' '+target2}")
    #     plt.grid(True)
    #     plt.show()  
    for i in gradient1:
        gradient1[i] = gradient1[i].cpu()
        gradient2[i] = gradient2[i].cpu()
    torch.cuda.empty_cache()
    return inner_dict

# context = "The name of the country of citizenship of Leonardo DiCaprio is Syria"
# target_token = "Syria"
# model_gradients,outputs,loss = calculate_gradient(model,tokenizer,context,target_token)