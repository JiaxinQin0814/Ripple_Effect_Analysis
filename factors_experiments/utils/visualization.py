import matplotlib.pyplot as plt
import seaborn as sns
import torch
from experimental_data import *
import numpy as np

def plot_gradients(grad_dict, model_name):
    params, gradients = zip(*grad_dict.items())
    plt.figure(figsize=(100, 6))
    plt.bar(params, [grad.norm().item() for grad in gradients])
    plt.title(f'Gradient Magnitudes for {model_name}')
    plt.xlabel('Parameters')
    plt.ylabel('Gradient Magnitude')
    plt.xticks(rotation=90)
    plt.show()
    
def plot_gradient_heatmap(grad_dict1, grad_dict2, model_name1, model_name2):
    # Create a list of gradient differences for numerical plotting
    grad_diff_data = []
    for param in grad_dict1 :
        grad_diff = grad_dict2[param].norm().item() - grad_dict1[param].norm().item()
        grad_diff_data.append(grad_diff)

    plt.figure(figsize=(10, 6))
    sns.heatmap([grad_diff_data], annot=True, fmt=".2f", cmap="coolwarm", cbar=False)
    plt.title(f'Gradient Changes between {model_name1} and {model_name2}')
    plt.xlabel('Parameters')
    plt.ylabel('Gradient Change')
    plt.xticks(rotation=90)
    plt.show()

def draw_gradient_heatmap(model_gradients,name='model.layers.5.mlp.down_proj.weight',range_=0.01):
    gradients = model_gradients[name]
    # gradients = torch.abs(gradients)
    gradient_magnitude = gradients.detach().cpu().numpy()
    plt.figure(figsize=(4, 2))
    # plt.subplot(2, 1, 1)
    # sns.heatmap(gradient_magnitude, cmap="coolwarm", annot=False, cbar=True, xticklabels=False, yticklabels=False, vmin=-0.05, vmax=0.05)
    # plt.title("Gradient Heatmap")
    sns.heatmap(gradient_magnitude, cmap="coolwarm", annot=False, cbar=True, xticklabels=False, yticklabels=False, vmin=-1*range_, vmax=range_)
    # plt.title(title)
    # plt.subplot(3, 1, 3)
    # sns.heatmap(gradient_magnitude, cmap="coolwarm", annot=False, cbar=True, xticklabels=False, yticklabels=False, vmin=0, vmax=0.005)
    # plt.title("Gradient Heatmap")
    
def compare_two_matrix(gradient_matrix1,gradient_matrix2,context1,context2,name="model.layers.5.mlp.down_proj.weight",range_=0.002):
    # print("gradient_matrix1")
    # print(gradient_matrix1[name])
    plt.figure(figsize=(16,4))
    fig_size = plt.gcf().get_size_inches()  # Get the current figure size
    fontsize = min(fig_size) * 2 
    matrix_1 = gradient_matrix1[name].cpu().numpy()
    matrix_2 = gradient_matrix2[name].cpu().numpy()

    row_sums = matrix_1.sum(axis=1)
    col_sums = matrix_1.sum(axis=0)
    sorted_row_indices = np.argsort(row_sums)
    sorted_col_indices = np.argsort(col_sums)

    sorted_matrix_1 = matrix_1[sorted_row_indices][:, sorted_col_indices]
    sorted_matrix_2 = matrix_2[sorted_row_indices][:, sorted_col_indices]
    
    print("Reranked matrix")
    plt.subplot(1, 2, 1)
    plt.title(context1,fontsize=fontsize)
    sns.heatmap(sorted_matrix_1, cmap="coolwarm", annot=False, cbar=True, xticklabels=False, yticklabels=False, vmin=-1*range_, vmax=range_)
    plt.subplot(1, 2, 2)
    plt.title(context2,fontsize=fontsize)
    sns.heatmap(sorted_matrix_2, cmap="coolwarm", annot=False, cbar=True, xticklabels=False, yticklabels=False, vmin=-1*range_, vmax=range_)
    
def inner_product_between_contexts_old(gradient0,gradient1,context0,context1,model_device,names):
    plt.figure(figsize=(16,16))
    inner_dict = dict()
    for i in gradient0:
        inner_product = torch.matmul(gradient0[i].to(model_device).view(1,-1),gradient1[i].to(model_device).view(-1,1))
        inner_dict[i] = inner_product[0][0].item()
    ys = {}
    for name in names:
        ys[name] = []
        for i in inner_dict:
            if name in i:
                ys[name].append(inner_dict[i])
    num_sets = 8
    bar_width = 0.5

    x = np.arange(32)
    a = 0
    for i in ys:
        plt.subplot(4,2,a+1)
        plt.bar(x,ys[i])
        plt.title(i)
        plt.grid(True)
        a+=1
    plt.suptitle(f"{context0}\n and\n {context1}")
    # plt.legend()
    plt.grid(True)
    
def inner_product_heapmap(gradients_model_contexts,model_device,divde,names):
    plt.figure(figsize=(36,16))
    a = 0
    fig_size = plt.gcf().get_size_inches() 
    fontsize = min(fig_size)/2
    for contexta in gradients_model_contexts:
        a_dict = dict()
        for i in names:
            a_dict[i] = np.zeros((6,32))
        # matrix_ = np.zeros((6,32))
        b = 0
        for contextb in gradients_model_contexts:
            gradienta = gradients_model_contexts[contexta]
            gradientb = gradients_model_contexts[contextb]
            inner_dict = dict()
            for i in gradienta:
                if divde:
                    inner_product = torch.matmul(gradienta[i].to(model_device).view(1,-1),gradientb[i].to(model_device).view(-1,1))/(torch.norm(gradienta[i],p=2)*torch.norm(gradientb[i],p=2))
                else:
                    inner_product = torch.matmul(gradienta[i].to(model_device).view(1,-1),gradientb[i].to(model_device).view(-1,1))
                inner_dict[i] = inner_product[0][0].item()
            ys = {} 
            for name in names:
                ys[name] = []
                for i in inner_dict:
                    if name in i:
                        ys[name].append(inner_dict[i])     
            if contexta!=contextb:
                for i in names:
                    a_dict[i][b][:] = ys[i]
                # matrix_[b][:] = ys["mlp.up_proj.weight"]   
            num_sets = 8
            bar_width = 0.5
            b = b+1
        count = 0
        for i in names:
            plt.subplot(8,6,count*6+a+1)
            if a==0 and count==0:
                plt.title(f"{contexta}|{i}",fontsize=fontsize)
            elif a==0: 
                plt.title(f"{i}",fontsize=fontsize)
            elif count==0:
                plt.title(f"{contexta}",fontsize=fontsize)
            # plt.title(f"{contexta}|{i}",fontsize=fontsize)
            sns.heatmap(a_dict[i],cmap="coolwarm", cbar=True,xticklabels=False,vmin=-1,vmax=1)
            plt.xticks(fontsize=fontsize/2)
            plt.yticks(fontsize=fontsize)
            count+=1
        a=a+1
      
def compare_two_matrix_svd(gradient_matrix1,gradient_matrix2,context1,context2,name="model.layers.5.mlp.down_proj.weight",range_=0.002):
    plt.figure(figsize=(16,4))
    fig_size = plt.gcf().get_size_inches()  # Get the current figure size
    fontsize = min(fig_size) * 2 
    matrix_1 = gradient_matrix1[name].to(dtype=torch.float64).cuda()
    matrix_2 = gradient_matrix2[name].to(dtype=torch.float64).cuda()

    U, S, V = torch.svd(matrix_1)
    
    u1 = U[0,:]
    v1 = V[:,0]
    P1 = torch.outer(u1,u1)
    P2 = torch.outer(v1,v1)
    
    F1 = P1@matrix_1
    F2 = matrix_1@P2
    
    col_sums = F1.sum(axis=0)
    sorted_col_indices = torch.argsort(col_sums)
    
    row_sums = F2.sum(axis=1)
    sorted_row_indices = torch.argsort(row_sums)
    
    sorted_matrix1 = matrix_1[sorted_row_indices.tolist()][:, sorted_col_indices.tolist()]
    sorted_matrix2 = matrix_2[sorted_row_indices.tolist()][:, sorted_col_indices.tolist()]
    
    print("Reranked matrix after SVD")
    plt.subplot(1, 2, 1)
    plt.title(context1,fontsize=fontsize)
    sns.heatmap(sorted_matrix1.cpu(), cmap="coolwarm", annot=False, cbar=True, xticklabels=False, yticklabels=False, vmin=-1*range_, vmax=range_)
    plt.subplot(1, 2, 2)
    plt.title(context2,fontsize=fontsize)
    sns.heatmap(sorted_matrix2.cpu(), cmap="coolwarm", annot=False, cbar=True, xticklabels=False, yticklabels=False, vmin=-1*range_, vmax=range_)
    
    matrix_1.cpu()
    matrix_2.cpu()
    