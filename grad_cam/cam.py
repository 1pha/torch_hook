import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Function

class CAM():

    def __init__(self,model):
        self.gradient = []
        self.model = model
        self.h = self.model.model.module.layer[-2].register_backward_hook(self.save_gradient)
        
    def save_gradient(self,*args):
        #print("Gradient saved!!!!")
        grad_input = args[1]
        grad_output= args[2]
        self.gradient.append(grad_output[0])
        #print(self.gradient[0].size())
        
    def get_gradient(self):
        return self.gradient[0]
    
    def remove_hook(self):
        self.h.remove()
            
    def normalize_cam(self,x):
        x = 2*(x-torch.min(x))/(torch.max(x)-torch.min(x)+1e-8)-1
        # x[x<torch.max(x)]=-1
        return x
    
    def visualize(self,cam_img,guided_img,img_var):
        guided_img = guided_img.numpy()
        cam_img = resize(cam_img.cpu().data.numpy(),output_shape=(28,28))
        x = img_var[0,:,:].cpu().data.numpy()

        fig = plt.figure(figsize=(20, 12)) 
        
        plt.subplot(1,4,1)
        plt.title("Original Image")
        plt.imshow(x,cmap="gray")

        plt.subplot(1,4,2)
        plt.title("Class Activation Map")
        plt.imshow(cam_img)

        plt.subplot(1,4,3)
        plt.title("Guided Backpropagation")
        plt.imshow(guided_img,cmap='gray')
        
        plt.subplot(1,4,4)
        plt.title("Guided x CAM")
        plt.imshow(guided_img*cam_img,cmap="gray")
        plt.show()
    
    def get_cam(self,idx):
        grad = self.get_gradient()
        alpha = torch.sum(grad,dim=3,keepdim=True)
        alpha = torch.sum(alpha,dim=2,keepdim=True)
        
        cam = alpha[idx]*grad[idx]
        cam = torch.sum(cam,dim=0)
        cam = self.normalize_cam(cam)
        self.remove_hook()
        
        return cam


class GuidedBackpropRelu(Function):

    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx,grad_output):
        input = ctx.saved_tensors[0] # forwarding 할 때의 입력값
        grad_input = grad_output.clone() # backward 할 때의 입력된 미분 값
        grad_input[grad_input<0] = 0 # 미분 양인 것만
        grad_input[input<0]=0 # 입력도 양인 것만
        return grad_input
     

class GuidedReluModel(nn.Module):

    def __init__(self,model,to_be_replaced,replace_to):
        super(GuidedReluModel,self).__init__()
        self.model = model
        self.to_be_replaced = to_be_replaced
        self.replace_to = replace_to
        self.layers = []
        self.output = []
        
        for m in self.model.modules():
            if isinstance(m,self.to_be_replaced):
                self.layers.append(self.replace_to )
                #self.layers.append(m)

            elif isinstance(m,nn.Conv2d):
                self.layers.append(m)

            elif isinstance(m,nn.BatchNorm2d):
                self.layers.append(m)

            elif isinstance(m,nn.Linear):
                self.layers.append(m)

            elif isinstance(m,nn.AvgPool2d):
                self.layers.append(m)
                
        for i in self.layers:
            print(i)
        
    def reset_output(self):
        self.output = []
    
    def hook(self, grad):
        out = grad[:, 0, :, :].cpu().data#.numpy()
        print("out_size:",out.size())
        self.output.append(out)
        
    def get_visual(self, idx, original_img):
        grad = self.output[0][idx]
        return grad
        
    def forward(self,x):
        out = x 
        out.register_hook(self.hook)
        for i in self.layers[:-3]:
            out = i(out)
        out = out.view(out.size()[0],-1)
        for j in self.layers[-3:]:
            out = j(out)
        return out