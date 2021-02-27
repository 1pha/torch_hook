from dataloader import get_dataloader
from model import CNN

import torch
import torch.nn as nn

def run():

    learning_rate = 0.0001
    num_epoch = 10
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = get_dataloader()
    model = CNN().to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(num_epoch):
        
        model.train()
        for j, [image, label] in enumerate(train_loader):
            
            x = image.to(device=device, dtype=torch.float)
            y_= label.to(device)
            
            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output,y_)
            loss.backward()
            optimizer.step()
            
            
        top_1_count = torch.FloatTensor([0])
        total = torch.FloatTensor([0])
        
        model.eval() 
        for image, label in test_loader:
            x = image.to(device=device, dtype=torch.float)
            y_= label.to(device)

            output = model.forward(x)
            
            values,idx = output.max(dim=1)
            top_1_count += torch.sum(y_==idx).float().cpu().data

            total += label.size(0)

        print("Test Data Accuracy: {}%".format(100 * (top_1_count / total).numpy()))
        if (top_1_count / total).numpy() > 0.98:
            break

    return model, (top_1_count / total).numpy()