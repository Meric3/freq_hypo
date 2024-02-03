import torch

import numpy as np
import sys 
# import tqdm
# sys.path.append((os.path.abspath(os.path.dirname(__file__)))
# from utils.util import make_fourier

weight_list = [0.52, 0.007, 0.24, 0.22] 
weight_list_normal = [0.562, 0.0002, 0.26, 0.176] 


def train(epoch, model,  loader, criterion, optimizer, device, opt):
    model.train()
    current_loss = 0

    
    for idx, (inputs, ehr, target, caseid) in enumerate(loader):
        inputs, target, ehr = inputs.to(device), target.to(device), ehr.to(device)
        optimizer.zero_grad()
#         import pdb; pdb.set_trace()
#         print(inputs)
        try:
            output = model( inputs, ehr )
#             print(output)
        except:
            continue
        loss = criterion(output.T[0], target)
        current_loss += loss.item()*inputs.size(0)
        loss.backward()
        optimizer.step()


    current_loss = current_loss/len(loader.dataset)

    return current_loss 



def evaluate(model,  loader, criterion, optimizer, device, opt):
    model.eval()
    target_stack = []
    output_stack = []
    case_stack = []
    input_stack = []
    current_loss = 0
    
    with torch.no_grad():
        for inputs, ehr, target, caseid  in loader:
            input_stack.extend (np.array (inputs) )
            inputs, target, ehr = inputs.to(device), target.to(device), ehr.to(device)
            output = model( inputs , ehr)
            target_stack.extend ( np.array ( target.cpu() ) )
            output_stack.extend ( np.array ( output.cpu().T[0] ) )
            case_stack.extend (np.array (caseid) )
            

            loss = criterion(output.T[0], target)
            current_loss += loss.item()*inputs.size(0)
    current_loss = current_loss/len(loader.dataset)    
    return current_loss, target_stack, output_stack, case_stack, input_stack

























