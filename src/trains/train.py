import torch

import numpy as np
import sys 

weight_list = [0.52, 0.007, 0.24, 0.22] 
weight_list_normal = [0.562, 0.0002, 0.26, 0.176] 


def train(epoch, model,  loader, criterion, optimizer, device, opt):
    model.train()
    current_loss = 0
    for idx, (inputs, ehr, target, caseid) in enumerate(loader):
        inputs, target, ehr = inputs.to(device), target.to(device), ehr.to(device)
        optimizer.zero_grad()
        try:
            output = model( inputs, ehr )
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




def train_vae(epoch, model,  loader, criterion, optimizer, device, opt):
    import torch.nn.functional as F
    def criterion(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3000), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    model.train()
    current_loss = 0
    for idx, (inputs, ehr, target, caseid) in enumerate(loader):
        inputs, target, ehr = inputs.to(device), target.to(device), ehr.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model( inputs, ehr )
        loss = criterion(recon_batch, inputs, mu, logvar)
        current_loss += loss.item()*inputs.size(0)
        loss.backward()
        optimizer.step()
    current_loss = current_loss/len(loader.dataset)

    return current_loss 



def evaluate_vae(model,  loader, criterion, optimizer, device, opt):
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
            output, _, _ = model( inputs , ehr)
            target_stack.extend ( np.array ( target.cpu() ) )
            output_stack.extend ( np.array ( output.cpu().T[0] ) )
            case_stack.extend (np.array (caseid) )
            

            loss = criterion(output.T[0], target)
            current_loss += loss.item()*inputs.size(0)
    current_loss = current_loss/len(loader.dataset)    
    return current_loss, target_stack, output_stack, case_stack, input_stack


def train_usad(epoch, model,  loader, criterion, optimizer, device, opt):
    # import torch.nn.functional as F
    # def criterion(recon_x, x, mu, logvar):
    #     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3000), reduction='sum')
    #     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #     return BCE + KLD

    # model.train()
    # current_loss = 0
    # for idx, (inputs, ehr, target, caseid) in enumerate(loader):
    #     inputs, target, ehr = inputs.to(device), target.to(device), ehr.to(device)
    #     optimizer.zero_grad()
    #     recon_batch, mu, logvar = model( inputs, ehr )
    #     loss = criterion(recon_batch, inputs, mu, logvar)
    #     current_loss += loss.item()*inputs.size(0)
    #     loss.backward()
    #     optimizer.step()
    # current_loss = current_loss/len(loader.dataset)


    current_loss = 0
    optimizer1 = torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    # for epoch in range(epochs):
    for idx, (inputs, ehr, target, caseid) in enumerate(loader):
        inputs, target, ehr = inputs.to(device), target.to(device), ehr.to(device)

        
        #Train AE1
        loss1,loss2 = model.training_step(inputs,epoch+1)
        loss1.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        
        
        #Train AE2
        loss1,loss2 = model.training_step(inputs,epoch+1)
        loss2.backward()
        optimizer2.step()
        optimizer2.zero_grad()

        current_loss += loss1.item()*inputs.size(0)

    return current_loss 



def evaluate_usad(model,  loader, criterion, optimizer, device, opt):
    model.eval()
    target_stack = []
    output_stack = []
    case_stack = []
    input_stack = []
    current_loss = 0
    
    # with torch.no_grad():
    #     for inputs, ehr, target, caseid  in loader:
    #         input_stack.extend (np.array (inputs) )
    #         inputs, target, ehr = inputs.to(device), target.to(device), ehr.to(device)
    #         output, _, _ = model( inputs , ehr)
    #         target_stack.extend ( np.array ( target.cpu() ) )
    #         output_stack.extend ( np.array ( output.cpu().T[0] ) )
    #         case_stack.extend (np.array (caseid) )
            

    #         loss = criterion(output.T[0], target)
    #         current_loss += loss.item()*inputs.size(0)
    # current_loss = current_loss/len(loader.dataset)    




    # if val_loader is None:
    #   trainval=train_loader
    # else:
    #   trainval = [d for dl in [train_loader, val_loader] for d in dl]
    results=[]
    alpha=.5; beta=.5; contamination=.1
    with torch.no_grad():
        for inputs, ehr, target, caseid  in loader:
            input_stack.extend (np.array (inputs) )
            target_stack.extend ( np.array ( target.cpu() ) )
            case_stack.extend (np.array (caseid) )
            inputs, target, ehr = inputs.to(device), target.to(device), ehr.to(device)
            w1=model.decoder1(model.encoder(inputs))
            w2=model.decoder2(model.encoder(w1))
            results.append(alpha*torch.mean((inputs-w1)**2,dim=1)+beta*torch.mean((inputs-w2)**2,dim=1))
        

    score_pred=np.concatenate([torch.stack(results[:-1]).flatten().cpu().detach().numpy(), 
                                            results[-1].flatten().cpu().detach().numpy()])
    threshold =  np.sort(score_pred)[int(len(score_pred)*(1-contamination))]
    y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                            results[-1].flatten().detach().cpu().numpy()])
    # threshold=results_threshold # Decide on your own threshold
    y_pred_label = [1.0 if (score > threshold) else 0 for score in y_pred ]
    return current_loss, target_stack, y_pred_label, case_stack, input_stack
























