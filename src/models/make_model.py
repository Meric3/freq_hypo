 # from .net import ResNet
# import .net
# from .model import Net, Informer
from .model import *

def make_model(opt, device, **kwargs):
    
    if opt['model'] == '1d_cnn':
        model = Net(features = opt['features'], task = 'classification', \
                    invasive = opt['invasive'], multi = opt['multi'], ehr_dim= len(opt['colum_ehr']), use_ehr=opt['use_ehr'])
    elif opt['model'] == 'four':
        model = Net(features = opt['features'], task = 'classification', \
                    invasive = opt['invasive'], multi = opt['multi'], check = 'fourier', ehr_dim= len(opt['colum_ehr']), use_ehr=opt['use_ehr'])
    elif opt['model'] == 'test':
        model = Net_freq( window = 200, task = 'classification', invasive = opt['invasive'], multi = opt['multi'], check = 'fourier')
#         print("asdasdasdasdasd")
    elif opt['model'] == 'comb':
        model = Nets(features = opt['features'],  window = 3000, opt = opt, task = 'classification',invasive = opt['invasive'],\
                     multi = opt['multi'], check = 'fourier',  ehr_dim= len(opt['colum_ehr']), use_ehr=opt['use_ehr'])
    else:
        raise Exception("model can not found")
        
    model = model.to(device)
    
    return model 
