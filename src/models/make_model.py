from .model import *

def make_model(opt, device, **kwargs):
    
    if opt['model'] == '1d_cnn':
        model = Net(features = opt['features'], task = 'classification', \
                    invasive = opt['invasive'], multi = opt['multi'], ehr_dim= len(opt['colum_ehr']), use_ehr=opt['use_ehr'])
    elif opt['model'] == 'comb':
        model = Nets(features = opt['features'],  window = 3000, opt = opt, task = 'classification',invasive = opt['invasive'],\
                     multi = opt['multi'], check = 'fourier',  ehr_dim= len(opt['colum_ehr']), use_ehr=opt['use_ehr'])
    elif opt['model'] == 'time_comb':
        model = Nets(features = opt['features'],  window = 3000, opt = opt, task = 'classification',invasive = opt['invasive'],\
                     multi = opt['multi'], check = 'fourier',  ehr_dim= len(opt['colum_ehr']), use_ehr=opt['use_ehr'])
    elif opt['model'] == 'lstm':
        model = LSTM_time(features = opt['features'], task = 'classification', \
                    invasive = opt['invasive'], multi = opt['multi'], ehr_dim= len(opt['colum_ehr']), use_ehr=opt['use_ehr'])  
    elif opt['model'] == 'knn':
        model = KNN()

    elif opt['model'] == 'svm':
        model = SVM()

    elif opt['model'] == 'rf': #random forest
        model = RF()

    elif opt['model'] == 'xgboost': 
        model = XGBoost()
    elif opt['model'] == 'vae':
        model = VAE(latent_dim = 20, features = opt['features'],  window = 3000, opt = opt, task = 'classification',invasive = opt['invasive'],\
                     multi = opt['multi'], check = 'fourier',  ehr_dim= len(opt['colum_ehr']), use_ehr=opt['use_ehr'])
    elif opt['model'] == 'usad':
        model = UsadModel(w_size = 3000, z_size= 512, features = opt['features'],  window = 3000, opt = opt, task = 'classification',invasive = opt['invasive'],\
                     multi = opt['multi'], check = 'fourier',  ehr_dim= len(opt['colum_ehr']), use_ehr=opt['use_ehr'])

    else:
        raise Exception("model can not found")
        
    model = model.to(device)
    
    return model 
