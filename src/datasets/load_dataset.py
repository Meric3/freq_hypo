# from .net import ResNet
# import .net
# from .dataset import dnn_dataset, Make_hypo_dataset, Make_hypo_dataset_weight
from .dataset import Make_hypo_dataset

def load_dataset(opt, log, random_key, dfcases, **kwargs):
    if opt['load_dataset_method'] == 'asd':
        loader = Make_hypo_dataset(opt, random_key, dfcases)
#         log.info("\n\n normal dataset \n")
    elif opt['load_dataset_method'] == 'normal':
        loader = Make_hypo_dataset_2(opt, random_key)
#         log.info("\n\n normal dataset \n")
    elif opt['load_dataset_method'] == 'weight':
        loader = Make_hypo_dataset_weight(opt, random_key)
        log.info("\n\n weight_normal_0121 dataset \n")
    elif opt['load_dataset_method'] == 'fourier':
        loader = Make_hypo_dataset_3(opt, random_key)
#         log.info("\n\n Make_hypo_dataset_3\n")
    elif opt['load_dataset_method'] == 'numpy4':
        loader = Make_hypo_dataset(opt, random_key, dfcases)
#         log.info("\n\n Make_hypo_dataset_3\n")
    else:
        raise Exception("load dataset method can not found")
    
    return loader 
