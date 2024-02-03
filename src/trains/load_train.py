


def load_train(opt, log, device, **kwargs):
    if opt['train_method'] == 'none':
        from .train import train as train        
        from .train import evaluate as evaluate
        log.info("\n\n Train normal \n")
        return train, evaluate
    elif opt['modification'] == 'fourier':
        from .train import train_fourier_2 as train        
        from .train import evaluate_fourier_2 as evaluate
        log.info("\n\n Train lib: fourier2222  \n")
        return train, evaluate
    elif opt['modification'] == 'new':
        from .train import train_new as train        
        from .train import evaluate_new as evaluate
        log.info("\n\n Train lib: NEW  \n")
        return train, evaluate
    elif opt['modification'] == 'fourier_1228':    
        from .train import train_fourier_3 as train        
        from .train import evaluate_fourier_3 as evaluate
        log.info("\n\n Train lib: 1229  \n")
        return train, evaluate
    elif opt['modification'] == 'weight_0109':    
        from .train import train_fourier_weight as train        
        from .train import evaluate_fourier_weight as evaluate
        log.info("\n\n weight_0109  \n")
        return train, evaluate
    elif opt['modification'] == 'weight_normal_0121':    
        from .train import train_weight_normal as train        
        from .train import evaluate_weight_normal as evaluate
        log.info("\n\n weight_normal_0121  \n")
        return train, evaluate
    elif opt['modification'] == 'normal_modificiation':    
        from .train import train as train        
        from .train import evaluate as evaluate    
        log.info("\n\n Train lib: NORMAL  \n")
        return train, evaluate
    else:
        from .train import train as train        
        from .train import evaluate as evaluate    
        log.info("\n\n Train lib: NORMAL  \n")
        return train, evaluate