


def load_train(opt, log, device, **kwargs):
    if opt['train_method'] == 'none':
        from .train import train as train        
        from .train import evaluate as evaluate
        return train, evaluate
    elif opt['train_method'] == 'vae':
        from .train import train_vae as train        
        from .train import evaluate_vae as evaluate
        return train, evaluate
    elif opt['train_method'] == 'usad':
        from .train import train_usad as train        
        from .train import evaluate_usad as evaluate
        return train, evaluate
    else:
        from .train import train as train        
        from .train import evaluate as evaluate    
        return train, evaluate