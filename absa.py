from abc import ABC, abstractmethod
import torch
import numpy as np
import random
import os


class AbsaModel(ABC):
    
    def __init__(self, seed):
        
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    @abstractmethod
    def train(self, train_loader, test_loader, epochs):
        
        pass
    
    @abstractmethod
    def predict(self, text, model):
        
        pass
    
    
    def save_model(self, model, name):
        
        torch.save(model.state_dict(), name)


    def load_model(self, model, path):
        
        model.load_state_dict(torch.load(path), strict=False)
        return model
