from Aspect_extraction.aspect_term_extraction import AspectTermExtraction, DatasetLoaderAspect
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report

class ABSAPipeline:
    
    def __init__(self, aspect_model):
        self.aspect_model = AspectTermExtraction(model=aspect_model, tokenizer=aspect_model, seed=50)

    def train_aspect_model(self, train_file, test_file, model_type,name, batch_size, num_epochs):
        train_ds_aspect = DatasetLoaderAspect(pd.read_csv(train_file, sep=';'), model_type)
        test_ds_aspect = DatasetLoaderAspect(pd.read_csv(test_file, sep=';'), model_type)
        aspect_train_loader = DataLoader(train_ds_aspect, batch_size=batch_size,
                                         collate_fn=self.aspect_model.create_mini_batch)
        aspect_test_loader = DataLoader(test_ds_aspect, batch_size=batch_size,
                                        collate_fn=self.aspect_model.create_mini_batch)
        self.aspect_model.train(aspect_train_loader, aspect_test_loader,num_epochs, name)

    def predict_aspect(self, text, name):
        max_length = 512  # Longitud máxima de la secuencia admitida por BERT
        if len(text) > max_length:
            text = text[:max_length]  
        return self.aspect_model.predict(text, self.aspect_model.model,name)
    
    
    
def predicted_bitmask(rev,aspects):
        binary_list = []
        for palabra in rev:
            if(palabra.lower() in aspects):
                binary_list.append(1)
            else:
                binary_list.append(0)
        return binary_list
          
def metrics(true_labels, predicted_labels):
    true = []
    predicted = []
    for string in true_labels:
        tags = string[1:-1].split(',')
        for value in tags:
            true.append(int(value))
    for string in predicted_labels:
        for s in string:
            predicted.append(s)
    print(classification_report(true, predicted, 
                                target_names=[str(i) for i in range(2)], 
                                digits=4)
    )
