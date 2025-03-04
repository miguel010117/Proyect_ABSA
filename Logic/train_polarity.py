from transformers import get_linear_schedule_with_warmup,AutoTokenizer,AutoModelForSequenceClassification, AutoModel, AlbertForSequenceClassification
from textwrap import wrap
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import torch
import pickle
from tqdm import tqdm


class CoahDataset(Dataset):

    def __init__(self, reviews, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self,index):
        review = str(self.reviews[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = "max_length",
            return_token_type_ids = False,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        
        return{
            'review': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype= torch.long)
        }
    


def train_polarity_model(modelo,train_data,num_epochs, batch):

    RANDOM_SEED = 42
    MAX_LEN = 250
    BATCH_SIZE = batch
    EPOCHES = num_epochs
    DATASET_PATH = train_data
    NCLASES = 2

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df = pd.read_csv(DATASET_PATH, sep=';')

    MODEL_TRAIN = modelo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TRAIN)

    # Agrega el token de padding (si no existe)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("PAD")

    df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)

    train_data_loader = data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_TRAIN)
    model.dropout = nn.Dropout(0.3)
    model.classifier = nn.Linear(model.config.hidden_size, 2) 
    model.resize_token_embeddings(len(tokenizer)) # Esencial para actualizar el modelo con el nuevo token
    model.config.pad_token_id = tokenizer.pad_token_id


    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader)  *  EPOCHES
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps= 0,
        num_training_steps= total_steps
    )
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHES):
        print('Epoch {} de {}'.format(epoch+1, EPOCHES))
        print('---------------------------')
        train_acc, train_loss = train_model(
            model, train_data_loader, loss_fn, optimizer, scheduler, len(df_train)
        )
        print('Entrenamiento: Loss: {}, accuracy: {}'.format(train_loss,train_acc) )
        test_acc, test_loss = eval_model(
            model, test_data_loader, loss_fn, len(df_test)
        )
        print('Validacion: Loss: {}, accuracy: {}'.format(test_loss,test_acc) )
        torch.save(model, 'Albert_Large.pth')


def data_loader(df,tokenizer, max_len , batch_size):
    dataset = CoahDataset(
        reviews = df['review'].to_numpy(),
        labels = df['sentiment'].to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )

    return DataLoader(dataset, batch_size = batch_size, num_workers= 0)


def train_model(model, data_loader,loss_fn,optimizer,scheduler,n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        logits = outputs.logits

        try:
            # Intenta el método para clasificación multiclase simple. Si falla, prueba el otro método.
            _, preds = torch.max(logits, dim=1)
            loss = loss_fn(logits, labels)      

        except RuntimeError as e:
            
            # Método para modelos que usan el primer token [CLS] para la clasificación.
            cls_logits = logits[:, 0, :]
            _, preds = torch.max(cls_logits, dim=1)
            loss = loss_fn(cls_logits, labels)
                    
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double()/n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn,n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            logits = outputs.logits

            try:
                # Intenta el método para clasificación multiclase simple. Si falla, prueba el otro método.
                _, preds = torch.max(logits, dim=1)
                loss = loss_fn(logits, labels)      

            except RuntimeError as e:
            
                # Método para modelos que usan el primer token [CLS] para la clasificación.
                cls_logits = logits[:, 0, :]
                _, preds = torch.max(cls_logits, dim=1)
                loss = loss_fn(cls_logits, labels)
        
        
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    return correct_predictions.double()/n_examples, np.mean(losses)