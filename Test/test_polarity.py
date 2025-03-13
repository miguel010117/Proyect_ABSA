from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoTokenizer
import pandas as pd
import torch
from colorama import Fore, Style
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,classification_report


def predict_polarity(model_base, model_train,data,name):
    df = pd.read_csv(data, sep=';')
    labels_true = df['polarity']
    textos = list(df['Segmento'])

    labels_pred = []

    if name in ['beto', 'bert', 'albert_base', 'albert_large', 'albert_xx_large']:
        for texto in tqdm(textos):
            labels_pred.append(cargar_bert(model_base,model_train,texto, name))

    elif name in ['bertin_base', 'bertin_large', 'electra_base', 'electra_small']:
        for texto in tqdm(textos):
            labels_pred.append(cargar_bertin(model_base,model_train,texto))

    else:
        for texto in tqdm(textos):
            labels_pred.append(cargar_gpt(model_base,model_train,texto))


    print("\n" + Fore.CYAN + "Métricas:" + Style.RESET_ALL)
    print(classification_report(labels_true, labels_pred, 
                                target_names=[str(i) for i in range(2)], 
                                digits=4)
    )
    matriz(labels_true, labels_pred)


    

def classifySentiment(review_text, tokenizer):
    
    encoding = tokenizer.encode_plus(
        review_text,
        add_special_tokens=True,
        max_length=250,
        padding="max_length",
        return_token_type_ids=False,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return encoding


# Cargar Bert, Beto, Albert
def cargar_bert(model_base, model_train, review_text, name):
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    if name in [ 'albert_large', 'albert_xx_large']:
        model = torch.load(model_train)
        model.eval()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_train)
    
    
    encoding = classifySentiment(review_text, tokenizer)

    # Mover los tensores al mismo dispositivo que el modelo
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Realizar la inferencia
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        output = model(input_ids, attention_mask)
        logits = output.logits

    _, prediction = torch.max(logits, dim=1)


    return prediction.item()



def cargar_bertin(model_base, model_train, review_text):
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    model = torch.load(model_train)
    model.eval()

    encoding = classifySentiment(review_text, tokenizer)

    # Mover los tensores al mismo dispositivo que el modelo
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Realizar la inferencia
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        output = model(input_ids, attention_mask)
        logits = output.logits
        cls_logits = logits[:,0,:]

    _, prediction = torch.max(cls_logits, dim=1)

    return prediction.item()

def cargar_gpt(model_base, model_train, review_text):
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    # Agrega el token de padding (si no existe)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    # Cargar el modelo y el tokenizador
    model = torch.load(model_train)
    model.resize_token_embeddings(len(tokenizer)) # Esencial para actualizar el modelo con el nuevo token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    encoding = classifySentiment(review_text, tokenizer)

    # Realizar la inferencia
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        output = model(**encoding)
        logits = output.logits

    _, prediction = torch.max(logits, dim=1)

    return prediction.item()

def matriz(labels_true, labels_pred):
    df_cm = confusion_matrix(labels_true, labels_pred,labels=[0,1])
    heatmap = sns.heatmap(df_cm,annot=True,fmt="d")
    heatmap.yaxis.set_ticklabels([0,1],rotation=0,ha='right')
    heatmap.xaxis.set_ticklabels([0,1],rotation=45,ha='right')
    plt.ylabel('Valor Verdadero')
    plt.xlabel('Valor Predicho')
    plt.show()