from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import  AutoModel, AutoTokenizer
from Aspect_extraction.absamodel import AbsaModel
from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import logging

#FINE-TUNNED BERT
class ModelForAspectExtraction(torch.nn.Module):
    def __init__(self, model):
        super(ModelForAspectExtraction, self).__init__()
        self.model = AutoModel.from_pretrained(model)
        self.drop = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, tags_tensors, masks_tensors):
    
        cls_output = self.model(input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False)
        drop_output = self.drop(cls_output[0])
        linear_outputs = self.linear(drop_output)
        
        if tags_tensors is not None:
            tags_tensors = tags_tensors.view(-1)
            linear_outputs = linear_outputs.view(-1, 2)
            loss = self.loss_fn(linear_outputs, tags_tensors)
            return loss
        else:
            return linear_outputs


class AspectTermExtraction(AbsaModel):
    
    def __init__(self, model, tokenizer, seed, DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        super().__init__(seed)
        self.model = ModelForAspectExtraction(model)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, strict=False)
        self.DEVICE = DEVICE
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5) #5e-6
        logging.set_verbosity_error()

    def create_mini_batch(self, samples):
        

        ids_tensors = [s[1] for s in samples]
        ids_tensors = pad_sequence(ids_tensors, batch_first=True)
        tags_tensors = [s[2] for s in samples]
        tags_tensors = pad_sequence(tags_tensors, batch_first=True)
        masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)
        return ids_tensors, tags_tensors, masks_tensors

    def train(self, train_loader,test_loader, epochs):
    
        num_data = len(train_loader) # = 140
        

        for epoch in range(epochs):
            data_processed = 0
            losses = []
            batch_count = 0

            for data in train_loader:
                batch_count += 1
                ids_tensors, tags_tensors, masks_tensors = data
                ids_tensors = ids_tensors.to(self.DEVICE)
                tags_tensors = tags_tensors.to(self.DEVICE)
                masks_tensors = masks_tensors.to(self.DEVICE)

                loss = self.model(ids_tensors=ids_tensors, tags_tensors=tags_tensors, masks_tensors=masks_tensors)

                losses.append(loss.item())

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                data_processed += 1
                print('epoch:', epoch+1, " batch:", data_processed, "/", num_data, " loss:", np.mean(losses))

            # Save the model a   
            self.save_model(self.model, 'electra_base_spanish_epoch_' + str(epoch+1) + '.pkl')
            # true_labels, predicted_labels = self.test(test_loader)

            # print(classification_report(true_labels, predicted_labels, target_names=[str(i) for i in range(2)]))

    def test(self, data_loader):
        """
        Computes predictions using the trained model on the given data.

        Args:
            data_loader: A PyTorch DataLoader object containing the data to be tested.

        Returns:
            A tuple of two lists containing the true labels and predicted labels respectively.

        """
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for data in data_loader:
                input_ids, tags, _, attention_masks = data

                input_ids = input_ids.to(self.DEVICE)
                tags = tags.to(self.DEVICE)
                attention_masks = attention_masks.to(self.DEVICE)

                outputs = self.model(ids_tensors=input_ids, tags_tensors=None, masks_tensors=attention_masks)

                _, predictions = torch.max(outputs, dim=2)

                predicted_labels += list([int(j) for i in predictions for j in i])
                true_labels += list([int(j) for i in tags for j in i])

        return true_labels, predicted_labels
    

    def predict(self, sentence, model): #SOFIA devuelve el comentario tokenizado y los aspectos(terms)

        # Elimina corchetes, comillas simples y espacios
        palabras_str = sentence.replace('[', '').replace(']', '').replace("'", '').strip()

        # Separa por comas
        palabras = palabras_str.split(',')

        # Limpia cada palabra
        palabras_limpias = [palabra.replace(" ", "") for palabra in palabras]
      
        sentence = " ".join(palabras_limpias)
        tokenized_sentence = []
        terms = []
        word = ""

        tokens = self.tokenizer.tokenize(sentence) 
        tokenized_sentence += tokens
        #print(tokenized_sentence)
        
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        input_tensor = torch.tensor([input_ids]).to(self.DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor, None, None)
            softmax_outputs = torch.softmax(outputs, dim=2) #lleva la salida de BETO de 768 dimensiones a 2, una para cada clase (aspecto o no aspecto).
            # print("SM", softmax_outputs)
            _, predictions = torch.max(softmax_outputs, dim=2) #SOFIA predictions tiene la bitmask en 0
            # print("P", predictions)
            # print("P", predictions[0].tolist())
        predicted_tokens = predictions[0].tolist() #SOFIA predicted tokens es la bitmask
        for i, flag in enumerate(predicted_tokens): #en flag guarda el valor de la bitmask
            
            if flag == 1 and tokenized_sentence[0][0] == '▁':
                
                if tokenized_sentence[i][0] != '▁':
                    if word:
                        word = word[:-1] + tokenized_sentence[i].replace("▁", "") + ' '
                    else:    
                        word = tokenized_sentence[i-1].replace("▁", "") + tokenized_sentence[i].replace("▁", "") + ' '
                else:
                    word = tokenized_sentence[i].replace("▁", "") + ' '

                    

                if len(tokenized_sentence) - 1 == i:
                    terms.append(word.strip())
                    word = ""

                elif tokenized_sentence[i+1][0] != '▁':
                    pass

                else:
                    terms.append(word.strip())
                    word = ""

            elif flag == 1 and tokenized_sentence[1][0] == 'Ġ':
                
                if tokenized_sentence[i][0] != 'Ġ':
                    if word:
                        word = word[:-1] + tokenized_sentence[i].replace("Ġ", "") + ' '
                    else:    
                        word = tokenized_sentence[i-1].replace("Ġ", "") + tokenized_sentence[i].replace("Ġ", "") + ' '
                else:
                    word = tokenized_sentence[i].replace("Ġ", "") + ' '

                    

                if len(tokenized_sentence) - 1 == i:
                    terms.append(word.strip())
                    word = ""

                elif tokenized_sentence[i+1][0] != 'Ġ':
                    pass

                else:
                    terms.append(word.strip())
                    word = ""

            elif flag == 1:
            
                if tokenized_sentence[i][0] == '▁':
                    word = tokenized_sentence[i].replace('▁', "")

                elif tokenized_sentence[i][0]== 'Ġ':
                    word = tokenized_sentence[i].replace('Ġ', "")

                elif tokenized_sentence[i].startswith("##"):
                    if word:
                        word = word[:-1] + tokenized_sentence[i].replace("##", "") + ' '
                    else:    
                        word = tokenized_sentence[i-1] + tokenized_sentence[i].replace("##", "") + ' '
                else:
                    word = tokenized_sentence[i] + ' '


                if len(tokenized_sentence) - 1 == i:
                    terms.append(word.strip())
                    word = ""

                elif tokenized_sentence[i+1].startswith("##"):
                    pass

                else:
                    terms.append(word.strip())
                    word = ""
           


        return tokenized_sentence, terms
    
def unir_tokens(tokens):
 
  resultado = []
  i = 0
  while i < len(tokens):
    if tokens[i].startswith('##'):
      # Si el token actual comienza con ##, únelo al anterior
      resultado[-1] += tokens[i][2:]
    else:
      # De lo contrario, agrega el token actual a la lista de resultados
      resultado.append(tokens[i])
    i += 1

  return resultado

class DatasetLoaderAspect(Dataset):

    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __getitem__(self, idx):
        
        tokens, tags = self.df.iloc[idx, :2].values

        tokens = tokens.replace("'", "").strip("][").split(', ')
        tags = tags.strip('][').split(', ')

        model_tokens = []
        model_tags = []
        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i])
            model_tokens += t
            model_tags += [int(tags[i])] * len(t)

        
        model_ids = self.tokenizer.convert_tokens_to_ids(model_tokens)
        ids_tensor = torch.tensor(model_ids)
        tags_tensor = torch.tensor(model_tags)
        
        
        
        return model_tokens, ids_tensor, tags_tensor

    def __len__(self):
        
        return len(self.df)