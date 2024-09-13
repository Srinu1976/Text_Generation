#!/usr/bin/env python
# coding: utf-8

# #                    **TEXT GENERATION USING LSTM IN PYTORCH**

# This tutorial demonstrates how to generate text using a **word-based LSTM**. We are going to work **wikitext-2** dataset. The dataset can be downloaded from [here](https://s3.amazonaws.com/fast-ai-nlp/wikitext-2.tgz). Given a corpus of text, we have to train a model using word-level LSTM and test the model on the test data.
# We are going to use **perplexity** as an evaluation metric.

# In[1]:


#IMPORTING LIBRARIES AND MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
from nltk import word_tokenize,sent_tokenize
import gc
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F


# # LOAD THE DATA

# In[2]:


#Loading data
def load_data(filepath):
    
    f=open(filepath)
    return f.read()


# # CLEAN THE DATA

# In[3]:


#Cleaning data
def Clean_data(data):
    """Removes all the unnecessary patterns and cleans the data to get a good sentence"""
    repl='' #String for replacement
    
    #removing all open brackets
    data=re.sub('\(', repl, data)
    
    #removing all closed brackets
    data=re.sub('\)', repl, data)
    
    #Removing all the headings in data
    for pattern in set(re.findall("=.*=",data)):
        data=re.sub(pattern, repl, data)
    
    #Removing unknown words in data
    for pattern in set(re.findall("<unk>",data)):
        data=re.sub(pattern,repl,data)
    
    #Removing all the non-alphanumerical characters
    for pattern in set(re.findall(r"[^\w ]", data)):
        repl=''
        if pattern=='-':
            repl=' '
        #Retaining period, apostrophe
        if pattern!='.' and pattern!="\'":
            data=re.sub("\\"+pattern, repl, data)
            
    return data


# # SPLIT DATA INTO WORDS AND SENTENCES

# In[4]:


def split_data(data, num_sentences=-1):
    """Splits text data into words and sentences """
    #Sentence tokenization
    if num_sentences==-1:
        sentences=sent_tokenize(data)
    else:
        sentences=sent_tokenize(data)[:num_sentences]
    
    #Word tokenization
    words=set()
    for sent in sentences:
        for word in str.split(sent,' '):
            words.add(word)
    words=list(words)
    
    #Adding empty string in list of words to avoid confusion while padding.
    #Padded zeroes can be interpreted as empty strings.
    words.insert(0,"")
    
    return sentences, words


# # CONVERT TEXT DATA INTO NUMERICAL FORM

# In[5]:


def Convert_data(sentences, words, seq_len):
    """Converts text data into numerical form"""
    
    sent_sequences=[]
    for i in range(len(sentences)):
        words_in_sent=str.split(sentences[i],' ')
        for j in range(1,len(words_in_sent)):
            if j<=(seq_len):
                sent_sequences.append(words_in_sent[:j])
            elif j>seq_len and j<len(words_in_sent):
                sent_sequences.append(words_in_sent[j-seq_len:j])
            elif j>len(words_in_sent)-seq_len:
                sent_sequences.append(words_in_sent[j-seq_len:])
                
    #The above code converts the text data into the following sequences
    #[['The', '2013'],
    #['The', '2013', '14'],
    #['The', '2013', '14', 'season'],
    #['The', '2013', '14', 'season', 'was']]
    
    #Splitting into predictors and class_labels
    predictors=[];class_labels=[]
    for i in range(len(sent_sequences)):
        predictors.append(sent_sequences[i][:-1])
        class_labels.append(sent_sequences[i][-1])
    
    #Padding the predictors manually with Empty strings
    pad_predictors=[]
    for i in range(len(predictors)):
        emptypad=['']*(seq_len-len(predictors[i])-1)
        emptypad.extend(predictors[i])
        pad_predictors.append(emptypad)
        
    #The following two chunks of code are useful to convert text into numeric form
    #Dictionary with words as keys and indices as values
    global word_ind
    word_ind=dict()
    for ind,word in enumerate(words):
        word_ind[word]=ind
    
    #Dictionary with indices as keys and words as values
    global ind_word
    ind_word=dict()
    for ind,word in enumerate(words):
        ind_word[ind]=word
        
    #Convert each word into their respective index
    for i in range(len(pad_predictors)):
        for j in range(len(pad_predictors[i])):
            pad_predictors[i][j]=word_ind[pad_predictors[i][j]]
        class_labels[i]=word_ind[class_labels[i]]
        
    #Convert sequences to tensors
    for i in range(len(pad_predictors)):
        pad_predictors[i]=torch.tensor(pad_predictors[i])
    pad_predictors=torch.stack(pad_predictors)
    class_labels=torch.tensor(class_labels)
     
    return pad_predictors, class_labels


# # DEFINE THE MODEL

# In[6]:


class LSTM(nn.Module):
    """Base class for all neural network modules.
       All models should subclass this class"""
    def __init__(self,num_embeddings, embedding_dim, padding_idx, hidden_size, Dropout_p, batch_size):
        super(LSTM,self).__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.padding_idx=padding_idx
        self.hidden_size=hidden_size
        self.dropout=Dropout_p
        self.batch_size=batch_size
        
        #Adding Embedding Layer
        self.Embedding=nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        
        #Adding LSTM Layer
        self.lstm=nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        
        #Adding Dropout Layer
        self.dropout=nn.Dropout(Dropout_p)
        
        #Adding fully connected dense Layer
        self.FC=nn.Linear(hidden_size, num_embeddings)
        
    def init_hidden(self, batch_size):
        """Initializes hiddens state tensors to zeros"""
        
        state_h=torch.zeros(1, batch_size, self.hidden_size)
        state_c=torch.zeros(1, batch_size, self.hidden_size)
        
        return (state_h,state_c)
        
    def forward(self,input_sequence, state_h,state_c):
        
        #Applying embedding layer to input sequence
        Embed_input=self.Embedding(input_sequence)
        
        #Applying LSTM layer
        output,(state_h,state_c)=self.lstm(Embed_input, (state_h,state_c)) 
        
        #Applying fully connected layer
        logits=self.FC(output[:,-1,:])
         
        return logits,(state_h,state_c)
    
    def topk_sampling(self, logits, topk):
        """Applies softmax layer and samples an index using topk"""
        
        #Applying softmax layer to logits
        logits_softmax=F.softmax(logits,dim=1)
        values,indices=torch.topk(logits_softmax[0],k=topk)
        choices=indices.tolist()
        sampling=random.sample(choices,1)
        
        return ind_word[sampling[0]]


# # DIVIDE INTO BATCHES

# In[7]:


def get_batch(pad_predictors, class_labels, batch_size):
    for i in range(0, len(pad_predictors), batch_size):
        if i+batch_size<len(pad_predictors):
            yield pad_predictors[i:i+batch_size], class_labels[i:i+batch_size]


# # TRAIN THE MODEL

# In[8]:


def train_model(pad_predictors, class_labels, n_vocab, embedding_dim, padding_idx, hidden_size, Dropout_p, batch_size, lr):
    """Trains an LSTM Model"""
    #Creates instance of LSTM class
    model=LSTM(n_vocab, embedding_dim, padding_idx, hidden_size, Dropout_p, batch_size)
    
    #Creates instance of CrossEntropLoss class
    criterion=nn.CrossEntropyLoss(ignore_index=0)
    
    #Creates instance of Adam optimizer class
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    num_epochs=10
    for epoch in range(num_epochs):
        state_h, state_c=model.init_hidden(batch_size)
        
        total_loss=0
        for x, y in get_batch(pad_predictors, class_labels, batch_size):
            #sets model in training mode
            model.train()
            
            state_h=state_h.detach()
            state_c=state_c.detach()
            
            logits,(state_h,state_c)=model(x, state_h, state_c)
           
            #compute loss
            loss = criterion(logits, y)
            loss_value = loss.item()
            total_loss+=len(x)*loss_value

            #Sets the gradients of all the optimized tensors to zero
            model.zero_grad()

            #computes dloss/dx and assigns gradient for every parameter
            loss.backward()

            #Clips the gradient norm to avoid exploding gradient problems
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            #Performs a single optimization step (parameter update).
            optimizer.step()
            
        total_loss/=len(pad_predictors)
            
        print("Epoch [{}/{}] Loss: {}, perplexity: {}".
              format(epoch+1, num_epochs, total_loss, np.exp(total_loss)))
        
        gen_text=generate(model, init='The', sent_len=100, topk=5)
        print("Text generated after epoch", epoch,":")
        print("\n",end='')
        print(gen_text)
        print('\n',end='')
        
    return model, total_loss


# # TEST THE MODEL

# In[9]:


def evaluate(model_path):
    """Evaluates the model on test data"""
    
    test_data=load_data("../input/wikitext2-data/test.txt")
    data=test_data[:]
    data=Clean_data(data)
    sentences, words=split_data(data, num_sentences=6000)
        
    pad_predictors, class_labels=Convert_data(sentences, words, seq_len)
    
    print("Number of input sequences: ",len(pad_predictors))
    
    #Load the saved model
    model=torch.load(model_path)
    
    #Locally disabling gradient computation
    with torch.no_grad():
        #sets model in evaluation mode
        model.eval()
        batch_size=200
        state_h, state_c=model.init_hidden(batch_size)
    
        state_h=state_h.detach()
        state_c=state_c.detach()
        
        total_loss=0
        for x,y in get_batch(pad_predictors, class_labels, batch_size):
            logits,(state_h,state_c)=model(x, state_h, state_c)

            #compute loss
            criterion=nn.CrossEntropyLoss()
            loss=criterion(logits, y)

            loss_value = loss.item()
            total_loss+=len(x)*loss_value
        total_loss/=len(pad_predictors)
        
        return total_loss, np.exp(total_loss)


# # GENERATE TEXT

# In[10]:


def generate(model, init, sent_len, topk):
    """Generates sentences from the model"""

    sentence=init
    for k in range(sent_len):
        #sets model in evaluation mode
        model.eval()
        
        #sets the length of sentence to seq_len
        input_indices=[]
        for word in str.split(sentence," "):
            input_indices.append(word_ind[word])
        if len(input_indices)<seq_len-1:
            input_tensor=[0]*(seq_len-len(input_indices)-1)
            input_tensor.extend(input_indices)
        else:
            input_tensor=input_indices[-seq_len+1:]
            
        #Initiates hidden state and cell state tensors to zeros
        state_h, state_c=model.init_hidden(len(input_tensor))
        
        input_tensor=torch.stack([torch.tensor(input_tensor)])
        out,(state_h,state_c)=model(input_tensor.transpose(0,1),state_h, state_c)
        
        #Samples a word from topk words
        word=model.topk_sampling(out, topk)
        
        if word!='' and word!=str.split(sentence,' ')[-1]:
            sentence=sentence+" "+word

    return sentence


# In[11]:


def main():
    
    train=load_data("../input/wikitext2-data/train.txt")
    data=train[:]
    data=Clean_data(data)
    sentences, words=split_data(data, num_sentences=25000)
        
    pad_predictors, class_labels=Convert_data(sentences, words, seq_len)
    
    print("Number of input sequences :",len(pad_predictors))
    
    model, loss=train_model(pad_predictors, class_labels, n_vocab=len(words), embedding_dim=100,
                padding_idx=0, hidden_size=128, Dropout_p=0.1, batch_size=200, lr=0.001)
    
    generated_sentence=generate(model, init='The', sent_len=100, topk=5)
    
    #save the model
    torch.save(model,"./Wiki_Model.pt")
    
    return loss


# In[12]:


if __name__ == "__main__":
    seq_len=6
    loss=main()
    
    print("Loss on train data: ", loss)
    print("Perplexity on train data: ", np.exp(loss))
    
    #Evaluating test data
    loss, perplexity=evaluate(model_path="./Wiki_Model.pt")
    print("Loss on test data: ", loss)
    print("Perplexity on test data: ", perplexity)

