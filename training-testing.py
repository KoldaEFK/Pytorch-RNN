import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from model import RNN

#DEVICE CONFIG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#HYPERPARAMETERS
sequence_length = 100 #number of steps we are looking back
batch_size = 40
learning_rate = 0.0001
input_size = 11 #the number of features
hidden_size = 64 #size of hidden layer in RNN
output_size = 3 #the targets
num_epochs = 20
n_layers = 3

#DATA
df = pd.read_csv("train.csv") #train.csv contains 7111 logs
df_train = df.drop("date_time",1)[:5000] #df_train is shape 5000,11
scaler = StandardScaler() 
scaler = scaler.fit(df_train) #computing the parameters for scaling | converts df to numpy array
df_train = scaler.transform(df_train) 

df_test = df.drop("date_time",1)[5000:]
df_test = scaler.transform(df_test)

#RESHAPING the problem from TIME SERIES to SUPERVISED LEARNING
#X shape --> (length-sequence_length),sequence_length,features 4900,300,11
#y shape --> (length-sequence_length),targets 4900,11
class Train_Time_Series_Dataset(Dataset):
    """creates a dataset for supervised learning from data series"""
    def __init__(self, complete_data,sequence_length):
        X, y = list(), list()
        for i in range(len(complete_data)):
            end_idx = i + sequence_length
            if end_idx > len(complete_data) - 1:
                break
                
            #seq_x shape --> sequence_length, features (for time steps up to t-1)
            #seq_y shape --> targets (for time step t)
            seq_x, seq_y = complete_data[i:end_idx,:], complete_data[end_idx,8:]
            X.append(seq_x)
            y.append(seq_y)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        self.X = torch.from_numpy(X) #converting to tensors
        self.y = torch.from_numpy(y)
        self.sequence_length = sequence_length
        self.n_samples = X.shape[0]
        
    def __getitem__(self,index):
        return self.X[index], self.y[index]
        
    def __len__(self):
        return self.n_samples

#Eval_Dataset is basically used mainly to convert a numpy arrays to tensors and return the data of shape --> "len of input data frame",features
#It is not different data than Train_Dataset, only different format used for evaluation not training
class Eval_Dataset(Dataset):
    """creates a dataset for evaluation"""
    def __init__(self,sequence):
        sequence = np.asarray(sequence, dtype=np.float32)
        self.X = torch.from_numpy(sequence) #converts to tensor
        self.n_samples = sequence.shape[0]
    
    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.n_samples

dataset = Train_Time_Series_Dataset(df_train, sequence_length)
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True) #train_loader is used to supply data in shuffled BATCHES during training

eval_dataset = Eval_Dataset(df_train) #used for training

test_dataset = Eval_Dataset(df_test) #not used for training

#MODEL    
model = RNN(input_size,n_layers,hidden_size,output_size,sequence_length).to(device)

#loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#TRAINING LOOP
n_total_steps = len(train_loader)

running_loss = 0.0

model.train() #setting model to training mode

for epoch in range(num_epochs):
    for i, (features,targets) in enumerate(train_loader):
        #features shape --> batch_size, sequence_length, features
        #targets shape --> batch_size, targets
        features = features.to(device)
        targets = targets.to(device)
        
        #forward
        outputs = model(features)
        loss = criterion(outputs, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+= loss.item() #item returns a number

        if (i+1) % 50 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item()}')
            running_loss = 0.0


#torch.save(model.state_dict(),"mymodel.pth")

#EVALUATION USING GRAPHS
#serves for visually assesing the performance of the model
test_outputs = model.predict(test_dataset[100:600],test_dataset[600:800]) #getting the outputs for data in eval_dataset in second argument of the function predict
test_outputs = torch.stack(test_outputs)
test_outputs = test_outputs.cpu().detach().numpy()

evv = test_dataset[600:800,8:] #the true data of those data points; not used in the "predict" function

#test_outputs = scaler.inverse_transform(test_outputs)
#evv = scaler.inverse_transform(evv)

plt.plot(test_outputs[:,0], label='forecast target 1')
plt.plot(test_outputs[:,1], label='forecast target 2')
plt.plot(test_outputs[:,2], label='forecast target 3')
plt.plot(evv[:,0], label='real target 1')
plt.plot(evv[:,1], label='real target 2')
plt.plot(evv[:,2], label='real target 3')
plt.legend()
plt.show()