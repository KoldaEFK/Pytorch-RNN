import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#MODEL

"""
note: I often get confused by the dimensions of data going through the neural net; it might be caused by the fact that Pytorch build in neworks
(nn.RNN) always expect batch_size to be one dimension of the data
"""

class RNN(nn.Module):
    def __init__(self,n_features,n_layers,hidden_size,output_size,sequence_length):
        super(RNN,self).__init__()
        self.sequence_length = sequence_length
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(n_features, hidden_size, n_layers, batch_first=True) #expects input of shape --> batch_size, sequence_length, n_features
        self.fc = nn.Linear(hidden_size, output_size) #fully connected layer
    
    def forward(self,x):
        """forward pass of the data during training"""
        #do not confuse HIDDEN state with WEIGHT matrix!
        h0 = torch.zeros(self.n_layers,x.size(0),self.hidden_size).to(device) #h0 shape --> n_layers, batch_size, hidden_size
        out, _ = self.rnn(x, h0) #returns hidden states for all time steps
        out = out[:,-1,:] #I need only the last hidden state (many to one RNN, see README.md)
        out = self.fc(out) #out shape --> batch_size, output_size
        return out
    
    #function predict accepts know_sequence as first input and future_sequence that must DIRECTLY follow the known_sequence
    def predict(self,known_sequence, future_sequence):
        """function for predicting future values for arbitrary number of future time steps"""
        with torch.no_grad():
            self.eval() #setting the model to evaluation mode and turning off the gradient computation

            if len(known_sequence)<len(self.sequence_length):
                return "known_sequence must be longer than sequence length"
            
            known_sequence = known_sequence.to(device)
            future_sequence = future_sequence.to(device) #future sequence can be of arbitrary length
            
            #x is going to be updated at each future time step
            x = known_sequence[-self.sequence_length:]
            x = torch.reshape(x,(1,-1,11)) #x must be of shape --> batch_size, sequence_length, n_features = 1, 100, 11
            
            outputs = [] #list where outputs of future timesteps will be stored
            
            for i in range(len(future_sequence)): #future_sequence shape --> future_sequence_length, n_features
                """calculating outputs and updating the x with the newly predicted target variables"""

                #more or less repeating forward pass with batch_size = 1
                h0 = torch.zeros(self.n_layers,1,self.hidden_size).to(device)
                out, _ = self.rnn(x, h0) #out shape --> 1,batch_size,hidden_size
                out = out[:,-1,:] #out shape --> 1,hidden_size
                out = self.fc(out) #out shape --> 1,output_size = 1,3
                out = out[0] #out shape --> output_size = 3
                
                outputs.append(out)
                
                #preparing the new data point new_t to be added to x
                new_x = future_sequence[i,:8] #known data of the future time step
                new_t = torch.cat((new_x,out),-1) #concatenating the 8 known features with 3 predicted

                x = torch.roll(x,-1,1) #shifting the elements in the tensor by -1 (the first element becomes last)
                x[0,-1] = new_t #replacing the last element with the newly made data point
        
            return outputs