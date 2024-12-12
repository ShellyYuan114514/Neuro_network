
## To train a LSTM NET according to your dataset ##

## key modoules

import os
import torch as tc
import torch.nn as nn

## Use MPS framework to do the calculation 

# this is for macOS users, If you use nvidia, you can use CUDA

print(f"MPS is available : {tc.backends.mps.is_available()}")

# print(f"MPS is available : {tc.backends.cuda.is_available()}") if you use CUDA

device = tc.device("mps")

# device = tc.device("cuda") for CUDA users

## To import your data

dataset = "input_your_dataset_file_path"

with open (dataset,'r') as file:
    cont= file.read()

chars = sorted(list(set(cont)))
char_size = len(chars)

char2idx = {ch:i for i, ch in enumerate(chars)}
idx2char = {i:ch for i, ch in enumerate(chars)}

# mapping an dixtionary

seq_len=100 # you can alter this 

X=[]
Y=[]

for i in range (len(cont)-seq_len):
    seq_in = cont[i:i+seq_len]
    seq_out = cont[i+seq_len]
    X.append([char2idx[ch] for ch in seq_in])
    Y.append(char2idx[seq_out])

X = tc.tensor(X, dtype=tc.long).to(device)
Y = tc.tensor(Y, dtype=tc.long).to(device)
print(Y.shape)
print(X.shape)

# tensorize X and Y, you can alter the ".to(device)"

class textgen(nn.Module):
    def __init__ (self, char_size, embed_size,hidden_size):
        super(textgen,self).__init__()
        self.embedding = nn.Embedding(char_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,char_size)
    
    def forward(self,x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:,-1,:])
        
        return out
        
model = textgen(char_size=char_size,embed_size=128,hidden_size=256)
model = model.to(device)
print(model)

cri = nn.CrossEntropyLoss()
opt = tc.optim.Adam(model.parameters(),lr=0.001)

## to train your net

num_epochs = 100 # the times you want the NET trained
for epoch in range(num_epochs):
    model.train()
    opt.zero_grad()
    output = model(X)
    loss = cri(output,Y)
    loss.backward()
    opt.step()

    if (epoch+1)%10 ==0:
        print (f'Epoch[{epoch+1}/{num_epochs}],Loss:{loss.item():.4f}')

# save the state dict so you don't have to train it again later you use it 

print("Training finished")
save_dir = '/Users/zitingyuan/Desktop/swearfun/testfile'

model_filename = 'where_you_want_to_save_it'

save_path = os.path.join(save_dir, model_filename)
 
tc.save(model.state_dict(), save_path)
 
print(f'Model saved to {save_path}')




