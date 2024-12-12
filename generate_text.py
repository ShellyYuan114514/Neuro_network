
## To use the trained NET to generate text"

## Key modoule

import torch as tc
import torch.nn as nn

data = "the_dataset_file_you_used_to_train_you_net"

with open (data,'r') as file:
    text = file.read()

chars = sorted(list(set(text)))
char_size = len(chars)

char2idx = {ch:i for i, ch in enumerate(chars)}
idx2char= {i:ch for i, ch in enumerate(chars)}

## Define a NET with the same structure 

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

# you can use ".to(device)" to assign CUDA or MPS

## loding state dict
model_path = 'where_you_save_you_state_dict'
model.load_state_dict(tc.load(model_path,weights_only=True))

def generate_text(model, start_seq, gen_length=100):
    model.eval()
    generated_text = start_seq
    
    input_seq = [char2idx[ch] for ch in start_seq]
    input_tensor = tc.tensor(input_seq).unsqueeze(0)
    print
    with tc.no_grad():
        for _ in range(gen_length):
            output = model(input_tensor)
            _, top_idx = tc.topk(output, 1)
            predicted_char = idx2char[top_idx.item()]
            generated_text += predicted_char
            
            # 更新输入序列
            input_tensor = tc.cat((input_tensor, top_idx), dim=1)
            input_tensor = input_tensor[:, -100:]  # 只保留最后的序列
    
    return generated_text


start_sequence = "挽狂澜于既倒"
generated = generate_text(model, start_sequence)
print(generated)

