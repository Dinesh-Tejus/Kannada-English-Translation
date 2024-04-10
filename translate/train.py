import torch
import torch.nn as nn
import torch.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from tqdm import tqdm, trange


MAX_LENGTH = 20 # maximum length of sentences
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(0)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights



class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, index):
        return self.source_sentences[index], self.target_sentences[index]



# with open('data/train.en', 'r') as f:
#     english_sentences = f.readlines()

# eng_tokens = [word_tokenize(sent) for sent in english_sentences]

# with open('data/train.kn', 'r') as f:
#     kannada_sentences = f.readlines()
# x = []
# for i in kannada_sentences:
#   x.append(i.strip("\n"))
# kannada_sentences = x

# kan_tokens = [indic_tokenize.trivial_tokenize(sent) for sent in kannada_sentences]

# get tokens from pre-processed files
with open('data/eng_tokens.txt', 'r') as f:
    tokens = f.readlines()
eng_tokens = []
for x in trange(len(tokens), desc='get english tokens...'):
    eng_tokens.append(tokens[x].strip('\n').split(' '))
print(eng_tokens[0])


with open('data/kan_tokens.txt', 'r') as f:
    tokens = f.readlines()
kan_tokens = []
for x in trange(len(tokens), desc='get kannada tokens...'):
    kan_tokens.append(tokens[x].strip('\n').split(' '))
print(kan_tokens[0])

# get vocabulary
eng_vocab = set()
kan_vocab = set()
for i in eng_tokens:
  for j in i:
    eng_vocab.add(j)
eng_vocab = list(eng_vocab)

for i in kan_tokens:
  for j in i:
    kan_vocab.add(j)
kan_vocab = list(kan_vocab)

print(eng_vocab[:10])
print(kan_vocab[:10])

# get index lists
eng_word2index = {word: index for index, word in enumerate(eng_vocab)}
kan_word2index = {word: index for index, word in enumerate(kan_vocab)}

eng_indices = [[eng_word2index[word] for word in sent] for sent in eng_tokens] 
kan_indices = [[kan_word2index[word] for word in sent] for sent in kan_tokens]

batch_size = 32
dataset = TranslationDataset(kan_indices, eng_indices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




lr = 0.01
epochs = 2
hidden_size = 100
encoder = EncoderRNN(input_size=len(kan_vocab), hidden_size=hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size=hidden_size, output_size=len(eng_vocab)).to(device)
optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
criterion = nn.CrossEntropyLoss()

print("training begin.")

# Training loop
MODEL_SAVE_INTERVAL = 100 # save the model every so often
losses = [] # average loss per epoch
bar = trange(epochs, desc=f'')
for epoch in bar:
    epoch_loss = 0
    for i, (kan_batch,eng_batch) in enumerate(dataloader): # TO-DO - Need to pad the data
        eng_batch = eng_batch.to(device)
        kan_batch = kan_batch.to(device)

        optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(kan_batch)
        decoder_outputs, decoder_hidden, attentions = decoder(encoder_outputs, encoder_hidden, target_tensor=eng_batch)

        loss = criterion(decoder_outputs.view(-1, len(eng_vocab)), eng_batch.view(-1))
        epoch_loss += (loss.item()/len(eng_batch))
        loss.backward()
        optimizer.step()
    losses.append(epoch_loss)
    bar.set_description(f'loss: {epoch_loss}')

    if epoch % MODEL_SAVE_INTERVAL == 0:
        torch.save(encoder.state_dict(), f"encoder.pt")
        torch.save(decoder.state_dict(), f"decoder.pt")

torch.save(encoder.state_dict(), f"encoder_final.pt")
torch.save(decoder.state_dict(), f"decoder_final.pt")

