import torch
import torch.nn as nn
from torch.autograd import Variable

from models.preembeddings import build_preembedding

class RNNLayer(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, direction=1):
    super(RNNLayer, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden

    self.i2h = nn.Linear(dim_input + dim_hidden, dim_hidden)
    self.i2o = nn.Linear(dim_input + dim_hidden, dim_output)
    self.direction = direction

  def forward(self, input, hidden):
    outputs = []
    if self.direction == 1:
      for i in range(input.size()[1]):
        combined = torch.cat((input[:, i, :], hidden), dim=1)
        hidden = self.i2h(combined)
        output_cell = self.i2o(combined)
        outputs.append(output_cell)
    else: 
      for i in range(input.size()[1]-1, -1, -1):
        combined = torch.cat((input[:, i, :], hidden), dim=1)
        hidden = self.i2h(combined)
        output_cell = self.i2o(combined)
        outputs.append(output_cell)
    return torch.stack(outputs, dim=1) # (batch_size, seq_len, dim_output)

  def init_hidden(self, batch_size):
    return torch.zeros(batch_size, self.dim_hidden)

class RNN(nn.Module):
  def __init__(self, vocab_size, dim_input, dim_hidden, dim_output, embedding_strategy='random', embedding_frozen=True, **kwargs):
    super(RNN, self).__init__()
    # Initialize the embedding using the factory function
    self.embedding_strategy = embedding_strategy
    if embedding_strategy == "empty": # TODO: for baseline only
      self.token_embedding = nn.Embedding(vocab_size, dim_input)
    else:
      self.token_embedding = build_preembedding(
          strategy=embedding_strategy,
          vocab_size=vocab_size,
          embedding_dim=dim_input,
          **kwargs
      )
    if embedding_frozen:
      self.token_embedding.weight.requires_grad = False
    # frozen embeddings option
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output

    self.rnn_layer = RNNLayer(dim_input, dim_hidden, dim_output)
    self.softmax = nn.LogSoftmax(dim=-1)
  
  def initialize(self):
    for p in self.parameters():
      # not initializing the embedding layer
      if p is not self.token_embedding.weight:
        nn.init.xavier_uniform_(p)
  
  def forward(self, input):
    hidden = self.rnn_layer.init_hidden(input.size()[0])
    embedded = self.token_embedding(input)
    outputs = self.rnn_layer(embedded, hidden)
    outputs = self.softmax(outputs)
    return outputs

  def init_hidden(self, batch_size):
    return Variable(torch.zeros((batch_size, self.dim_hidden)))