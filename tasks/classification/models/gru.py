import torch
import torch.nn as nn
from .preembeddings import build_preembedding

class GRUSubLayer(nn.Module):
    def __init__(self, dim_input, dim_hidden, direction=1):
        super(GRUSubLayer, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.direction = direction

        # GRU gates
        self.update_gate = nn.Linear(dim_input + dim_hidden, dim_hidden)
        self.reset_gate = nn.Linear(dim_input + dim_hidden, dim_hidden)
        self.new_gate = nn.Linear(dim_input + dim_hidden, dim_hidden)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input_seq, hidden_state):
        seq_len = input_seq.size(1)
        batch_size = input_seq.size(0)
        H = hidden_state  # Hidden state

        outputs = []

        indices = range(seq_len) if self.direction == 1 else range(seq_len - 1, -1, -1)
        for i in indices:
            combined = torch.cat((input_seq[:, i, :], H), dim=1)
            Z = self.sigmoid(self.update_gate(combined))
            R = self.sigmoid(self.reset_gate(combined))
            combined_new = torch.cat((input_seq[:, i, :], R * H), dim=1)
            H_tilde = self.tanh(self.new_gate(combined_new))
            H = (1 - Z) * H + Z * H_tilde
            outputs.append(H.unsqueeze(1))  # Shape: [batch_size, 1, dim_hidden]

        if self.direction == -1:
            outputs.reverse()  # Reverse outputs if processing backward

        outputs = torch.cat(outputs, dim=1)  # Shape: [batch_size, seq_len, dim_hidden]
        return outputs, H  # Return outputs and the last hidden state

    def init_hidden(self, batch_size, device):
        H = torch.zeros(batch_size, self.dim_hidden).to(device)
        return H

class MultilayerGRU(nn.Module):
    def __init__(self, vocab_size, dim_input, dim_hidden, dim_output, num_layers=1, embedding_strategy='random', embedding_frozen=True, **kwargs):
        super(MultilayerGRU, self).__init__()
        self.embedding_strategy = embedding_strategy

        # Initialize the embedding layer
        if embedding_strategy == "empty":  # For baseline only
            self.token_embedding = nn.Embedding(vocab_size, dim_input)
        else:
            self.token_embedding = build_preembedding(
                strategy=embedding_strategy,
                vocab_size=vocab_size,
                embedding_dim=dim_input,
                **kwargs
            )
            
        if embedding_frozen:
            try:
                self.token_embedding.weight.requires_grad = False
            except:
                self.token_embedding.embedding.weight.requires_grad = False

        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_layers = num_layers

        # Stack GRU layers
        self.gru_layers = nn.ModuleList()
        for layer in range(num_layers):
            input_size = dim_input if layer == 0 else dim_hidden
            self.gru_layers.append(GRUSubLayer(input_size, dim_hidden))

        # Output layer
        self.fc = nn.Linear(dim_hidden, dim_output)
        self.softmax = nn.LogSoftmax(dim=-1)

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad and p is not self.token_embedding.weight:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        batch_size = input.size(0)
        device = input.device

        # Embedding layer
        embedded = self.token_embedding(input)  # Shape: [batch_size, seq_len, dim_input]

        # Initialize hidden states for all layers
        hidden_states = [layer.init_hidden(batch_size, device) for layer in self.gru_layers]

        # Pass through stacked GRU layers
        input_seq = embedded
        for idx, layer in enumerate(self.gru_layers):
            outputs, hidden_states[idx] = layer(input_seq, hidden_states[idx])
            input_seq = outputs  # Input to the next layer

        # Apply the output layer to the last layer's outputs
        logits = self.fc(outputs)  # Shape: [batch_size, seq_len, dim_output]
        preds = self.softmax(logits)
        return preds
    
class BiGRUSubLayer(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(BiGRUSubLayer, self).__init__()
        self.forward_layer = GRUSubLayer(dim_input, dim_hidden, direction=1)
        self.backward_layer = GRUSubLayer(dim_input, dim_hidden, direction=-1)

    def forward(self, input_seq, hidden_state):
        batch_size = input_seq.size(0)
        device = input_seq.device

        # Forward direction
        H_fwd = hidden_state[0]
        outputs_fwd, H_fwd = self.forward_layer(input_seq, H_fwd)

        # Backward direction
        H_bwd = hidden_state[1]
        outputs_bwd, H_bwd = self.backward_layer(input_seq, H_bwd)

        # Concatenate the outputs
        outputs = torch.cat((outputs_fwd, outputs_bwd), dim=2)  # Shape: [batch_size, seq_len, 2 * dim_hidden]

        # Update hidden states
        hidden_state = (H_fwd, H_bwd)

        return outputs, hidden_state

    def init_hidden(self, batch_size, device):
        H_fwd = self.forward_layer.init_hidden(batch_size, device)
        H_bwd = self.backward_layer.init_hidden(batch_size, device)
        return (H_fwd, H_bwd)

class MultilayerBiGRU(nn.Module):
    def __init__(self, vocab_size, dim_input, dim_hidden, dim_output, num_layers=1, embedding_strategy='random', embedding_frozen=True, **kwargs):
        super(MultilayerBiGRU, self).__init__()
        self.embedding_strategy = embedding_strategy

        # Initialize the embedding layer
        if embedding_strategy == "empty":  # For baseline only
            self.token_embedding = nn.Embedding(vocab_size, dim_input)
        else:
            self.token_embedding = build_preembedding(
                strategy=embedding_strategy,
                vocab_size=vocab_size,
                embedding_dim=dim_input,
                **kwargs
            )
        
        if embedding_frozen:
            try:
                self.token_embedding.weight.requires_grad = False
            except:
                self.token_embedding.embedding.weight.requires_grad = False

        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_layers = num_layers

        # Stack BiGRU layers
        self.gru_layers = nn.ModuleList()
        for layer in range(num_layers):
            input_size = dim_input if layer == 0 else dim_hidden * 2
            self.gru_layers.append(BiGRUSubLayer(input_size, dim_hidden))

        # Output layer
        self.fc = nn.Linear(dim_hidden * 2, dim_output)
        self.softmax = nn.LogSoftmax(dim=-1)

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad and p is not self.token_embedding.weight:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        batch_size = input.size(0)
        device = input.device

        # Embedding layer
        embedded = self.token_embedding(input)  # Shape: [batch_size, seq_len, dim_input]

        # Initialize hidden states for all layers
        hidden_states = [layer.init_hidden(batch_size, device) for layer in self.gru_layers]

        # Pass through stacked BiGRU layers
        input_seq = embedded
        for idx, layer in enumerate(self.gru_layers):
            outputs, hidden_states[idx] = layer(input_seq, hidden_states[idx])
            input_seq = outputs  # Input to the next layer

        # Apply the output layer to the last layer's outputs
        logits = self.fc(outputs)  # Shape: [batch_size, seq_len, dim_output]
        preds = self.softmax(logits)
        return preds
