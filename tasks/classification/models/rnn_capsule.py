import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from .preembeddings import build_preembedding
# Adapted from https://github.com/thuwyq/WWW18-rnn-capsule/blob/master/attentionlayer.py
class Attention(nn.Module):
    def __init__(self, attention_size, return_attention=False, use_cuda=True):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction
        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.use_cuda = use_cuda
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.attention_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = Variable((idxes < input_lengths.unsqueeze(1)).float())
        mask = mask.cuda() if self.use_cuda else mask

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return (representations, attentions if self.return_attention else None)

class Capsule(nn.Module):
    def __init__(self, dim_vector, final_dropout_rate, use_cuda=True):
        super(Capsule, self).__init__()
        self.dim_vector = dim_vector
        self.add_module('linear_prob', nn.Linear(dim_vector, 1))
        self.add_module('final_dropout', nn.Dropout(final_dropout_rate))
        self.add_module('attention_layer', Attention(attention_size=dim_vector, return_attention=True, use_cuda=use_cuda))

    def forward(self, vect_instance, matrix_hidden_pad, len_hidden_pad=None):
        r_s, attention = self.attention_layer(matrix_hidden_pad, torch.LongTensor(len_hidden_pad))
        prob = F.sigmoid(self.linear_prob(self.final_dropout(r_s)))
        r_s_prob = prob * r_s
        return prob, r_s_prob


class EncoderRNN(nn.Module):
    def __init__(self,
            dim_input,
            dim_hidden,
            n_layers,
            n_label,
            tokenizer,
            embed_dropout_rate,
            cell_dropout_rate,
            final_dropout_rate,
            bidirectional,
            rnn_type,
            use_cuda=True,
            embedding_strategy='random', 
            embedding_frozen=True,):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.n_label = n_label
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.use_cuda = use_cuda
        
        if embedding_strategy == "empty": # TODO: for baseline only
            self.add_module('embed', nn.Embedding(tokenizer.get_vocab_size(), dim_input))
        else:
            self.add_module('embed', build_preembedding(
                strategy=embedding_strategy,
                tokenizer=tokenizer,
                embedding_dim=dim_input,
                **kwargs
            ))
        if embedding_frozen:
            try:
                self.embed.weight.requires_grad = False
            except:
                self.embed.embedding.weight.requires_grad = False
        self.add_module('embed_dropout', nn.Dropout(embed_dropout_rate))
        self.add_module('rnn', getattr(nn, self.rnn_type)(dim_input, dim_hidden, n_layers, batch_first=True, 
                dropout=cell_dropout_rate, bidirectional=bidirectional,))

        for i in range(self.n_label):
            self.add_module('capsule_%s' % i, Capsule(dim_hidden * (2 if self.bidirectional else 1), final_dropout_rate, self.use_cuda))

        # self.init_weights(embed_list)
        # ignored_params = list(map(id, self.embed.parameters()))
        self.base_params = filter(lambda p: id(p) not in ignored_params,
                     self.parameters())

    # def init_weights(self, embed_list):
        # self.embed.weight.data.copy_(torch.from_numpy(embed_list))

    def forward(self, input):

        hidden = self.init_hidden(input.size()[0])
        embedded = self.embed(input)
        embedded = self.embed_dropout(embedded)
        input_packed = pack_padded_sequence(embedded, lengths=lengths, batch_first=True)
        output, hidden = self.rnn(input_packed, hidden)
        output_pad, output_len = pad_packed_sequence(output, batch_first=True)

        variable_len = Variable(torch.from_numpy(1.0/lengths.astype(np.float32))).unsqueeze(-1)
        v_s = torch.sum(output_pad, 1) * (variable_len.cuda() if self.use_cuda else variable_len)
        list_prob, list_r_s = [], []
        for i in range(self.n_label):
            prob_tmp, r_s_tmp = getattr(self, 'capsule_%s' % i)(v_s, output_pad, torch.LongTensor(output_len))
            list_prob.append(prob_tmp)
            list_r_s.append(r_s_tmp)

        list_r_s = torch.stack(list_r_s)
        list_sim = torch.sum(v_s*list_r_s, 2).t()
        prob = torch.stack(list_prob).squeeze(-1).t()   

        return list_sim, prob

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        h_0 = Variable(weight.new(self.n_layers * (2 if self.bidirectional else 1), batch_size, self.dim_hidden).zero_(), requires_grad=False)
        h_0 = h_0.cuda() if self.use_cuda else h_0
        return (h_0, h_0) if self.rnn_type == "LSTM" else h_0



class rnnCapsule(object):
    def __init__(self,
            dim_input,
            dim_hidden,
            n_layers,
            n_label,
            batch_size,
            max_length,
            learning_rate,
            lr_word_vector=0.01,
            weight_decay=0,
            vocab=None,
            embed=None,
            embed_dropout_rate=0.,
            cell_dropout_rate=0.,
            final_dropout_rate=0.,
            bidirectional=True,
            optim_type="Adam",
            rnn_type="LSTM",
            use_cuda=True):
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.lang = Lang(vocab)
        self.model = EncoderRNN(dim_input, dim_hidden, n_layers, n_label, len(vocab), 
                embed_dropout_rate, cell_dropout_rate, final_dropout_rate, embed, bidirectional, rnn_type, use_cuda)
        if self.use_cuda:
            self.model.cuda()
        self.optimizer = getattr(optim, optim_type)([
                                        {'params': self.model.base_params},
                                        {'params': self.model.embed.parameters(), 'lr': lr_word_vector, 'weight_decay': 0}, 
                                    ], lr=self.learning_rate, weight_decay=weight_decay)
        self.encoder_hidden = self.model.init_hidden(self.batch_size)

    def get_batch_data(self, batched_data):
        input_sentence, tensor_label, sen_len = batched_data['sentence'], batched_data['labels'], batched_data['sentence_length']
        input_variable = self.lang.VariablesFromSentences(input_sentence, True, self.use_cuda)
        tensor_label = Variable(torch.from_numpy(batched_data['labels']))
        tensor_label = tensor_label.cuda() if self.use_cuda else tensor_label
        return input_variable, tensor_label, sen_len

    def stepTrain(self, batched_data, inference=False):
        # Turn on training mode which enables dropout.
        self.model.eval() if inference else self.model.train()
        input_variable, tensor_label, sen_len = self.get_batch_data(batched_data)
        hidden = self.model.init_hidden(len(batched_data['sentence_length']))
        
        if inference == False:
            # zero the parameter gradients
            self.optimizer.zero_grad()

        loss_sim, prob = self.model(input_variable, hidden, sen_len, tensor_label)
        loss_hinge_classify = F.multi_margin_loss(prob, tensor_label)
        loss_hinge = F.multi_margin_loss(loss_sim, tensor_label)
        loss = loss_hinge_classify + loss_hinge

        if inference == False:
            loss.backward()
            self.optimizer.step()
        
        return np.array([loss.data.cpu().numpy(), loss_hinge_classify.data.cpu().numpy(), loss_hinge.data.cpu().numpy()]).reshape(3), prob.data.cpu().numpy()

    def save_model(self, dir, idx):
        os.mkdir(dir) if not os.path.isdir(dir) else None
        torch.save(self, '%s/model%s.pkl' % (dir, idx))