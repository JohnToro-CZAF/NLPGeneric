# models/preembedding.py
import numpy as np
import fasttext
from gensim.models import Word2Vec

import torch
import torch.nn as nn

class PreEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(PreEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def forward(self, input_ids):
        raise NotImplementedError("Each embedding strategy must implement the forward method.")

class RandomInitEmbedding(PreEmbedding):
    def __init__(self, vocab_size, embedding_dim):
        super(RandomInitEmbedding, self).__init__(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids):
        return self.embedding(input_ids)
    

class GloveEmbedding(PreEmbedding):
    def __init__(self, vocab_size, embedding_dim, word_to_idx, glove_path):
        super(GloveEmbedding, self).__init__(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.load_glove_embeddings(word_to_idx, glove_path)

    def load_glove_embeddings(self, word_to_idx, glove_path):
        embeddings_index = {}
        with open(glove_path, 'r', encoding='utf8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                try:
                    vector = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = vector
                except ValueError:
                    continue  # Skip lines with formatting issues

        # Initialize embedding weights
        embedding_weights = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_dim))
        for word, idx in word_to_idx.items():
            if word in embeddings_index:
                embedding_weights[idx] = embeddings_index[word]

        self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))

    def forward(self, input_ids):
        return self.embedding(input_ids)


class FastTextEmbedding(PreEmbedding):
    def __init__(self, vocab_size, embedding_dim, word_to_idx, fasttext_path):
        super(FastTextEmbedding, self).__init__(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.load_fasttext_embeddings(word_to_idx, fasttext_path)

    def load_fasttext_embeddings(self, word_to_idx, fasttext_path):
        ft_model = fasttext.load_model(fasttext_path)
        embedding_weights = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_dim))
        for word, idx in word_to_idx.items():
            embedding_weights[idx] = ft_model.get_word_vector(word)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))

    def forward(self, input_ids):
        return self.embedding(input_ids)
    
class ContextAverageEmbedding(PreEmbedding):
    """
    Represent OOV words by averaging embeddings of known words in the context.
    """
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super(ContextAverageEmbedding, self).__init__(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        embeddings = self.embedding(input_ids)
        mask = (input_ids != self.padding_idx).float().unsqueeze(-1)
        sum_embeddings = torch.sum(embeddings * mask, dim=1)
        count_non_pad = torch.sum(mask, dim=1)
        avg_embeddings = sum_embeddings / (count_non_pad + 1e-8)
        avg_embeddings = avg_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        # Replace embeddings for OOV words with average embeddings
        oov_mask = (input_ids >= self.vocab_size).unsqueeze(-1)
        embeddings = embeddings * (~oov_mask) + avg_embeddings * oov_mask
        return embeddings

class MorphologicalEmbedding(PreEmbedding):
    """
    Decompose OOV words into subwords and average their embeddings.
    """
    def __init__(self, vocab_size, embedding_dim, word_to_idx, subword_vocab):
        super(MorphologicalEmbedding, self).__init__(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.subword_embedding = nn.Embedding(len(subword_vocab), embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        nn.init.uniform_(self.subword_embedding.weight, -0.25, 0.25)
        self.word_to_idx = word_to_idx
        self.subword_vocab = subword_vocab
        self.subword_to_idx = {subword: idx for idx, subword in enumerate(subword_vocab)}

    def decompose_word(self, word):
        # Simple example using prefixes and suffixes
        prefixes = ['un', 're', 'in', 'im', 'dis']
        suffixes = ['ing', 'ed', 'ly', 's', 'es']
        subwords = []
        for prefix in prefixes:
            if word.startswith(prefix):
                subwords.append(prefix)
                word = word[len(prefix):]
                break
        for suffix in suffixes:
            if word.endswith(suffix):
                subwords.append(suffix)
                word = word[:-len(suffix)]
                break
        if word:
            subwords.append(word)
        return subwords

    def forward(self, input_ids):
        embeddings = []
        for idx in input_ids.view(-1):
            if idx.item() < self.vocab_size:
                embeddings.append(self.embedding(idx))
            else:
                # OOV word handling
                word = self.idx_to_word(idx.item())
                subwords = self.decompose_word(word)
                subword_indices = [self.subword_to_idx.get(sw, 0) for sw in subwords]
                subword_embeddings = self.subword_embedding(torch.tensor(subword_indices))
                avg_embedding = torch.mean(subword_embeddings, dim=0)
                embeddings.append(avg_embedding)
        embeddings = torch.stack(embeddings).view(input_ids.size(0), input_ids.size(1), -1)
        return embeddings

    def idx_to_word(self, idx):
        # Assuming you have a reverse mapping from index to word
        return self.idx_to_word_map.get(idx, '<UNK>')


class Word2VecEmbedding(PreEmbedding):
    """
    Train domain-specific embeddings and use them.
    """
    def __init__(self, vocab_size, embedding_dim, word_to_idx, domain_corpus):
        super(Word2VecEmbedding, self).__init__(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.train_domain_embeddings(word_to_idx, domain_corpus)

    def train_domain_embeddings(self, word_to_idx, domain_corpus):
        # Assuming domain_corpus is a list of tokenized sentences
        model = Word2Vec(sentences=domain_corpus, vector_size=self.embedding_dim, window=5, min_count=1, workers=4)
        embedding_weights = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_dim))
        for word, idx in word_to_idx.items():
            if word in model.wv:
                embedding_weights[idx] = model.wv[word]
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))

    def forward(self, input_ids):
        return self.embedding(input_ids)
    
class CharLevelEmbedding(PreEmbedding):
    """
    Use character-level embeddings to represent words.
    """
    def __init__(self, vocab_size, embedding_dim, char_vocab_size, max_word_length):
        super(CharLevelEmbedding, self).__init__(vocab_size, embedding_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.max_word_length = max_word_length

    def forward(self, char_input_ids):
        # char_input_ids: [batch_size, seq_len, max_word_length]
        batch_size, seq_len, max_word_length = char_input_ids.size()
        char_embeddings = self.char_embedding(char_input_ids.view(-1, max_word_length))
        char_embeddings = char_embeddings.transpose(1, 2)  # [batch_size*seq_len, embedding_dim, max_word_length]
        conv_out = self.conv(char_embeddings)
        pooled = self.pool(conv_out).squeeze(-1)
        word_embeddings = pooled.view(batch_size, seq_len, -1)
        return word_embeddings

def build_preembedding(strategy, vocab_size, embedding_dim, **kwargs):
    if strategy == 'random':
        return RandomInitEmbedding(vocab_size, embedding_dim)
    elif strategy == 'fasttext':
        word_to_idx = kwargs['word_to_idx']
        fasttext_path = kwargs['fasttext_path']
        return FastTextEmbedding(vocab_size, embedding_dim, word_to_idx, fasttext_path)
    elif strategy == 'context_average':
        padding_idx = kwargs.get('padding_idx', 0)
        return ContextAverageEmbedding(vocab_size, embedding_dim, padding_idx)
    elif strategy == 'morphological':
        word_to_idx = kwargs['word_to_idx']
        subword_vocab = kwargs['subword_vocab']
        return MorphologicalEmbedding(vocab_size, embedding_dim, word_to_idx, subword_vocab)
    elif strategy == 'domain_specific':
        word_to_idx = kwargs['word_to_idx']
        domain_corpus = kwargs['domain_corpus']
        return Word2VecEmbedding(vocab_size, embedding_dim, word_to_idx, domain_corpus)
    elif strategy == 'char_level':
        char_vocab_size = kwargs['char_vocab_size']
        max_word_length = kwargs['max_word_length']
        return CharLevelEmbedding(vocab_size, embedding_dim, char_vocab_size, max_word_length)
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy}")
