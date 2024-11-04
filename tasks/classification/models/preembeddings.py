# models/preembedding.py
import numpy as np
import fasttext
from gensim.models import Word2Vec
import gensim.downloader as api

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

class Word2VecEmbedding(nn.Module):
    """
    Train domain-specific embeddings and use them.
    """
    def __init__(self, vocab_size, embedding_dim, pretrained_path, oov_handling):
        super(Word2VecEmbedding, self).__init__()
        word2vec_model = api.load(pretrained_path)
        pretrained_embedding = word2vec_model.vectors 
        vocab = word2vec_model.key_to_index
        # Handling OOV words, the model accepts [0, max_vocab + 1], while the embeddings are [0, max_vocab-1]
        self.unk_id = len(vocab.keys())
        self.pad_id = len(vocab.keys()) + 1
        self.oov_handling = oov_handling
        print("Pretrained embedding size: ", pretrained_embedding.shape)
        pretrained_embedding = np.append(pretrained_embedding, [np.random.rand(*pretrained_embedding[-1].shape)], axis=0) # unk_id
        pretrained_embedding = np.append(pretrained_embedding, [np.zeros_like(pretrained_embedding[-1])], axis=0) # pad_id
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(pretrained_embedding, dtype=torch.float),
        )
        assert vocab_size == len(vocab.keys()) + 2, f"Vocab size mismatch: {vocab_size} != {len(vocab.keys()) + 2}"
        assert embedding_dim == pretrained_embedding.shape[1], f"Embedding dim mismatch: {embedding_dim} != {pretrained_embedding.shape[1]}"

    def forward(self, input_ids):
        # Handling UNK token id and pad_id
        embeddings = self.embedding(input_ids)
        if self.oov_handling == "using_unk":
            return embeddings
        elif self.oov_handling == "average":
            mask = (input_ids != self.unk_id) & (input_ids != self.pad_id)
            mask = mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(embeddings * mask, dim=1)
            count_non_pad = torch.sum(mask, dim=1)
            avg_embeddings = sum_embeddings / (count_non_pad + 1e-8)
            avg_embeddings = avg_embeddings.unsqueeze(1).expand(-1, input_ids.size(1), -1)
            embeddings = embeddings * mask + avg_embeddings * (1-mask)
            return embeddings
        else:
            raise ValueError(f"Unknown OOV handling strategy: {self.oov_handling}")
    

def build_preembedding(strategy, vocab_size, embedding_dim, **kwargs):
    if strategy == 'random':
        return RandomInitEmbedding(vocab_size, embedding_dim)
    elif strategy == 'word2vec':
        pretrained_path = kwargs['pretrained_path']
        oov_handling = kwargs.get('oov_handling', 'average')
        return Word2VecEmbedding(vocab_size, embedding_dim, pretrained_path, oov_handling)
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy}")
