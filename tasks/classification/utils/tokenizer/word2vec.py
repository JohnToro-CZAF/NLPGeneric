# tokenizers/word2vec_tokenizer.py
import os
import json
from typing import Dict, Union, List
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from utils.tokenizer import BaseTokenizer  # Ensure this import points to your BaseTokenizer

class Word2VecTokenizer(BaseTokenizer):
    def __init__(self, name='word2vec_tokenizer', word2vec_path=None, special_tokens=None):
        """
        Initializes the Word2VecTokenizer.

        Args:
            name (str): Name of the tokenizer.
            word2vec_path (str): Path to the pretrained Word2Vec model file.
            special_tokens (List[str]): List of special tokens (e.g., PAD, UNK).
        """
        self.name = name
        self.word2vec_path = word2vec_path
        self.special_tokens = special_tokens if special_tokens is not None else ['<PAD>', '<UNK>']
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.embedding_dim = 0
        self.model = None  # Will hold the pretrained embeddings

        if word2vec_path:
            self.load_word2vec_model(word2vec_path)
            self.build_vocab_from_model()

    def load_word2vec_model(self, word2vec_path):
        """
        Loads the pretrained Word2Vec model.

        Args:
            word2vec_path (str): Path to the pretrained Word2Vec model file.
        """
        print(f"Loading Word2Vec model from: {word2vec_path}")
        # Load the model using gensim
        self.model = KeyedVectors.load_word2vec_format(word2vec_path, binary=word2vec_path.endswith('.bin'))
        self.embedding_dim = self.model.vector_size
        print(f"Loaded Word2Vec model with vector size: {self.embedding_dim}")

    def build_vocab_from_model(self):
        """
        Builds the vocabulary mappings from the pretrained Word2Vec model.
        """
        # Start indexing with special tokens
        idx = 0
        for token in self.special_tokens:
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
            idx += 1

        # Add words from the Word2Vec model to the vocabulary
        for word in self.model.index_to_key:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1

        self.vocab_size = len(self.word_to_idx)
        print(f"Vocabulary size (including special tokens): {self.vocab_size}")

    def save(self, folder_path: str) -> str:
        """
        Saves the tokenizer state to the specified folder.

        Args:
            folder_path (str): Path to the folder where the tokenizer state will be saved.

        Returns:
            str: Path to the folder where the tokenizer was saved.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # Save configuration
        config = {
            'name': self.name,
            'word2vec_path': self.word2vec_path,
            'special_tokens': self.special_tokens,
            'embedding_dim': self.embedding_dim
        }
        with open(os.path.join(folder_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        # Save vocabulary mappings
        state = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word
        }
        with open(os.path.join(folder_path, 'state.json'), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
        return folder_path

    @classmethod
    def from_pretrained(cls, folder_path: str):
        if os.path.isdir(folder_path):
            config_file = os.path.join(folder_path, "config.json")
            state_file = os.path.join(folder_path, "state.json")
            if os.path.exists(config_file):
                config_dict = json.load(open(config_file, "r", encoding='utf-8'))
                print("Loading tokenizer from cache: ", folder_path)
                print("Configuration: ", config_dict)
            else:
                raise Exception("Configuration file of tokenizer does not exist: ", config_file)
            if os.path.exists(state_file):
                state_dict = json.load(open(state_file, "r", encoding='utf-8'))
                tokenizer = cls(
                    name=config_dict["name"],
                    word2vec_path=config_dict["word2vec_path"],
                    special_tokens=config_dict["special_tokens"]
                )
                tokenizer.embedding_dim = config_dict["embedding_dim"]
                tokenizer.word_to_idx = state_dict["word_to_idx"]
                tokenizer.idx_to_word = {int(k): v for k, v in state_dict["idx_to_word"].items()}
                tokenizer.vocab_size = len(tokenizer.word_to_idx)
                # Load the Word2Vec model
                tokenizer.load_word2vec_model(tokenizer.word2vec_path)
                return tokenizer
            else:
                raise Exception("State dict of tokenizer does not exist: ", state_file)
        else:
            raise Exception("Folder path to tokenizer does not exist")

    def build_vocab(self):
        """
        Not required as the vocabulary is built from the pretrained Word2Vec model.
        """
        raise NotImplementedError("Vocabulary is built from the pretrained Word2Vec model.")

    def tokenize(self, text: str) -> Dict[str, Union[List[str], List[int]]]:
        """
        Tokenizes the input text into tokens and indices.

        Args:
            text (str): Input text string.

        Returns:
            Dict[str, Union[List[str], List[int]]]: Dictionary containing 'tokens' and 'indices'.
        """
        tokens = word_tokenize(text.lower())
        indices = []
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                indices.append(self.word_to_idx.get('<UNK>', 1))  # Default to index 1 for '<UNK>'
        return {'tokens': tokens, 'indices': indices}
