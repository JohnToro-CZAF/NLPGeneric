python train.py --config configs/embeddings_word2vec_oov=using_unk_unfrozen.json
python train.py --config configs/embeddings_word2vec_oov=average_unfrozen.json
python train.py --config configs/embeddings_word2vec_oov=context_average_unfrozen.json

python train.py --config configs/embeddings_word2vec_oov=using_unk.json
python train.py --config configs/embeddings_word2vec_oov=average.json
python train.py --config configs/embeddings_word2vec_oov=context_average.json