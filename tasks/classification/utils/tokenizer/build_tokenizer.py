from .BPE import BPETokenizer
from .word2vec import NLTKTokenizer

SUPPORTED_TOKENIZERS = {
    "bpe": BPETokenizer,
    "word2vec": NLTKTokenizer
}

def build_tokenizer(args):
    if "pretrained_path" in args["args"]:
        if args["tokenizer_type"] in SUPPORTED_TOKENIZERS:
            return SUPPORTED_TOKENIZERS[args["tokenizer_type"]].from_pretrained(args["args"]["pretrained_path"])
        else:
            raise ValueError(f"Tokenizer type {args['tokenizer_type']} not supported")
    else:
        print("Building tokenizer from scratch")
        if args["tokenizer_type"] in SUPPORTED_TOKENIZERS:
            tokenizer = SUPPORTED_TOKENIZERS[args["tokenizer_type"]](args["args"])
            tokenizer.build_vocab()
        else:
            raise ValueError(f"Tokenizer type {args['tokenizer_type']} not supported")
        