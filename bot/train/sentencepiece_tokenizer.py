import sentencepiece as spm


vocab_size = 40
corpus_file = "data/germ/text/sentencepiece_corpus.txt"
sample_text = """
This is an example sentence.
Transformers have revolutionized NLP.
They use the attention mechanism to handle long-range dependencies.
SentencePiece tokenization is language-agnostic and handles subwords effectively.
"""

with open(corpus_file, "w", encoding="utf-8") as f:
    f.write(sample_text)


training_args = [
    f"--input={corpus_file}",
    f"--model_prefix=models/germ/tokenizer/germ_sentencepiece",
    f"--vocab_size={vocab_size}",
    "--model_type=unigram",
]

spm.SentencePieceTrainer.Train(" ".join(training_args))
