from datasets import load_dataset
import argparse
import logging
import os
import random
import re
import sentencepiece as spm

from observability.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


arg_parser = argparse.ArgumentParser(description="Train a sentencepiece tokenization model.")
arg_parser.add_argument("--train", action="store_true", default=False,
                        help="Train a sentencepiece tokenization model.")
arg_parser.add_argument("--wikipedia", action="store_true", default=False,
                        help="Use wikipedia dataset.")
args = arg_parser.parse_args()

input_sentence_size = 3_000_000
max_line_char_len = 4192
vocab_size = 1_000_000

corpus_dir = "data/germ/text"
corpus_file_prefix = f"{corpus_dir}/sentencepiece_corpus"
model_file_prefix = "models/germ/tokenizer/germ_sentencepiece"
uber_chunk_file = f"{corpus_dir}/wikipedia_uber_chunks.txt"
white_space_pattern = re.compile(r"\s+")

if args.wikipedia:
    wikipedia_dataset_name = "20231101.en"
    wikipedia_dataset = load_dataset("wikimedia/wikipedia", wikipedia_dataset_name)
    total_page_cnt = len(wikipedia_dataset["train"])
    logger.info(f"loaded {wikipedia_dataset_name} containing {total_page_cnt} pages")

    max_processed_pages = total_page_cnt  # Change to single digits for spot checking / debugging
    pages_processed_cnt = 0

    corpus_file_part_idx = 0
    current_corpus_file_char_len = 0
    is_completed = False
    iter_idx = 0
    while not is_completed:  # Do till completed
        with open(f"{corpus_file_prefix}_{corpus_file_part_idx}.txt", "a", encoding="utf-8") as f:
            while iter_idx < (total_page_cnt - 1):
                page = wikipedia_dataset["train"][iter_idx]
                page_char_len = len(page["text"])  # Character len because bytes requires encoding
                if page_char_len + current_corpus_file_char_len > 1_000_000_000:
                    corpus_file_part_idx += 1  # New partition
                    current_corpus_file_char_len = 0  # Reset tally
                    break

                page_chunk_cnt = 0
                for page_chunk in page["text"].split("\n\n"):
                    page_chunk_len = len(page_chunk)
                    if not page_chunk or page_chunk[0] == " ":
                        continue
                    elif page_chunk_len > max_line_char_len:
                        with open(uber_chunk_file, "a", encoding="utf-8") as uber_chunk_f:
                            uber_chunk_f.write(page_chunk + "\n\n")
                        continue

                    page_chunk_lines = page_chunk.split("\n")
                    for chunk_line in page_chunk_lines:
                        if not chunk_line or chunk_line[0] == " ":
                            continue
                        elif len(white_space_pattern.split(chunk_line)) > 10:  # Require at least 10 naive tokens
                            f.write(chunk_line + "\n")
                            current_corpus_file_char_len += len(chunk_line)
                    page_chunk_cnt += 1

                iter_idx += 1
                pages_processed_cnt += 1

                if (pages_processed_cnt % 100) == 0:
                    logger.info(f"processed {pages_processed_cnt}/{total_page_cnt} pages")
                if pages_processed_cnt >= max_processed_pages:
                    is_completed = True
                    break
            if not is_completed and iter_idx == (total_page_cnt - 1):
                is_completed = True

if args.train:
    corpus_files = [f"{corpus_dir}/{f}" for f in os.listdir(corpus_dir) if f.startswith("sentencepiece_corpus")]
    logger.info(f"corpus_files: {corpus_files}")

    spm_training_args = [
        "--model_prefix=models/germ/tokenizer/germ_sentencepiece",
        "--model_type=unigram",
        "--shuffle_input_sentence=true",
        "--split_digits=true",
        f"--input={','.join(random.sample(corpus_files, 15))}",
        f"--input_sentence_size={input_sentence_size}",
        f"--max_sentence_length={max_line_char_len}",
        f"--vocab_size={vocab_size}",
    ]
    spm.SentencePieceTrainer.Train(" ".join(spm_training_args))

# Now you can load the model and test it:
sp = spm.SentencePieceProcessor()
sp.LoadFromFile(f"{model_file_prefix}.model")

print(sp.EncodeAsPieces("Hello world!"))
print(sp.EncodeAsPieces("127.0.0.1 is the localhost address."))
print(sp.EncodeAsPieces("1/2 is equivalent to 0.5 or 50%"))
print(sp.EncodeAsPieces("John was running so fast, you can just tell he's a runner."))
print(sp.EncodeAsPieces("He excels at math and competed in the Math Olympiad"))
print(sp.EncodeAsPieces("Watson was on his way to 221B Baker Street when the robbery occurred."))
print(sp.EncodeAsPieces("That's Uncopyrightable."))
print(sp.EncodeAsPieces("She's full of incomprehensibilities."))
print(sp.EncodeAsPieces("He's a total sesquipedalian."))
