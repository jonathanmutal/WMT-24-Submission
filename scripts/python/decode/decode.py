from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM
import torch
import sys

class Translator:
    def __init__(self,
                 model_name: str,
                 src_lang: str,
                 tgt_lang: str,
                 use_cuda: bool = True):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        self.tgt_lang = tgt_lang
        # Check if CUDA is available and use_cuda flag is True
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def translate(self,
                  text: list,
                  max_length: int=150,
                  num_beams: int=5,
                  num_sentences: int=5):
        # Tokenize the text
        inputs = self.tokenizer(text,
                                return_tensors="pt",
                                max_length=max_length,
                                truncation=True,
                                padding=True)
        # Send inputs to the appropriate device
        inputs = inputs.to(self.device)
        # Generate translation using the model
        outputs = self.model.generate(**inputs,
                                      forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                                      max_length=max_length,
                                      num_return_sequences=num_sentences,
                                      num_beams=num_beams)
        # Decode the translated tokens to text
        translated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated_text

class Ranker:
    def __init__(self,
                 model_name: str,
                 use_cuda: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
            self.model.to(self.device)
        self.model.eval()

    def select_best_sentence(self, sentences: list):
        """We pick the sentence that maximizes perplexity"""
        max_loss, best_index = 0.0, 0
        for i, sentence in enumerate(sentences):
            encodings = self.tokenizer(sentence, return_tensors="pt")
            encodings = encodings.to(self.device)
            # Create labels by shifting the input tokens to the right
            labels = encodings.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens
            loss = self.model(**encodings, labels=labels).loss
            if loss > max_loss:
                max_loss = loss
                best_index = i
        return sentences[best_index]

def translate_file(input_file_path: str,
                   output_file_path: str,
                   translator,
                   batch_size: int = 16,
                   num_sentences: int = 1,
                   num_beams: int = 5):
    # Read the content of the input file and split into lines
    with open(input_file_path, 'r', encoding='utf-8') as f:
       lines = [line.strip() for line in f.readlines()]
    # Process the text in batches
    translated_lines = []
    translated_best_sentences = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        translated_batch = translator.translate(
            batch,
            num_sentences=num_sentences,
            num_beams=num_beams)
        translated_lines.extend(
            translated_batch[index] for index in range(0, len(batch)*num_sentences, num_sentences)
        )
        translated_best_sentences.append(translated_batch)

    # Write the translated text to the output file
    with open(output_file_path, 'w', encoding='utf-8') as o:
        print(*translated_lines, sep='\n', file=o)

    return translated_best_sentences


def rerank_translations(ranker,
                        batch_translations: list[list],
                        output_file_path: str,
                        batch_size: int=16,
                        num_sentences = 5):
    best_sentences = []
    for batch_sentences in batch_translations:
        for index in range(0, len(batch_sentences), num_sentences):

            best_sentence = ranker.select_best_sentence(batch_sentences[index:index + num_sentences])
            best_sentences.append(best_sentence)
    with open(output_file_path, 'w') as o:
        print(*best_sentences, sep='\n', file=o)
    return best_sentences

input_file = sys.argv[1]
output_file = sys.argv[2]
model_name = sys.argv[3]
src_lang = sys.argv[4]
tgt_lang = sys.argv[5]
num_beams = int(sys.argv[6])
has_lm = bool(int(sys.argv[7]))
if has_lm:
  lm_model = sys.argv[8]

num_sentences=1
batch_size=1

if has_lm:
    ranker = Ranker(lm_model)
    num_sentences=num_beams
translator = Translator(model_name, src_lang, tgt_lang)
translated_best_sentences = translate_file(input_file,
                                           output_file_path=output_file,
                                           translator=translator,
                                           batch_size=batch_size,
                                           num_sentences=num_sentences,
                                           num_beams=num_beams)

if has_lm:
    rerank_translations(ranker,
                        batch_translations=translated_best_sentences,
                        output_file_path=output_file + ".best_lm",
                        batch_size=batch_size,
                        num_sentences=num_sentences)

