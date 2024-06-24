import torch
import time
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from accelerate import Accelerator
#
#
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
#
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
print(tokenizer.all_special_tokens)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
accelerator = Accelerator()
encoder = accelerator.prepare(model)
print(pipeline('sentiment-analysis')('I love you'))
