from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import set_seed
from tqdm import tqdm

import sys
import torch


set_seed(111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_file = sys.argv[1]
output_file = sys.argv[2]
model_name = sys.argv[3]
do_sample=bool(sys.argv[4])
num_lines_to_generate = int(sys.argv[5])
num_beams=1
print("***** Generation Config ******* ")
print(f" do_sample={do_sample}")
print(f" model_name={model_name}")
print(f" input_file={input_file}")
print(f" output_file={output_file}")
batch_size = 128
with open(input_file, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

example = ["Lo luns, los cientificos d'a facultat", "Los investigadors principals mas destacaus"]
print(f"Example: {example}")
model_inputs = tokenizer(example, return_tensors='pt', padding=True).to(device)
output_model = model.generate(**model_inputs, max_new_tokens=60, do_sample=do_sample, num_beams=num_beams)
output = tokenizer.batch_decode(output_model, skip_special_tokens=True)
print(f"output: {output}")

new_lines_generated = []
for i in tqdm(range(3, num_lines_to_generate+3), total=num_lines_to_generate):
    sent_to_generate = []
    for index in range(0, len(lines), batch_size):
        sent_to_generate = [' '.join(sent.split(' ')[:i]) for sent in lines[index:index+batch_size]]
        model_inputs = tokenizer(sent_to_generate, return_tensors='pt', padding=True).to(device)
        output = tokenizer.batch_decode(model.generate(**model_inputs, max_new_tokens=60, do_sample=do_sample, num_beams=num_beams), skip_special_tokens=True)
        new_lines_generated.extend(output)

with open(output_file, 'w') as f:
    print(*[line.replace('\n', ' ') for line in new_lines_generated], sep='\n', file=f)
