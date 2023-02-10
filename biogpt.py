"""Try out BioGPT."""

import sys

import torch
from transformers import AutoTokenizer, BioGptForCausalLM, BioGptModel

tokenizer = AutoTokenizer.from_pretrained(
    "/models/biogpttokenizer", local_files_only=True
)
model = BioGptForCausalLM.from_pretrained("/models/biogptcausal", local_files_only=True)
# model.to("cuda")


# Read prompt from command line
args = sys.argv
del args[0]
prompt = " ".join(args)
if not prompt:
    print("No prompt provided, adding a default one")
    prompt = "Hello, my dog is cute"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model(**inputs)
loss = outputs.loss
logits = outputs.logits
