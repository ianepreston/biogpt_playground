import os

import torch
from transformers import AutoTokenizer, BioGptForCausalLM, BioGptModel

model_path = "/models/biogptmodel"
pipe = BioGptModel.from_pretrained(
    "microsoft/biogpt",
    use_auth_token=os.getenv("HUGGINGFACE_KEY"),
)
pipe.save_pretrained(model_path)

tokenizer_path = "/models/biogpttokenizer"
pipe = AutoTokenizer.from_pretrained(
    "microsoft/biogpt",
    use_auth_token=os.getenv("HUGGINGFACE_KEY"),
)
pipe.save_pretrained(tokenizer_path)

causal_model_path = "/models/biogptcausal"
pipe = BioGptForCausalLM.from_pretrained(
    "microsoft/biogpt",
    use_auth_token=os.getenv("HUGGINGFACE_KEY"),
)
pipe.save_pretrained(causal_model_path)
