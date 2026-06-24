import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json

device = "cuda"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
llama_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

story = "[Charles] went to his mother [Victoria] ''s house to play cards. [Andrew], [Victoria]'s other son, was there too. [Donald] showed up later and asked his son [Charles] to deal him in too. [Gilbert] got his son, [Samuel], a car for his birthday. [Andrew] meet his uncle, [Samuel], at the baseball game, excited for their team to win."

prompt = """You are an expert at extracting family trees into JSON graphs.
Extract all explicitly stated family relationships from the story as a strict JSON list of objects.
Format:
[
  {"subject": "PersonA", "relation": "relationship", "object": "PersonB"}
]

CRITICAL RULE: This means that `subject` is the `relation` of `object`.
Example: "Alice is the mother of Bob." -> {"subject": "Alice", "relation": "mother", "object": "Bob"}

Use ONLY these exact relationship words: father, mother, son, daughter, grandfather, grandmother, grandson, granddaughter, uncle, aunt, nephew, niece, brother, sister, husband, wife, son-in-law, daughter-in-law, father-in-law, mother-in-law, brother-in-law, sister-in-law.

Story: """ + story + """
JSON:"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = llama_model.generate(**inputs, max_new_tokens=400, do_sample=False)
ans = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"RAW LLAMA OUTPUT:\n{ans}")
