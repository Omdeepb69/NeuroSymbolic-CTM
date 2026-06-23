import torch
import torch.nn as nn
import json
import re
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import os

print("==================================================")
print(" NEURO-SYMBOLIC INTEGRATION: LLAMA-3 + CTM ")
print("==================================================")

# --- 1. CONFIGURATION & AUTHENTICATION ---
# On Kaggle, this will automatically use the secret if provided
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
except Exception:
    print("Not running in Kaggle or HF_TOKEN secret not found. Assuming already logged in.")

ARTIFACT_DIR = "ctm_artifacts/processed"
NUM_RELS = 22
MAX_N = 15
TEST_SAMPLES = 50  # Number of stories to evaluate

KIN2ID = {'father': 0, 'mother': 1, 'son': 2, 'daughter': 3, 'brother': 4, 'sister': 5, 
          'husband': 6, 'wife': 7, 'grandfather': 8, 'grandmother': 9, 'grandson': 10, 
          'granddaughter': 11, 'uncle': 12, 'aunt': 13, 'nephew': 14, 'niece': 15, 
          'father-in-law': 16, 'mother-in-law': 17, 'son-in-law': 18, 'daughter-in-law': 19, 
          'brother-in-law': 20, 'sister-in-law': 21}
ID2KIN = {v: k for k, v in KIN2ID.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CTM ARCHITECTURE (TROPICAL SEMIRING) ---
class ConstraintTopologyMachine(nn.Module):
    def __init__(self, num_rels=22):
        super().__init__()
        self.A = nn.Parameter(torch.ones(num_rels, num_rels, num_rels) * -3.0)
        self.scale = nn.Parameter(torch.tensor(10.0))
        
    def forward(self, W0, qa, qb, steps=10):
        W = W0
        P = torch.sigmoid(self.A)
        
        for _ in range(steps):
            U = torch.zeros_like(W)
            W_left = W.unsqueeze(-1).unsqueeze(2)
            W_right = W.transpose(1, 2).unsqueeze(-2).unsqueeze(1)
            
            for r3 in range(NUM_RELS):
                P_r3 = P[:, :, r3].view(1, 1, 1, 1, NUM_RELS, NUM_RELS)
                V_pairs = W_left * W_right * P_r3
                max_r1, _ = V_pairs.max(dim=-2)
                max_r2, _ = max_r1.max(dim=-1)
                max_k, _ = max_r2.max(dim=-1)
                U[:, :, :, r3] = max_k
                
            W = torch.max(W, U)
            
        B = W.shape[0]
        preds = W[torch.arange(B), qa, qb, :]
        return preds * self.scale

print("Loading CTM Max-Plus Engine...")
ctm_model = ConstraintTopologyMachine(NUM_RELS).to(device)
try:
    ctm_model.load_state_dict(torch.load(f"{ARTIFACT_DIR}/ctm_maxplus_model.pt", map_location=device))
    ctm_model.eval()
    print("CTM loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: {ARTIFACT_DIR}/ctm_maxplus_model.pt not found. Train it first!")
    exit(1)

# --- 3. LLAMA-3 LOADING ---
print("Loading Llama-3-8B-Instruct in 4-bit...")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

llama_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# --- 4. PROMPTING STRATEGIES ---
def prompt_llama_direct(story_text, char_a, char_b):
    prompt = f"""You are a logical reasoning expert. Read the following story and determine the relationship between {char_a} and {char_b}.

Story: {story_text}

Question: How is {char_a} related to {char_b}?
Answer ONLY with the single lowercase word representing the relationship (e.g. brother, aunt, grandfather).

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llama_model.generate(**inputs, max_new_tokens=5, temperature=0.1, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    return response

def prompt_llama_extract(story_text):
    prompt = f"""Extract all explicitly stated family relationships from the story into a strict JSON list of lists.
Format: [["Person1", "Person2", "relationship"], ...]
Use ONLY these exact relationship words: {list(KIN2ID.keys())}.

Story: Alice went to the store with her brother, Bob. Bob introduced Alice to his mother, Carol.
JSON: [["Alice", "Bob", "brother"], ["Bob", "Carol", "mother"]]

Story: {story_text}
JSON:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llama_model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

# --- 5. EXECUTION LOOP ---
print("Loading CLUTRR Test Dataset (Depth >= 5)...")
with open(f"{ARTIFACT_DIR}/clutrr_test.json") as f:
    test_data = json.load(f)

print(f"Evaluating on {TEST_SAMPLES} samples...")

llama_correct = 0
ctm_correct = 0

for i, story in enumerate(test_data[:TEST_SAMPLES]):
    text = story['story_text']
    char_a = story['query_chars'][0]
    char_b = story['query_chars'][1]
    
    # We use the integer target from the parsed artifact to check correctness
    target_id = int(story['target_rel_name'])
    target_rel = ID2KIN[target_id]
    
    print(f"\n--- Story {i+1} ---")
    
    # 1. Baseline: Llama-3 Direct
    direct_ans = prompt_llama_direct(text, char_a, char_b)
    # Strip punctuation
    direct_ans = re.sub(r'[^\w\s]', '', direct_ans)
    
    if target_rel in direct_ans:
        llama_correct += 1
        llama_res = "CORRECT"
    else:
        llama_res = "WRONG"
        
    print(f"Llama-3 Direct Answer: {direct_ans} (True: {target_rel}) -> {llama_res}")
    
    # 2. Pipeline: Llama-3 Extraction + CTM
    json_str = prompt_llama_extract(text)
    
    # Parse the extracted JSON safely
    try:
        # Find the first [ and last ]
        start = json_str.find('[')
        end = json_str.rfind(']') + 1
        edges = json.loads(json_str[start:end])
    except Exception:
        edges = []
        
    print(f"Llama-3 Extracted Edges: {edges}")
    
    # Map entities to indices
    entities = list(set([e[0] for e in edges] + [e[1] for e in edges] + [char_a, char_b]))
    ent2id = {e: idx for idx, e in enumerate(entities)}
    
    if len(entities) > MAX_N:
        print("Story too large, skipping CTM.")
        continue
        
    W0 = torch.zeros(1, MAX_N, MAX_N, NUM_RELS).to(device)
    for edge in edges:
        if len(edge) == 3:
            u, v, r = edge
            if r in KIN2ID and u in ent2id and v in ent2id:
                W0[0, ent2id[u], ent2id[v], KIN2ID[r]] = 1.0
                
    # Run CTM
    qa_id = torch.tensor([ent2id[char_a]])
    qb_id = torch.tensor([ent2id[char_b]])
    
    with torch.no_grad():
        logits = ctm_model(W0, qa_id, qb_id, steps=10)
        pred_id = logits.argmax(dim=1).item()
        
    ctm_ans = ID2KIN[pred_id]
    
    if ctm_ans == target_rel:
        ctm_correct += 1
        ctm_res = "CORRECT"
    else:
        ctm_res = "WRONG"
        
    print(f"CTM Deduced Answer: {ctm_ans} (True: {target_rel}) -> {ctm_res}")

# --- 6. RESULTS & VISUALIZATION ---
llama_acc = (llama_correct / TEST_SAMPLES) * 100
ctm_acc = (ctm_correct / TEST_SAMPLES) * 100

print("\n==================================================")
print(" FINAL BENCHMARK RESULTS (Depth >= 5)")
print("==================================================")
print(f"Base Llama-3-8B Accuracy: {llama_acc:.2f}%")
print(f"Llama-3 + CTM Accuracy:   {ctm_acc:.2f}%")

plt.figure(figsize=(8, 6))
bars = plt.bar(['Llama-3 Alone\n(Hallucinates)', 'Llama-3 + CTM\n(Neuro-Symbolic)'], [llama_acc, ctm_acc], color=['#ff6b6b', '#4ecdc4'])
plt.ylabel('Accuracy (%)')
plt.title('Zero-Shot Generalization on Length-10 Logical Chains')
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.savefig('llama3_vs_ctm_results.png', dpi=300, bbox_inches='tight')
print("\nChart saved to llama3_vs_ctm_results.png")
