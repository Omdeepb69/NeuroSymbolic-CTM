import torch
import torch.nn as nn
import json
import re
import ast
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

print("==================================================")
print(" NEURO-SYMBOLIC INTEGRATION: LLAMA-3 + CTM ")
print("==================================================")

# --- 1. AUTHENTICATION ---
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("HuggingFace authenticated.")
except Exception:
    print("HF_TOKEN not found. Assuming already logged in.")

ARTIFACT_DIR = "ctm_artifacts/processed"
NUM_RELS = 22
MAX_N = 15
TEST_SAMPLES = 50

KINSHIP_RELATIONS = [
    "father", "mother", "son", "daughter", "grandfather", "grandmother",
    "grandson", "granddaughter", "uncle", "aunt", "nephew", "niece",
    "brother", "sister", "husband", "wife", "son-in-law", "daughter-in-law",
    "father-in-law", "mother-in-law", "brother-in-law", "sister-in-law"
]
KIN2ID = {k: i for i, k in enumerate(KINSHIP_RELATIONS)}
ID2KIN = {v: k for k, v in KIN2ID.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CTM MODEL ---
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
            for r3 in range(W.shape[-1]):
                P_r3 = P[:, :, r3].view(1, 1, 1, 1, W.shape[-1], W.shape[-1])
                V_pairs = W_left * W_right * P_r3
                max_r1, _ = V_pairs.max(dim=-2)
                max_r2, _ = max_r1.max(dim=-1)
                max_k, _ = max_r2.max(dim=-1)
                U[:, :, :, r3] = max_k
            W = torch.max(W, U)
        B = W.shape[0]
        preds = W[torch.arange(B), qa, qb, :]
        return preds * self.scale

print("\nLoading CTM Max-Plus Engine...")
ctm_model = ConstraintTopologyMachine(NUM_RELS).to(device)
ctm_model.load_state_dict(torch.load(f"{ARTIFACT_DIR}/ctm_maxplus_model.pt", map_location=device))
ctm_model.eval()
print("CTM loaded successfully.")

# --- 3. LOAD RAW CLUTRR FROM HUGGINGFACE ---
print("\nLoading raw CLUTRR stories from HuggingFace...")
from datasets import load_dataset
clutrr_hf = load_dataset("kendrivp/CLUTRR_v1_extracted")

# Parse test stories (depth >= 5) with raw text
test_stories = []
for split_name, split_data in clutrr_hf.items():
    for row in split_data:
        try:
            story_text = row.get("story", row.get("clean_story", ""))
            if not story_text:
                continue

            # Parse query safely
            query_raw = row.get("query", "")
            qa_name, qb_name = "A", "B"
            if isinstance(query_raw, (list, tuple)) and len(query_raw) >= 2:
                qa_name, qb_name = str(query_raw[0]).strip(), str(query_raw[1]).strip()
            elif isinstance(query_raw, str):
                try:
                    parsed = ast.literal_eval(query_raw)
                    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                        qa_name, qb_name = str(parsed[0]).strip(), str(parsed[1]).strip()
                except Exception:
                    if "," in query_raw:
                        qa_name, qb_name = [x.strip() for x in query_raw.split(",", 1)]

            # Parse target relation safely
            target_str = str(row.get("target_text", row.get("answer", ""))).strip().lower()
            if target_str not in KIN2ID:
                continue

            # Parse depth safely
            f_comb = row.get("f_comb", "")
            chain = f_comb.split("-") if isinstance(f_comb, str) and f_comb else []
            try:
                depth = int(row.get("num_hops", len(chain) if chain else 2))
            except Exception:
                depth = len(chain) if chain else 2

            if depth < 5:
                continue

            # Parse graph edges safely
            genders = row.get("genders", "")
            story_edges = row.get("story_edges", [])
            edge_types = row.get("edge_types", [])

            names = []
            if isinstance(genders, str) and genders:
                names = [x.split(":")[0].strip() for x in genders.split(",") if ":" in x]
            elif isinstance(genders, list):
                names = [str(x).split(":")[0].strip() for x in genders]

            edges_parsed = story_edges
            if isinstance(story_edges, str):
                try: edges_parsed = ast.literal_eval(story_edges)
                except Exception: edges_parsed = []

            types_parsed = edge_types
            if isinstance(edge_types, str):
                try: types_parsed = ast.literal_eval(edge_types)
                except Exception: types_parsed = []

            graph_edges = []
            if isinstance(edges_parsed, list) and isinstance(types_parsed, list) and len(names) > 0:
                for (u, v), rel in zip(edges_parsed, types_parsed):
                    rel_lower = str(rel).strip().lower()
                    if isinstance(u, int) and isinstance(v, int):
                        if u < len(names) and v < len(names) and rel_lower in KIN2ID:
                            graph_edges.append((names[u], names[v], rel_lower))

            test_stories.append({
                "text": story_text,
                "qa": qa_name,
                "qb": qb_name,
                "target": target_str,
                "depth": depth,
                "names": names,
                "edges": graph_edges,
            })
        except Exception as e:
            continue

print(f"Found {len(test_stories)} depth>=5 stories with valid text and graphs.")
test_stories = test_stories[:TEST_SAMPLES]
print(f"Evaluating on {len(test_stories)} samples.\n")

# --- 4. LOAD LLAMA-3 ---
print("Loading Llama-3-8B-Instruct in 4-bit...")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
llama_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
print("Llama-3 loaded.\n")

# --- 5. PROMPTING FUNCTIONS ---
def ask_llama_direct(story, char_a, char_b):
    valid_rels = ", ".join(KINSHIP_RELATIONS)
    prompt = f"""Read the story and determine the family relationship.

Story: {story}

Question: How is {char_a} related to {char_b}?
Answer with ONLY one word from this list: {valid_rels}

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = llama_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

def ask_llama_extract(story):
    prompt = f"""Extract all explicitly stated family relationships from the story as a JSON list.
Format: [["Person1", "Person2", "relationship"], ...]
Use ONLY these words: {", ".join(KINSHIP_RELATIONS)}.

Story: {story}
JSON:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = llama_model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# --- 6. EVALUATION LOOP ---
print("=" * 60)
print(" RUNNING HEAD-TO-HEAD BENCHMARK")
print("=" * 60)

llama_correct = 0
ctm_oracle_correct = 0
ctm_pipeline_correct = 0
results = []

for i, story in enumerate(test_stories):
    print(f"\n--- Story {i+1}/{len(test_stories)} (Depth {story['depth']}) ---")
    target = story["target"]

    # === TEST A: Bare Llama-3 ===
    raw_ans = ask_llama_direct(story["text"], story["qa"], story["qb"])
    cleaned = re.sub(r'[^\w\s-]', '', raw_ans).split()[0] if raw_ans else ""
    llama_hit = cleaned == target
    if llama_hit: llama_correct += 1
    print(f"  Llama-3 Direct: '{cleaned}' (True: {target}) -> {'CORRECT' if llama_hit else 'WRONG'}")

    # === TEST B: CTM with Oracle Graph ===
    W0 = torch.zeros(1, MAX_N, MAX_N, NUM_RELS).to(device)
    ent2id = {n: idx for idx, n in enumerate(story["names"][:MAX_N])}

    for u, v, r in story["edges"]:
        if u in ent2id and v in ent2id and r in KIN2ID:
            W0[0, ent2id[u], ent2id[v], KIN2ID[r]] = 1.0

    if story["qa"] in ent2id and story["qb"] in ent2id:
        qa_id = torch.tensor([ent2id[story["qa"]]]).to(device)
        qb_id = torch.tensor([ent2id[story["qb"]]]).to(device)
        with torch.no_grad():
            logits = ctm_model(W0, qa_id, qb_id, steps=10)
            pred_id = logits.argmax(dim=1).item()
        ctm_ans = ID2KIN.get(pred_id, "unknown")
        ctm_hit = ctm_ans == target
        if ctm_hit: ctm_oracle_correct += 1
        print(f"  CTM (Oracle):   '{ctm_ans}' (True: {target}) -> {'CORRECT' if ctm_hit else 'WRONG'}")
    else:
        ctm_ans = "N/A"
        ctm_hit = False
        print(f"  CTM (Oracle):   Query entity not in graph, skipped.")

    # === TEST C: Llama-3 Extraction + CTM ===
    json_str = ask_llama_extract(story["text"])
    try:
        start = json_str.find('[')
        end = json_str.rfind(']') + 1
        extracted_edges = json.loads(json_str[start:end])
    except Exception:
        extracted_edges = []

    entities = list(set([e[0] for e in extracted_edges if len(e)==3] +
                        [e[1] for e in extracted_edges if len(e)==3] +
                        [story["qa"], story["qb"]]))
    pipe_ent2id = {e: idx for idx, e in enumerate(entities[:MAX_N])}

    W0_pipe = torch.zeros(1, MAX_N, MAX_N, NUM_RELS).to(device)
    for edge in extracted_edges:
        if len(edge) == 3:
            u, v, r = edge[0], edge[1], str(edge[2]).strip().lower()
            if r in KIN2ID and u in pipe_ent2id and v in pipe_ent2id:
                W0_pipe[0, pipe_ent2id[u], pipe_ent2id[v], KIN2ID[r]] = 1.0

    if story["qa"] in pipe_ent2id and story["qb"] in pipe_ent2id:
        qa_p = torch.tensor([pipe_ent2id[story["qa"]]]).to(device)
        qb_p = torch.tensor([pipe_ent2id[story["qb"]]]).to(device)
        with torch.no_grad():
            logits_p = ctm_model(W0_pipe, qa_p, qb_p, steps=10)
            pred_p = logits_p.argmax(dim=1).item()
        pipe_ans = ID2KIN.get(pred_p, "unknown")
        pipe_hit = pipe_ans == target
        if pipe_hit: ctm_pipeline_correct += 1
        print(f"  Llama+CTM Pipe: '{pipe_ans}' (True: {target}) -> {'CORRECT' if pipe_hit else 'WRONG'}")
    else:
        pipe_ans = "N/A"
        pipe_hit = False
        print(f"  Llama+CTM Pipe: Query entity not found in extracted graph.")

    results.append({"story": i, "depth": story["depth"], "target": target,
                     "llama": cleaned, "ctm_oracle": ctm_ans, "ctm_pipeline": pipe_ans,
                     "llama_correct": llama_hit, "ctm_oracle_correct": ctm_hit, "ctm_pipeline_correct": pipe_hit})

# --- 7. RESULTS ---
N = len(test_stories)
llama_acc = 100 * llama_correct / N
oracle_acc = 100 * ctm_oracle_correct / N
pipe_acc = 100 * ctm_pipeline_correct / N

print("\n" + "=" * 60)
print(" FINAL BENCHMARK RESULTS (Depth >= 5)")
print("=" * 60)
print(f"  Bare Llama-3-8B Accuracy:    {llama_acc:.1f}%")
print(f"  CTM (Oracle Graph) Accuracy: {oracle_acc:.1f}%")
print(f"  Llama-3 + CTM Pipeline:      {pipe_acc:.1f}%")

# --- 8. VISUALIZATION ---
labels = ['Llama-3 Alone\n(Zero-Shot)', 'CTM\n(Oracle Graph)', 'Llama-3 + CTM\n(Full Pipeline)']
accs = [llama_acc, oracle_acc, pipe_acc]
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, accs, color=colors, edgecolor='white', linewidth=2)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{acc:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Neuro-Symbolic Integration: Zero-Shot Reasoning (Depth ≥ 5)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('llama3_vs_ctm_results.png', dpi=300, bbox_inches='tight')
print("\nChart saved to llama3_vs_ctm_results.png")

# Save raw results
with open('integration_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Raw results saved to integration_results.json")
print("\nExperiment Complete!")
