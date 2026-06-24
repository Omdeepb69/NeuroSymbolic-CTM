"""
Llama-3 + CTM v2 Integration Script
====================================
Synced to True Log-Space Max-Plus Geometry (matches train_ctm_v2.py exactly).
All bugs from critic review fixed:
  - Forward pass: log-space max-plus (W_left + W_right + log_P)
  - W0: -1e9 null, 0.0 edges
  - Inverse edges injected in both Oracle and Pipeline modes
  - Oracle: steps = depth+1, Pipeline: steps = 6
  - Parsing crash guard on empty LLM output
"""
import torch
import torch.nn as nn
import json
import re
import ast
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from collections import defaultdict

print("==================================================")
print(" NEURO-SYMBOLIC v2: LLAMA-3 + CTM (LOG MAX-PLUS) ")
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

INVERSE_KINSHIP = {
    "father": "son",         "mother": "daughter",
    "son": "father",         "daughter": "mother",
    "grandfather": "grandson",   "grandmother": "granddaughter",
    "grandson": "grandfather",   "granddaughter": "grandmother",
    "uncle": "nephew",       "aunt": "niece",
    "nephew": "uncle",       "niece": "aunt",
    "brother": "brother",    "sister": "sister",
    "husband": "wife",       "wife": "husband",
    "father-in-law": "son-in-law",     "mother-in-law": "daughter-in-law",
    "son-in-law": "father-in-law",     "daughter-in-law": "mother-in-law",
    "brother-in-law": "brother-in-law","sister-in-law": "sister-in-law",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CTM MODEL (MUST MATCH train_ctm_v2.py EXACTLY) ---
class ConstraintTopologyMachine(nn.Module):
    def __init__(self, num_rels=22):
        super().__init__()
        self.A = nn.Parameter(torch.ones(num_rels, num_rels, num_rels) * -4.0)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, W0, qa, qb, steps=5):
        W = W0
        # Log-space max-plus: MUST match train_ctm_v2.py
        log_P = torch.log(torch.sigmoid(self.A) + 1e-9)

        for _ in range(steps):
            U = torch.full_like(W, -1e9)
            W_left = W.unsqueeze(-1).unsqueeze(2)        # (B, i, 1, k, r1, 1)
            W_right = W.transpose(1, 2).unsqueeze(-2).unsqueeze(1)  # (B, 1, j, k, 1, r2)

            for r3 in range(W.shape[-1]):
                log_P_r3 = log_P[:, :, r3].view(1, 1, 1, 1, W.shape[-1], W.shape[-1])
                V_pairs = W_left + W_right + log_P_r3
                max_r1 = V_pairs.max(dim=-2).values
                max_r2 = max_r1.max(dim=-1).values
                max_k = max_r2.max(dim=-1).values
                U[:, :, :, r3] = max_k

            W = torch.max(W, U)

        B = W.shape[0]
        preds = W[torch.arange(B), qa, qb, :]
        return preds * self.scale


print("\nLoading CTM v2 Max-Plus Engine...")
ctm_model = ConstraintTopologyMachine(NUM_RELS).to(device)
# Load the v2 weights trained with the matching log-space forward pass
MODEL_PATH = f"{ARTIFACT_DIR}/ctm_maxplus_v2_model.pt"
ctm_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
ctm_model.eval()
print(f"CTM v2 loaded from: {MODEL_PATH}")


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
        except Exception:
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
llama_model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="auto"
)
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
    prompt = f"""You are an expert at extracting family trees into JSON graphs.
Extract all explicitly stated family relationships from the story as a strict JSON list of objects.
Format:
[
  {{"subject": "PersonA", "relation": "relationship", "object": "PersonB"}}
]

CRITICAL RULE FOR DIRECTION:
This format means that the `subject` is the `relation` of the `object`.

EXAMPLES:
Story: Alice is the mother of Bob. (This means Alice is Bob's mother).
JSON: [{{"subject": "Alice", "relation": "mother", "object": "Bob"}}]

Story: David took his nephew, Eve, to the park. (This means Eve is David's nephew).
JSON: [{{"subject": "Eve", "relation": "nephew", "object": "David"}}]

Story: Carol's brother is Frank. (This means Frank is Carol's brother).
JSON: [{{"subject": "Frank", "relation": "brother", "object": "Carol"}}]

Use ONLY these exact relationship words: {", ".join(KINSHIP_RELATIONS)}.

Story: {story}
JSON:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = llama_model.generate(**inputs, max_new_tokens=400, do_sample=False)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# --- 6. HELPER: Build W0 with inverse edges ---
def build_W0(ent2id, edges, device):
    """Build W0 tensor in log-space with inverse edges."""
    W0 = torch.full((1, MAX_N, MAX_N, NUM_RELS), -1e9).to(device)
    for u, v, r in edges:
        if u in ent2id and v in ent2id and r in KIN2ID:
            # Forward edge
            W0[0, ent2id[u], ent2id[v], KIN2ID[r]] = 0.0
            # Inverse edge
            inv = INVERSE_KINSHIP.get(r)
            if inv and inv in KIN2ID:
                W0[0, ent2id[v], ent2id[u], KIN2ID[inv]] = 0.0
    return W0


# --- 7. EVALUATION LOOP ---
print("=" * 60)
print(" RUNNING HEAD-TO-HEAD BENCHMARK (v2)")
print("=" * 60)

llama_correct = 0
ctm_oracle_correct = 0
ctm_pipeline_correct = 0
results = []

# Per-depth tracking
depth_llama = defaultdict(lambda: [0, 0])
depth_oracle = defaultdict(lambda: [0, 0])
depth_pipe = defaultdict(lambda: [0, 0])

for i, story in enumerate(test_stories):
    print(f"\n--- Story {i+1}/{len(test_stories)} (Depth {story['depth']}) ---")
    target = story["target"]
    d = story["depth"]

    # === TEST A: Bare Llama-3 ===
    raw_ans = ask_llama_direct(story["text"], story["qa"], story["qb"])
    # Bug 10 fix: guard against empty split
    parts = re.sub(r'[^\w\s-]', '', raw_ans).split()
    cleaned = parts[0] if parts else ""
    llama_hit = cleaned == target
    if llama_hit: llama_correct += 1
    depth_llama[d][0] += int(llama_hit)
    depth_llama[d][1] += 1
    print(f"  Llama-3 Direct: '{cleaned}' (True: {target}) -> {'CORRECT' if llama_hit else 'WRONG'}")

    # === TEST B: CTM with Oracle Graph ===
    ent2id = {n: idx for idx, n in enumerate(story["names"][:MAX_N])}

    # Build W0 with inverse edges (log-space)
    oracle_edges = [(u, v, r) for u, v, r in story["edges"]]
    W0 = build_W0(ent2id, oracle_edges, device)

    if story["qa"] in ent2id and story["qb"] in ent2id:
        qa_id = torch.tensor([ent2id[story["qa"]]]).to(device)
        qb_id = torch.tensor([ent2id[story["qb"]]]).to(device)

        # Oracle mode: steps = depth + 1
        oracle_steps = story["depth"] + 1

        with torch.no_grad():
            logits = ctm_model(W0, qa_id, qb_id, steps=oracle_steps)
            pred_id = logits.argmax(dim=1).item()
        ctm_ans = ID2KIN.get(pred_id, "unknown")
        ctm_hit = ctm_ans == target
        if ctm_hit: ctm_oracle_correct += 1
        depth_oracle[d][0] += int(ctm_hit)
        depth_oracle[d][1] += 1
        print(f"  CTM (Oracle):   '{ctm_ans}' (True: {target}) -> {'CORRECT' if ctm_hit else 'WRONG'}")
    else:
        ctm_ans = "N/A"
        ctm_hit = False
        depth_oracle[d][1] += 1
        print(f"  CTM (Oracle):   Query entity not in graph, skipped.")

    # === TEST C: Llama-3 Extraction + CTM ===
    json_str = ask_llama_extract(story["text"])
    try:
        start = json_str.find('[')
        end = json_str.rfind(']') + 1
        if start == -1 or end <= start:
            extracted_edges = []
        else:
            extracted_edges = ast.literal_eval(json_str[start:end])
    except Exception:
        extracted_edges = []

    print(f"  Extracted Graph: {extracted_edges}")

    # Build entity map from extracted edges
    entities = list(set(
        [e.get("subject") for e in extracted_edges if isinstance(e, dict)] +
        [e.get("object") for e in extracted_edges if isinstance(e, dict)] +
        [story["qa"], story["qb"]]
    ))
    entities = [e for e in entities if e]
    pipe_ent2id = {e: idx for idx, e in enumerate(entities[:MAX_N])}

    # Build pipeline edges in (u, v, r) format for build_W0
    pipe_edges = []
    for edge in extracted_edges:
        if isinstance(edge, dict):
            u = edge.get("subject")
            v = edge.get("object")
            r = str(edge.get("relation", "")).strip().lower()
            if u and v and r in KIN2ID:
                pipe_edges.append((u, v, r))

    W0_pipe = build_W0(pipe_ent2id, pipe_edges, device)

    if story["qa"] in pipe_ent2id and story["qb"] in pipe_ent2id:
        qa_p = torch.tensor([pipe_ent2id[story["qa"]]]).to(device)
        qb_p = torch.tensor([pipe_ent2id[story["qb"]]]).to(device)

        # Pipeline mode: fixed steps = 6 (depth unknown before reasoning)
        with torch.no_grad():
            logits_p = ctm_model(W0_pipe, qa_p, qb_p, steps=6)
            pred_p = logits_p.argmax(dim=1).item()
        pipe_ans = ID2KIN.get(pred_p, "unknown")
        pipe_hit = pipe_ans == target
        if pipe_hit: ctm_pipeline_correct += 1
        depth_pipe[d][0] += int(pipe_hit)
        depth_pipe[d][1] += 1
        print(f"  Llama+CTM Pipe: '{pipe_ans}' (True: {target}) -> {'CORRECT' if pipe_hit else 'WRONG'}")
    else:
        pipe_ans = "N/A"
        pipe_hit = False
        depth_pipe[d][1] += 1
        print(f"  Llama+CTM Pipe: Query entity not found in extracted graph.")

    results.append({
        "story": i, "depth": story["depth"], "target": target,
        "llama": cleaned, "ctm_oracle": ctm_ans, "ctm_pipeline": pipe_ans,
        "llama_correct": llama_hit, "ctm_oracle_correct": ctm_hit, "ctm_pipeline_correct": pipe_hit
    })


# --- 8. RESULTS ---
N = len(test_stories)
if N == 0:
    print("\nNo test stories found!")
else:
    llama_acc = 100 * llama_correct / N
    oracle_acc = 100 * ctm_oracle_correct / N
    pipe_acc = 100 * ctm_pipeline_correct / N

    print("\n" + "=" * 60)
    print(" FINAL BENCHMARK RESULTS (Depth >= 5)")
    print("=" * 60)
    print(f"  Bare Llama-3-8B Accuracy:    {llama_acc:.1f}%")
    print(f"  CTM (Oracle Graph) Accuracy: {oracle_acc:.1f}%")
    print(f"  Llama-3 + CTM Pipeline:      {pipe_acc:.1f}%")

    # Per-depth breakdown
    print("\n  Per-Depth Breakdown:")
    print(f"  {'Depth':<8} {'Llama-3':<12} {'CTM Oracle':<14} {'Pipeline':<12}")
    print(f"  {'-'*46}")
    for d in sorted(set(list(depth_llama.keys()) + list(depth_oracle.keys()) + list(depth_pipe.keys()))):
        l_c, l_t = depth_llama[d]
        o_c, o_t = depth_oracle[d]
        p_c, p_t = depth_pipe[d]
        l_a = f"{100*l_c/l_t:.0f}%" if l_t > 0 else "N/A"
        o_a = f"{100*o_c/o_t:.0f}%" if o_t > 0 else "N/A"
        p_a = f"{100*p_c/p_t:.0f}%" if p_t > 0 else "N/A"
        print(f"  {d:<8} {l_a:<12} {o_a:<14} {p_a:<12}")

    # --- 9. VISUALIZATION ---
    labels = ['Llama-3 Alone\n(Zero-Shot)', 'CTM v2\n(Oracle Graph)', 'Llama-3 + CTM v2\n(Full Pipeline)']
    accs = [llama_acc, oracle_acc, pipe_acc]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, accs, color=colors, edgecolor='white', linewidth=2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=14)

    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Neuro-Symbolic v2: Zero-Shot Reasoning (Depth ≥ 5)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('llama3_vs_ctm_v2_results.png', dpi=300, bbox_inches='tight')
    print("\nChart saved to llama3_vs_ctm_v2_results.png")

    # Save raw results
    with open('integration_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Raw results saved to integration_v2_results.json")

print("\nExperiment v2 Complete!")
