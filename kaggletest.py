"""
╔══════════════════════════════════════════════════════════════════════╗
║        CTM RESEARCH — FULL DATA PIPELINE (KAGGLE-READY)            ║
║        Constraint Topology Machine — Data Preparation               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Datasets:  WN18RR  │  FB15k-237  │  CLUTRR  │  Synthetic Kinship  ║
║  Outputs:   Composition Tables │ 2/3-hop Paths │ CTM Triples        ║
║  Runtime:   ~15-25 min on Kaggle CPU                                ║
║  Disk:      ~2GB in /kaggle/working/ctm_data/                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  ENABLE INTERNET ACCESS in Kaggle notebook settings before running  ║
╚══════════════════════════════════════════════════════════════════════╝

WHY EACH DATASET:
  WN18RR      : 11 clean relation types (hypernym, has_part etc.)
                Perfect for validating A tensor learns typed transitivity.
  FB15k-237   : 237 relations, real-world facts.
                Tests A at scale with messier composition patterns.
  CLUTRR      : Multi-hop kinship reasoning, 2–10 hops.
                The PRIMARY eval benchmark — transformers fail at hop≥4.
  Synthetic   : Always-available fallback. Used for data augmentation.

KEY OUTPUT — WHY IT MATTERS:
  composition_table[r1, r2, r3] = P(path via r1→r2 implies direct r3)
  This IS the supervisory signal for learning A[i,k,j] in the TCS step.
"""

# ═══════════════════════════════════════════════════════════════════════
# SECTION 0: SETUP & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

import os
import sys
import json
import math
import time
import random
import urllib.request
import tarfile
import zipfile
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
from itertools import product as iproduct

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

# Try matplotlib — optional for visualizations
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = Path("/kaggle/working/ctm_data")
RAW_DIR    = BASE_DIR / "raw"
PROC_DIR   = BASE_DIR / "processed"
VIZ_DIR    = BASE_DIR / "visualizations"
REPORT_DIR = BASE_DIR / "reports"

for d in [BASE_DIR, RAW_DIR, PROC_DIR, VIZ_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────
SEED               = 42
TOP_K_ENTITIES     = 2048   # CTM concept vocabulary size (N in the paper)
NEG_SAMPLE_RATIO   = 3      # negatives per positive in 2-hop dataset
MAX_PATHS_PER_PAIR = 100    # cap paths between any (r1,r2) pair to prevent explosion
MAX_FB_RELATIONS   = 50     # only top-50 FB15k relations for composition table

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

START_TIME = time.time()

# ── Checkpointing Logic ────────────────────────────────────────────────
def atomic_save_torch(obj, filepath: Path):
    """Saves safely to prevent corruption if Kaggle is interrupted."""
    tmp_path = filepath.with_suffix('.tmp')
    torch.save(obj, tmp_path)
    if filepath.exists():
        filepath.unlink() # Delete old checkpoint
    tmp_path.rename(filepath)

def atomic_save_json(obj, filepath: Path):
    tmp_path = filepath.with_suffix('.tmp')
    with open(tmp_path, "w") as f:
        json.dump(obj, f)
    if filepath.exists():
        filepath.unlink() # Delete old checkpoint
    tmp_path.rename(filepath)

def elapsed():
    return f"{(time.time() - START_TIME):.1f}s"

def section(title):
    print(f"\n{'═'*64}")
    print(f"  {title}  [{elapsed()}]")
    print(f"{'═'*64}")

def ok(msg):  print(f"  ✅ {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def info(msg): print(f"  ℹ️  {msg}")

print("╔══════════════════════════════════════════════════════════════╗")
print("║          CTM RESEARCH DATA PIPELINE                         ║")
print("╚══════════════════════════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════

section("1 / 11  │  INSTALLING DEPENDENCIES")

def try_install(pkg):
    ret = os.system(f"pip install {pkg} -q --no-warn-script-location 2>/dev/null")
    return ret == 0

PYKEEN_OK   = try_install("pykeen")
DATASETS_OK = try_install("datasets")

if PYKEEN_OK:
    try:
        from pykeen.datasets import WN18RR as PyWN18RR, FB15k237 as PyFB237
        ok("pykeen loaded")
    except Exception:
        PYKEEN_OK = False
        warn("pykeen import failed, will use direct download")
else:
    warn("pykeen install failed, will use direct download")

if DATASETS_OK:
    try:
        from datasets import load_dataset
        ok("HuggingFace datasets loaded")
    except Exception:
        DATASETS_OK = False
        warn("datasets import failed")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

section("2 / 11  │  CORE DATA STRUCTURES")

class KGDataset:
    """
    Unified container for any (h, r, t) knowledge graph dataset.
    All entity/relation IDs are integer-mapped.
    Pre-builds adjacency indexes on init for fast path queries.
    """
    def __init__(
        self,
        name: str,
        train: List[Tuple[int,int,int]],
        valid: List[Tuple[int,int,int]],
        test:  List[Tuple[int,int,int]],
        entity2id: Dict[str,int],
        relation2id: Dict[str,int],
    ):
        self.name        = name
        self.train       = train
        self.valid       = valid
        self.test        = test
        self.entity2id   = entity2id
        self.relation2id = relation2id
        self.id2entity   = {v: k for k, v in entity2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.n_entities  = len(entity2id)
        self.n_relations = len(relation2id)
        self.all_triples = train + valid + test

        info(f"Building adjacency index for {name}...")
        self._build_index()
        ok(f"{name} → {self.n_entities:,} entities, "
           f"{self.n_relations} relations, "
           f"{len(self.all_triples):,} total triples")

    def _build_index(self):
        # outgoing[h] = list of (r, t)
        self.outgoing: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
        # incoming[t] = list of (r, h)
        self.incoming: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
        # ht2rels[(h,t)] = set of r — fast edge existence check
        self.ht2rels:  Dict[Tuple[int,int], Set[int]] = defaultdict(set)
        # entity degree (in + out)
        self.degree:   Dict[int, int] = Counter()

        for h, r, t in self.all_triples:
            self.outgoing[h].append((r, t))
            self.incoming[t].append((r, h))
            self.ht2rels[(h, t)].add(r)
            self.degree[h] += 1
            self.degree[t] += 1

    def edge_exists(self, h: int, t: int) -> bool:
        return (h, t) in self.ht2rels

    def top_k_entities(self, k: int) -> List[int]:
        """Return entity IDs sorted by degree (highest first)."""
        return [e for e, _ in self.degree.most_common(k)]

    def as_train_tensor(self) -> torch.Tensor:
        """Training triples as (M, 3) int64 tensor."""
        return torch.tensor(self.train, dtype=torch.long)

    def relation_frequencies(self) -> Counter:
        return Counter(r for _, r, _ in self.all_triples)


def _triples_from_pykeen(ds_obj) -> List[Tuple[int,int,int]]:
    return [tuple(row) for row in ds_obj.mapped_triples.tolist()]

def load_wn18rr() -> KGDataset:
    """Load WN18RR — try pykeen first, fallback to GitHub raw."""
    if PYKEEN_OK:
        try:
            ds = PyWN18RR()
            return KGDataset(
                name="WN18RR",
                train     = _triples_from_pykeen(ds.training),
                valid     = _triples_from_pykeen(ds.validation),
                test      = _triples_from_pykeen(ds.testing),
                entity2id = ds.entity_to_id,
                relation2id = ds.relation_to_id,
            )
        except Exception as e:
            warn(f"pykeen WN18RR failed ({e}), falling back to direct download")

    # Direct download fallback
    wn_dir = RAW_DIR / "wn18rr"
    wn_dir.mkdir(exist_ok=True)
    base = ("https://raw.githubusercontent.com/villmow/"
            "datasets_knowledge_embedding/master/WN18RR/original")
    for split in ["train", "valid", "test"]:
        p = wn_dir / f"{split}.txt"
        if not p.exists():
            info(f"Downloading WN18RR/{split}.txt ...")
            urllib.request.urlretrieve(f"{base}/{split}.txt", p)
    return _load_from_txt_files("WN18RR", wn_dir)

def load_fb15k237() -> KGDataset:
    """Load FB15k-237."""
    if PYKEEN_OK:
        try:
            ds = PyFB237()
            return KGDataset(
                name="FB15k-237",
                train     = _triples_from_pykeen(ds.training),
                valid     = _triples_from_pykeen(ds.validation),
                test      = _triples_from_pykeen(ds.testing),
                entity2id = ds.entity_to_id,
                relation2id = ds.relation_to_id,
            )
        except Exception as e:
            warn(f"pykeen FB15k-237 failed ({e}), direct download...")

    fb_dir = RAW_DIR / "fb15k237"
    fb_dir.mkdir(exist_ok=True)
    base = ("https://raw.githubusercontent.com/villmow/"
            "datasets_knowledge_embedding/master/FB15k-237")
    for split in ["train", "valid", "test"]:
        p = fb_dir / f"{split}.txt"
        if not p.exists():
            info(f"Downloading FB15k-237/{split}.txt ...")
            urllib.request.urlretrieve(f"{base}/{split}.txt", p)
    return _load_from_txt_files("FB15k-237", fb_dir)

def _load_from_txt_files(name: str, folder: Path) -> KGDataset:
    entity2id: Dict[str,int] = {}
    relation2id: Dict[str,int] = {}
    splits = {}
    for split in ["train", "valid", "test"]:
        triples = []
        with open(folder / f"{split}.txt") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]
                if h not in entity2id:   entity2id[h]   = len(entity2id)
                if t not in entity2id:   entity2id[t]   = len(entity2id)
                if r not in relation2id: relation2id[r] = len(relation2id)
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
        splits[split] = triples
    return KGDataset(name, splits["train"], splits["valid"], splits["test"],
                     entity2id, relation2id)

# Load both KGs
section("3 / 11  │  LOADING WN18RR & FB15k-237")
wn = load_wn18rr()
fb = load_fb15k237()

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: CLUTRR — KINSHIP REASONING BENCHMARK
# ═══════════════════════════════════════════════════════════════════════

section("4 / 11  │  LOADING CLUTRR")

# ── Kinship domain constants ───────────────────────────────────────────
KINSHIP_RELATIONS = [
    "father", "mother", "son", "daughter",
    "grandfather", "grandmother", "grandson", "granddaughter",
    "uncle", "aunt", "nephew", "niece",
    "brother", "sister", "husband", "wife",
    "son-in-law", "daughter-in-law", "father-in-law", "mother-in-law",
]
KIN2ID = {k: i for i, k in enumerate(KINSHIP_RELATIONS)}

# Composition table for kinship (hand-coded ground truth for the A tensor)
# (r1, r2) → r3 : if A-r1->B and B-r2->C then A-r3->C
KINSHIP_COMPOSITION = {
    ("father",  "father"):  "grandfather",
    ("father",  "mother"):  "grandmother",
    ("mother",  "father"):  "grandfather",
    ("mother",  "mother"):  "grandmother",
    ("father",  "brother"): "uncle",
    ("father",  "sister"):  "aunt",
    ("mother",  "brother"): "uncle",
    ("mother",  "sister"):  "aunt",
    ("father",  "son"):     "brother",
    ("father",  "daughter"):"sister",
    ("mother",  "son"):     "brother",
    ("mother",  "daughter"):"sister",
    ("grandfather","son"):  "uncle",
    ("grandfather","daughter"):"aunt",
    ("grandmother","son"):  "uncle",
    ("grandmother","daughter"):"aunt",
    ("uncle",   "son"):     "cousin",
    ("aunt",    "son"):     "cousin",
    ("uncle",   "daughter"):"cousin",
    ("aunt",    "daughter"):"cousin",
    ("brother", "son"):     "nephew",
    ("sister",  "son"):     "nephew",
    ("brother", "daughter"):"niece",
    ("sister",  "daughter"):"niece",
    ("husband", "father"):  "father-in-law",
    ("husband", "mother"):  "mother-in-law",
    ("wife",    "father"):  "father-in-law",
    ("wife",    "mother"):  "mother-in-law",
    ("father",  "husband"): "son-in-law",
    ("mother",  "husband"): "son-in-law",
}
# Add "cousin" dynamically
if "cousin" not in KIN2ID:
    KIN2ID["cousin"] = len(KIN2ID)
    KINSHIP_RELATIONS.append("cousin")

def generate_kinship_chains(
    n_stories: int = 3000,
    max_depth: int = 6,
    seed: int = SEED,
) -> List[Dict]:
    """
    Generate synthetic multi-hop kinship stories.
    Each story: a chain of (person_i, relation, person_j) facts,
    with a query (person_0, person_n) whose answer requires n-1 hops.

    Returns list of dicts with CTM-ready structure.
    """
    rng = random.Random(seed)
    NAMES = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank",
        "Grace", "Henry", "Iris", "Jack", "Karen", "Leo",
        "Mary", "Noah", "Olivia", "Paul", "Quinn", "Rose",
        "Sam", "Tina", "Uma", "Victor", "Wendy", "Xena", "Yara", "Zach",
    ]
    CHAINS = [
        # Simple is-a chains
        ["father", "father"],            # 2-hop → grandfather
        ["mother", "father"],            # 2-hop → grandfather
        ["father", "brother"],           # 2-hop → uncle
        ["father", "sister"],            # 2-hop → aunt
        ["father", "son"],               # 2-hop → brother
        ["mother", "son"],               # 2-hop → brother
        ["father", "father", "father"],  # 3-hop → great-grandfather (not in table, None)
        ["father", "father", "son"],     # 3-hop → uncle
        ["father", "brother", "son"],    # 3-hop → cousin
        ["mother", "brother", "son"],    # 3-hop → cousin
        ["father", "father", "father", "son"],      # 4-hop
        ["father", "brother", "son", "son"],        # 4-hop
        ["mother", "father", "brother", "son"],     # 4-hop
        ["father", "father", "brother", "son"],     # 4-hop
        ["father", "father", "father", "son", "son"], # 5-hop
        ["mother", "father", "father", "brother", "son"], # 5-hop
        ["father", "father", "father", "father"],    # 6-hop
    ]

    stories = []
    for _ in range(n_stories):
        depth = rng.randint(2, max_depth)
        # Pick a chain of relations from CHAINS matching depth, or sample randomly
        candidate_chains = [c for c in CHAINS if len(c) == depth]
        if not candidate_chains:
            # Random chain of given depth
            base_rels = ["father", "mother", "brother", "sister"]
            chain = [rng.choice(base_rels) for _ in range(depth)]
        else:
            chain = rng.choice(candidate_chains)

        # Assign people names
        n_people = depth + 1
        people = rng.sample(NAMES, min(n_people, len(NAMES)))
        if len(people) < n_people:
            people += [f"Person{i}" for i in range(n_people - len(people))]

        # Build hard constraints (observed facts)
        constraints = []
        for i, rel in enumerate(chain):
            constraints.append({
                "entity_i":  people[i],
                "entity_j":  people[i+1],
                "relation":  rel,
                "strength":  1.0,
            })

        # Compute answer by walking composition table
        answer = chain[0]
        valid_answer = True
        for rel in chain[1:]:
            composed = KINSHIP_COMPOSITION.get((answer, rel))
            if composed is None:
                valid_answer = False
                answer = "unknown"
                break
            answer = composed

        stories.append({
            "story_id":        f"syn_{len(stories):05d}",
            "source":          "synthetic",
            "hop_depth":       depth,
            "chain":           chain,
            "people":          people,
            "constraints":     constraints,
            "query": {
                "entity_a":    people[0],
                "entity_b":    people[-1],
            },
            "target_relation": answer if valid_answer else "unknown",
            "valid_answer":    valid_answer,
            # CTM concept space: people in this story (small N for this domain)
            "concept_vocab":   {p: i for i, p in enumerate(people)},
            # CTM hard constraint matrix (N×N sparse)
            "n_concepts":      len(people),
        })

    ok(f"Generated {len(stories):,} synthetic kinship stories "
       f"(depth 2–{max_depth})")
    depth_dist = Counter(s["hop_depth"] for s in stories)
    info(f"Depth distribution: {dict(sorted(depth_dist.items()))}")
    return stories

# Load HuggingFace CLUTRR (Strictly real data, no synthetic fallback)
clutrr_stories = []
clutrr_source  = "hf"

if DATASETS_OK:
    info("Trying HuggingFace CLUTRR dataset...")
    try:
        # Try to load HuggingFace token from Kaggle Secrets for gated access
        hf_token = os.environ.get("HF_TOKEN", None)
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN_READ")
            os.environ["HF_TOKEN"] = hf_token
            info("✅ Authenticated HuggingFace via Kaggle Secrets (HF_TOKEN_READ)")
        except Exception:
            pass

        # Use the token and remove trust_remote_code as requested by the HF warning
        # Using kendrivp/CLUTRR_v1_extracted as the original CLUTRR/clutrr is broken on modern HF
        clutrr_hf = load_dataset("kendrivp/CLUTRR_v1_extracted", token=hf_token)
        ok(f"CLUTRR loaded from HuggingFace: {clutrr_hf}")
        # Parse HF CLUTRR into our format
        import ast
        for split_name, split_data in clutrr_hf.items():
            for row in split_data:
                try:
                    # Parse query entities
                    query = row.get("query", "")
                    if isinstance(query, str):
                        try:
                            # Try to safely parse "('Clarence', 'Michael')"
                            parsed = ast.literal_eval(query)
                            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                                qa, qb = parsed[0], parsed[1]
                            else:
                                qa, qb = query.split(",", 1)
                        except Exception:
                            if "," in query:
                                qa, qb = query.split(",", 1)
                            else:
                                qa, qb = "A", "B"
                    elif isinstance(query, (list, tuple)) and len(query) == 2:
                        qa, qb = query[0], query[1]
                    else:
                        qa, qb = "A", "B"

                    target = row.get("target", row.get("answer", ""))
                    
                    # Compute hop depth (fallback to counting relations if missing)
                    rel_data = row.get("f_comb", "")
                    chain = rel_data.split("-") if isinstance(rel_data, str) and rel_data else []
                    hop_depth = row.get("num_hops", len(chain) if chain else 2)

                    # Extract true graph constraints for CTM
                    constraints = []
                    genders = row.get("genders", "")
                    story_edges = row.get("story_edges", "[]")
                    edge_types = row.get("edge_types", "[]")
                    
                    if genders and story_edges != "[]" and edge_types != "[]":
                        # Map IDs to names (e.g., "Clarence:male,Emily:female" -> ["Clarence", "Emily"])
                        names = [x.split(":")[0].strip() for x in genders.split(",")]
                        edges = ast.literal_eval(story_edges) if isinstance(story_edges, str) else story_edges
                        types = ast.literal_eval(edge_types) if isinstance(edge_types, str) else edge_types
                        
                        for (u, v), rel in zip(edges, types):
                            if u < len(names) and v < len(names):
                                constraints.append({
                                    "entity_i": names[u],
                                    "entity_j": names[v],
                                    "relation": rel,
                                    "strength": 1.0
                                })
                    else:
                        names = []

                    clutrr_stories.append({
                        "story_id":        f"hf_{len(clutrr_stories):06d}",
                        "source":          "clutrr_hf",
                        "split":           split_name,
                        "story_text":      row.get("story", row.get("clean_story", "")),
                        "hop_depth":       int(hop_depth),
                        "chain":           chain,
                        "query": {"entity_a": qa.strip() if isinstance(qa, str) else str(qa), 
                                  "entity_b": qb.strip() if isinstance(qb, str) else str(qb)},
                        "target_relation": str(target),
                        "constraints":     constraints,
                        "people":          names,
                    })
                except Exception as e:
                    continue
        ok(f"Parsed {len(clutrr_stories):,} CLUTRR stories from HuggingFace")
        clutrr_source = "hf"
    except Exception as e:
        warn(f"HuggingFace CLUTRR failed: {e}")

if len(clutrr_stories) < 100:
    raise ValueError("❌ Failed to load real CLUTRR data from HuggingFace! Aborting because synthetic data fallback is disabled.")

ok(f"CLUTRR source: {clutrr_source} | total: {len(clutrr_stories):,} stories")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: RELATION COMPOSITION TABLES
# ═══════════════════════════════════════════════════════════════════════

section("5 / 11  │  RELATION COMPOSITION TABLES (A TENSOR GROUND TRUTH)")

print("""
  This is the most important analysis for CTM.
  
  We compute: for each (r1, r2) pair, what direct relation r3 
  is implied when a 2-hop path r1→r2 exists?
  
  composition_table[r1, r2, r3] = count of (h,r1,m,r2,t) paths
                                   where (h,r3,t) also exists directly.
  
  High composition_table[r1,r2,r3] → A tensor should learn:
    W[h,m] strong via r1 + W[m,t] strong via r2  ⟹  W[h,t] increases
  
  Low composition_table[r1,r2,:] → those triangles DON'T close.
    A[h,m,t] should be near 0 for this (r1,r2) combination.
""")

def compute_composition_table(
    kg: KGDataset,
    use_relations: Optional[List[int]] = None,
) -> Dict:
    """
    Compute relation composition table from training triples only.
    (Valid/test triples excluded to prevent data leakage.)

    Parameters
    ----------
    kg             : KGDataset
    use_relations  : optional subset of relation IDs to include (for FB)

    Returns dict with:
      counts        : (R, R, R) int64 tensor — raw co-occurrence counts
      total         : (R, R) int64 tensor — total 2-hop paths per (r1,r2)
      probability   : (R, R, R) float32 — normalized conditional P(r3|r1,r2)
      transitivity  : (R, R) float32 — max P(r3|r1,r2) over r3
                       ≈ "how reliably does this composition close?"
      most_likely_r3: (R, R) int64 — argmax over r3
    """
    R_all = kg.n_relations

    if use_relations is not None:
        rel_set = set(use_relations)
        R = len(use_relations)
        remap = {old: new for new, old in enumerate(use_relations)}
    else:
        rel_set = set(range(R_all))
        R = R_all
        remap = {r: r for r in range(R)}

    # Build outgoing index ONLY from training triples
    # outgoing_train[h] = list of (r, t)
    outgoing_train: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    ht2rels_train:  Dict[Tuple[int,int], Set[int]]  = defaultdict(set)

    for h, r, t in kg.train:
        outgoing_train[h].append((r, t))
        ht2rels_train[(h, t)].add(r)

    counts = torch.zeros(R, R, R, dtype=torch.int64)
    total  = torch.zeros(R, R, dtype=torch.int64)

    processed = 0
    for h, r1, m in kg.train:
        if r1 not in rel_set:
            continue
        r1_ = remap[r1]
        outgoing_m = outgoing_train.get(m, [])
        for r2, t in outgoing_m:
            if r2 not in rel_set or t == h:
                continue
            r2_ = remap[r2]
            total[r1_, r2_] += 1
            direct = ht2rels_train.get((h, t), set())
            for r3 in direct:
                if r3 in rel_set:
                    counts[r1_, r2_, remap[r3]] += 1
        processed += 1
        if processed % 10000 == 0:
            print(f"     {processed:,} / {len(kg.train):,} triples processed...",
                  end="\r")

    print()
    probability    = counts.float() / (total.float().unsqueeze(-1) + 1e-8)
    transitivity   = probability.max(dim=-1).values   # (R, R)
    most_likely_r3 = probability.argmax(dim=-1)        # (R, R)

    return {
        "counts":         counts,
        "total":          total,
        "probability":    probability,
        "transitivity":   transitivity,
        "most_likely_r3": most_likely_r3,
        "relation_ids":   use_relations if use_relations else list(range(R_all)),
        "R":              R,
    }

# WN18RR — full table (11 × 11 × 11)
info("Computing WN18RR composition table (all 11 relations)...")
wn_comp_ckpt = PROC_DIR / "wn18rr_comp_ckpt.pt"
if wn_comp_ckpt.exists():
    info("Resuming WN18RR composition from checkpoint!")
    wn_comp = torch.load(wn_comp_ckpt)
else:
    wn_comp = compute_composition_table(wn)
    atomic_save_torch(wn_comp, wn_comp_ckpt)
ok(f"WN18RR composition table: {wn_comp['R']}³ = {wn_comp['R']**3:,} entries")

# Print the WN18RR transitivity matrix (the most informative output)
print("\n  WN18RR Transitivity Scores (max P(r3|r1,r2) per pair):")
print("  Diagonal = self-composition (r ∘ r), high = transitive relation")
print()
tmat = wn_comp["transitivity"].numpy()
rel_names_short = {
    v: k.split("/")[-1][:14].replace("_", " ")
    for k, v in wn.relation2id.items()
}
header = " " * 17 + "  ".join(f"{rel_names_short.get(j, str(j)):>14}"
                               for j in range(wn.n_relations))
print("  " + header)
for i in range(wn.n_relations):
    row_vals = "  ".join(f"{tmat[i,j]:>14.3f}" for j in range(wn.n_relations))
    print(f"  {rel_names_short.get(i, str(i)):>16s}  {row_vals}")

# FB15k-237 — top-50 most common relations only
info("\nComputing FB15k-237 composition table (top-50 relations)...")
fb_rel_freq = fb.relation_frequencies()
top_fb_rels = [r for r, _ in fb_rel_freq.most_common(MAX_FB_RELATIONS)]
fb_comp_ckpt = PROC_DIR / "fb15k237_comp_ckpt.pt"
if fb_comp_ckpt.exists():
    info("Resuming FB15k-237 composition from checkpoint!")
    fb_comp = torch.load(fb_comp_ckpt)
else:
    fb_comp = compute_composition_table(fb, use_relations=top_fb_rels)
    atomic_save_torch(fb_comp, fb_comp_ckpt)
ok(f"FB15k-237 composition table: top-{MAX_FB_RELATIONS} relations, "
   f"{MAX_FB_RELATIONS}³ = {MAX_FB_RELATIONS**3:,} entries")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: 2-HOP & 3-HOP PATH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

section("6 / 11  │  MULTI-HOP PATH EXTRACTION (A TENSOR TRAINING DATA)")

print("""
  For each 2-hop path (h, r1, m, r2, t):
    label = 1  if (h, r3, t) exists for some r3 in training KG
    label = 0  otherwise (triangle does NOT close)
  
  This is the supervised training signal for A[h, m, t].
  Negative sampling: 3 negatives per positive (configurable above).
""")

def extract_multihop_paths(
    kg: KGDataset,
    max_paths_per_relation_pair: int = MAX_PATHS_PER_PAIR,
    neg_ratio: int = NEG_SAMPLE_RATIO,
    rng: random.Random = random.Random(SEED),
) -> Dict:
    """
    Extract 2-hop and 3-hop paths with triangle closure labels.

    Returns dict with tensors:
      paths_2hop  : (M, 7) — (h, r1, m, r2, t, label, r3_or_neg1)
      paths_3hop  : (K, 9) — (h, r1, m1, r2, m2, r3, t, label, composed_r)
    """
    # Index from training only
    outgoing: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    ht2rels: Dict[Tuple[int,int], Set[int]] = defaultdict(set)
    for h, r, t in kg.train:
        outgoing[h].append((r, t))
        ht2rels[(h, t)].add(r)

    # Track paths per (r1,r2) to cap explosion
    pair_counts: Dict[Tuple[int,int], int] = Counter()

    positive_2hop = []  # (h, r1, m, r2, t, 1, r3)
    negative_2hop = []  # (h, r1, m, r2, t, 0, -1)

    print(f"     Extracting 2-hop paths from {len(kg.train):,} training triples...")
    for idx, (h, r1, m) in enumerate(kg.train):
        if idx % 20000 == 0 and idx > 0:
            print(f"     {idx:,} / {len(kg.train):,} | "
                  f"pos={len(positive_2hop):,} neg={len(negative_2hop):,}", end="\r")
        for r2, t in outgoing.get(m, []):
            if t == h:  # skip trivial self-loops
                continue
            pair_key = (r1, r2)
            if pair_counts[pair_key] >= max_paths_per_relation_pair:
                continue
            pair_counts[pair_key] += 1

            direct_rels = ht2rels.get((h, t), set())
            if direct_rels:
                r3 = next(iter(direct_rels))  # take one direct relation
                positive_2hop.append((h, r1, m, r2, t, 1, r3))
            else:
                negative_2hop.append((h, r1, m, r2, t, 0, -1))

    print()
    ok(f"Raw 2-hop: {len(positive_2hop):,} positive, {len(negative_2hop):,} negative")

    # Balance dataset
    n_pos    = len(positive_2hop)
    n_neg_keep = min(len(negative_2hop), n_pos * neg_ratio)
    rng.shuffle(negative_2hop)
    balanced_neg = negative_2hop[:n_neg_keep]

    all_2hop = positive_2hop + balanced_neg
    rng.shuffle(all_2hop)

    ok(f"Balanced 2-hop: {len(all_2hop):,} total "
       f"({n_pos:,} pos, {n_neg_keep:,} neg, ratio 1:{neg_ratio})")

    # Convert to tensor: (M, 7) columns: h, r1, m, r2, t, label, r3
    paths_2hop = torch.tensor(all_2hop, dtype=torch.long)

    # 3-hop extraction (h, r1, m1, r2, m2, r3, t, label, _)
    print(f"     Extracting 3-hop paths...")
    pair3_counts: Dict[Tuple[int,int,int], int] = Counter()
    positive_3hop = []
    negative_3hop = []

    for h, r1, m1 in kg.train:
        for r2, m2 in outgoing.get(m1, []):
            for r3, t in outgoing.get(m2, []):
                if t == h or t == m1:
                    continue
                key3 = (r1, r2, r3)
                if pair3_counts[key3] >= max_paths_per_relation_pair // 2:
                    continue
                pair3_counts[key3] += 1
                direct_rels = ht2rels.get((h, t), set())
                if direct_rels:
                    r_direct = next(iter(direct_rels))
                    positive_3hop.append((h, r1, m1, r2, m2, r3, t, 1, r_direct))
                else:
                    negative_3hop.append((h, r1, m1, r2, m2, r3, t, 0, -1))

    n_pos_3 = len(positive_3hop)
    rng.shuffle(negative_3hop)
    neg3_keep = negative_3hop[:n_pos_3 * neg_ratio]
    all_3hop  = positive_3hop + neg3_keep
    rng.shuffle(all_3hop)

    paths_3hop = torch.tensor(all_3hop, dtype=torch.long)

    ok(f"Balanced 3-hop: {len(all_3hop):,} total "
       f"({n_pos_3:,} pos, {len(neg3_keep):,} neg)")

    return {
        "paths_2hop": paths_2hop,   # (M, 7): h,r1,m,r2,t,label,r3
        "paths_3hop": paths_3hop,   # (K, 9): h,r1,m1,r2,m2,r3,t,label,r_direct
        "n_pos_2hop": n_pos,
        "n_pos_3hop": n_pos_3,
        "pair_coverage": len(pair_counts),   # how many (r1,r2) pairs seen
    }

def get_paths_with_ckpt(kg, name):
    ckpt_path = PROC_DIR / f"{name}_paths_ckpt.pt"
    if ckpt_path.exists():
        info(f"Resuming {name} multihop paths from checkpoint!")
        return torch.load(ckpt_path)
    res = extract_multihop_paths(kg)
    atomic_save_torch(res, ckpt_path)
    return res

wn_paths = get_paths_with_ckpt(wn, "wn18rr")
fb_paths = get_paths_with_ckpt(fb, "fb15k237")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: CTM CONCEPT VOCABULARY
# ═══════════════════════════════════════════════════════════════════════

section("7 / 11  │  CTM CONCEPT VOCABULARY (N = TOP_K_ENTITIES)")

print(f"""
  CTM needs a fixed N-dimensional concept space.
  We select the top-{TOP_K_ENTITIES} entities by graph degree as the
  'concept vocabulary'. These are the semantic hubs of each KG.
  
  In the full CTM: entity embeddings from KG training would define
  the N concept dimensions (N ≈ 512–2048). This is the initial design.
""")

wn_top_entities  = wn.top_k_entities(TOP_K_ENTITIES)
fb_top_entities  = fb.top_k_entities(TOP_K_ENTITIES)

wn_concept_vocab = {
    "entity2concept": {str(e): i for i, e in enumerate(wn_top_entities)},
    "concept2entity": {i: str(e) for i, e in enumerate(wn_top_entities)},
    "entity_names":   {str(e): wn.id2entity.get(e, str(e))
                       for e in wn_top_entities},
    "n_concepts":     len(wn_top_entities),
    "n_relations":    wn.n_relations,
    "relation_names": {str(r): wn.id2relation.get(r, str(r))
                       for r in range(wn.n_relations)},
}

fb_concept_vocab = {
    "entity2concept": {str(e): i for i, e in enumerate(fb_top_entities)},
    "concept2entity": {i: str(e) for i, e in enumerate(fb_top_entities)},
    "entity_names":   {str(e): fb.id2entity.get(e, str(e))
                       for e in fb_top_entities},
    "n_concepts":     len(fb_top_entities),
    "n_relations":    fb.n_relations,
    "relation_names": {str(r): fb.id2relation.get(r, str(r))
                       for r in range(fb.n_relations)},
}

ok(f"WN18RR concept vocab: {len(wn_top_entities):,} concepts, "
   f"{wn.n_relations} relations")
ok(f"FB15k-237 concept vocab: {len(fb_top_entities):,} concepts, "
   f"{fb.n_relations} relations")

# Print degree distribution stats for the concept vocabulary
wn_degrees  = [wn.degree[e] for e in wn_top_entities]
info(f"WN18RR top-{TOP_K_ENTITIES} entity degree: "
     f"min={min(wn_degrees)}, median={sorted(wn_degrees)[len(wn_degrees)//2]}, "
     f"max={max(wn_degrees)}")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: CLUTRR → CTM FORMAT CONVERSION
# ═══════════════════════════════════════════════════════════════════════

section("8 / 11  │  CLUTRR → CTM CONSTRAINT TRIPLE FORMAT")

print("""
  Convert CLUTRR stories to CTM-ready format:
  
  Each story becomes:
    {
      "n_concepts"  : N (people in story = concept space size)
      "concept_ids" : {name: int}
      "hard_mask"   : N×N — 1 where a constraint is known
      "hard_values" : N×N — relation strength (1.0 for known facts)
      "rel_matrix"  : N×N — relation type for each known pair (-1=none)
      "query"       : [a_idx, b_idx]
      "target_rel"  : int (relation type of answer)
      "hop_depth"   : int
    }
  
  The hard_mask and hard_values directly feed into CTM.forward().
""")

def story_to_ctm(story: Dict) -> Optional[Dict]:
    """Convert a single story dict to CTM tensor format."""
    constraints = story.get("constraints", [])
    people      = story.get("people", [])
    query       = story.get("query", {})
    target_rel  = story.get("target_relation", "unknown")

    if not people or not constraints:
        return None

    # Build concept index for THIS story
    concept_ids = {p: i for i, p in enumerate(people)}
    N = len(people)

    hard_mask   = torch.zeros(N, N)
    hard_values = torch.zeros(N, N)
    rel_matrix  = torch.full((N, N), -1, dtype=torch.long)

    for c in constraints:
        a_name = c.get("entity_i", "")
        b_name = c.get("entity_j", "")
        rel    = c.get("relation", "")
        val    = float(c.get("strength", 1.0))

        if a_name not in concept_ids or b_name not in concept_ids:
            continue

        ai, bi   = concept_ids[a_name], concept_ids[b_name]
        rel_id   = KIN2ID.get(rel, -1)

        hard_mask[ai, bi]   = 1.0
        hard_values[ai, bi] = val
        rel_matrix[ai, bi]  = rel_id

        # Symmetric (undirected for now — directed extension is future work)
        hard_mask[bi, ai]   = 1.0
        hard_values[bi, ai] = val

    qa = query.get("entity_a", "")
    qb = query.get("entity_b", "")
    if qa not in concept_ids or qb not in concept_ids:
        return None

    return {
        "story_id":    story.get("story_id", ""),
        "source":      story.get("source", ""),
        "hop_depth":   story.get("hop_depth", -1),
        "n_concepts":  N,
        "concept_ids": concept_ids,
        "hard_mask":   hard_mask,       # (N, N)
        "hard_values": hard_values,     # (N, N)
        "rel_matrix":  rel_matrix,      # (N, N) relation type indices
        "query":       [concept_ids[qa], concept_ids[qb]],
        "target_rel":  KIN2ID.get(target_rel, -1),
        "target_rel_name": target_rel,
    }

ctm_stories = []
skipped = 0
for story in clutrr_stories:
    result = story_to_ctm(story)
    if result:
        ctm_stories.append(result)
    else:
        skipped += 1

ok(f"Converted {len(ctm_stories):,} stories to CTM format "
   f"({skipped} skipped — missing constraints)")

# Train / val / test split by hop depth
depth_buckets: Dict[int, List[Dict]] = defaultdict(list)
for s in ctm_stories:
    depth_buckets[s["hop_depth"]].append(s)

print("\n  CTM Stories per hop depth:")
for d in sorted(depth_buckets):
    print(f"    depth-{d}: {len(depth_buckets[d]):,} stories")

# Generalization split: train on depth 2-3, test on depth 4+
clutrr_train = [s for s in ctm_stories if s["hop_depth"] <= 3]
clutrr_val   = [s for s in ctm_stories if s["hop_depth"] == 4]
clutrr_test  = [s for s in ctm_stories if s["hop_depth"] >= 5]

print(f"\n  Generalization split:")
print(f"    Train (depth ≤3): {len(clutrr_train):,}")
print(f"    Val  (depth  =4): {len(clutrr_val):,}")
print(f"    Test (depth ≥5):  {len(clutrr_test):,}")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 9: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

section("9 / 11  │  VISUALIZATIONS")

if HAS_MPL:
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0f0f0f")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ACCENT  = "#f5a623"
    ACCENT2 = "#7ed321"
    BG      = "#1a1a1a"
    TEXT    = "#e0e0e0"

    # ── Plot 1: WN18RR Transitivity Heatmap ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(BG)
    tmat = wn_comp["transitivity"].numpy()
    im   = ax1.imshow(tmat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax1, label="P(triangle closes)")
    ax1.set_title("WN18RR Relation Composition Transitivity\n"
                  "(diagonal = r∘r, high = transitive relation)",
                  color=TEXT, fontsize=11, pad=10)
    rel_labels = [rel_names_short.get(i, str(i)) for i in range(wn.n_relations)]
    ax1.set_xticks(range(wn.n_relations))
    ax1.set_yticks(range(wn.n_relations))
    ax1.set_xticklabels(rel_labels, rotation=45, ha="right",
                        color=TEXT, fontsize=7)
    ax1.set_yticklabels(rel_labels, color=TEXT, fontsize=7)
    ax1.set_xlabel("r2 (second step)", color=TEXT)
    ax1.set_ylabel("r1 (first step)",  color=TEXT)
    # Annotate cells
    for i in range(wn.n_relations):
        for j in range(wn.n_relations):
            ax1.text(j, i, f"{tmat[i,j]:.2f}", ha="center", va="center",
                     fontsize=6, color="black" if tmat[i,j] > 0.5 else "white")

    # ── Plot 2: 2-hop Path Label Distribution ────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG)
    wn_2hop = wn_paths["paths_2hop"]
    labels_cnt = Counter(wn_2hop[:, 5].tolist())
    bars = ax2.bar(
        ["Positive\n(triangle\ncloses)", "Negative\n(no direct\nedge)"],
        [labels_cnt[1], labels_cnt[0]],
        color=[ACCENT2, "#e74c3c"], alpha=0.85, edgecolor="white", linewidth=0.5
    )
    for bar, val in zip(bars, [labels_cnt[1], labels_cnt[0]]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{val:,}", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax2.set_title("WN18RR 2-hop Dataset\nLabel Distribution",
                  color=TEXT, fontsize=11)
    ax2.set_ylabel("Count", color=TEXT)
    ax2.tick_params(colors=TEXT)

    # ── Plot 3: CLUTRR Depth Distribution ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG)
    depths = [s["hop_depth"] for s in ctm_stories]
    depth_counter = Counter(depths)
    d_vals = sorted(depth_counter.keys())
    d_cnts = [depth_counter[d] for d in d_vals]
    bars3 = ax3.bar(d_vals, d_cnts, color=ACCENT, alpha=0.85,
                    edgecolor="white", linewidth=0.5)
    ax3.axvline(x=3.5, color="#e74c3c", linestyle="--", linewidth=1.5,
                label="Train/Test split")
    ax3.set_title(f"CLUTRR Stories by Hop Depth\n(source: {clutrr_source})",
                  color=TEXT, fontsize=11)
    ax3.set_xlabel("Hop Depth", color=TEXT)
    ax3.set_ylabel("# Stories",  color=TEXT)
    ax3.tick_params(colors=TEXT)
    ax3.legend(facecolor=BG, edgecolor="white", labelcolor=TEXT, fontsize=8)

    # ── Plot 4: WN18RR Relation Frequency ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG)
    wn_rel_freq  = wn.relation_frequencies()
    rel_sorted   = sorted(wn_rel_freq.items(), key=lambda x: -x[1])
    rel_labels_s = [rel_names_short.get(r, str(r)) for r, _ in rel_sorted]
    rel_vals     = [cnt for _, cnt in rel_sorted]
    ax4.barh(range(len(rel_labels_s)), rel_vals,
             color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax4.set_yticks(range(len(rel_labels_s)))
    ax4.set_yticklabels(rel_labels_s, color=TEXT, fontsize=7)
    ax4.set_title("WN18RR Relation Frequencies\n(all triples)",
                  color=TEXT, fontsize=11)
    ax4.set_xlabel("Triple count", color=TEXT)
    ax4.tick_params(colors=TEXT)

    # ── Plot 5: Concept Degree Distribution ──────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(BG)
    all_degs = list(wn.degree.values())
    ax5.hist(all_degs, bins=50, color=ACCENT2, alpha=0.8,
             edgecolor="white", linewidth=0.3, log=True)
    ax5.axvline(x=wn.degree[wn_top_entities[-1]], color=ACCENT,
                linestyle="--", linewidth=1.5,
                label=f"top-{TOP_K_ENTITIES} cutoff")
    ax5.set_title(f"WN18RR Entity Degree Distribution\n(log scale)",
                  color=TEXT, fontsize=11)
    ax5.set_xlabel("Degree (in + out)", color=TEXT)
    ax5.set_ylabel("Count (log)",       color=TEXT)
    ax5.tick_params(colors=TEXT)
    ax5.legend(facecolor=BG, edgecolor="white", labelcolor=TEXT, fontsize=8)

    plt.suptitle("CTM Research — Data Pipeline Analysis",
                 color=ACCENT, fontsize=15, fontweight="bold", y=1.01)

    viz_path = VIZ_DIR / "ctm_data_analysis.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    ok(f"Visualization saved: {viz_path}")
else:
    warn("matplotlib not available, skipping visualizations")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 10: SAVE ALL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════

section("10 / 11  │  SAVING ALL ARTIFACTS")

def save_tensor(t: torch.Tensor, name: str):
    p = PROC_DIR / name
    atomic_save_torch(t, p)
    size_mb = p.stat().st_size / 1e6
    ok(f"{name} → {list(t.shape)} | {size_mb:.2f} MB")

def save_json(obj: Dict, name: str):
    # Convert non-serializable items (tensors → lists)
    def convert(o):
        if isinstance(o, torch.Tensor): return o.tolist()
        if isinstance(o, np.ndarray):   return o.tolist()
        if isinstance(o, dict):         return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):return [convert(i) for i in o]
        if isinstance(o, (np.int64, np.int32)): return int(o)
        if isinstance(o, (np.float64, np.float32)): return float(o)
        return o
    p = PROC_DIR / name
    atomic_save_json(convert(obj), p)
    size_mb = p.stat().st_size / 1e6
    ok(f"{name} → {size_mb:.2f} MB")

# ── WN18RR ────────────────────────────────────────────────────────────
info("Saving WN18RR...")
save_tensor(wn.as_train_tensor(),              "wn18rr_train_triples.pt")
save_tensor(torch.tensor(wn.valid),            "wn18rr_valid_triples.pt")
save_tensor(torch.tensor(wn.test),             "wn18rr_test_triples.pt")
save_tensor(wn_comp["counts"],                 "wn18rr_composition_counts.pt")
save_tensor(wn_comp["probability"],            "wn18rr_composition_prob.pt")
save_tensor(wn_comp["transitivity"],           "wn18rr_transitivity.pt")
save_tensor(wn_comp["most_likely_r3"],         "wn18rr_most_likely_r3.pt")
save_tensor(wn_paths["paths_2hop"],            "wn18rr_2hop_paths.pt")
save_tensor(wn_paths["paths_3hop"],            "wn18rr_3hop_paths.pt")
save_json(wn_concept_vocab,                    "wn18rr_concept_vocab.json")

# ── FB15k-237 ─────────────────────────────────────────────────────────
info("\nSaving FB15k-237...")
save_tensor(fb.as_train_tensor(),              "fb15k237_train_triples.pt")
save_tensor(fb_comp["counts"],                 "fb15k237_composition_counts.pt")
save_tensor(fb_comp["probability"],            "fb15k237_composition_prob.pt")
save_tensor(fb_comp["transitivity"],           "fb15k237_transitivity.pt")
save_tensor(fb_paths["paths_2hop"],            "fb15k237_2hop_paths.pt")
save_tensor(fb_paths["paths_3hop"],            "fb15k237_3hop_paths.pt")
save_json(fb_concept_vocab,                    "fb15k237_concept_vocab.json")
save_json({"top_relations": top_fb_rels,
           "relation_names": {str(r): fb.id2relation.get(r, str(r))
                              for r in top_fb_rels}},
                                               "fb15k237_top_relations.json")

# ── CLUTRR / Kinship ──────────────────────────────────────────────────
info("\nSaving CLUTRR...")

# Save as JSON (stories have variable N — can't be a single rectangular tensor)
def stories_to_json_serializable(stories):
    result = []
    for s in stories:
        row = {k: v for k, v in s.items()
               if not isinstance(v, torch.Tensor)}
        # Serialize tensors separately
        if "hard_mask" in s:
            row["hard_mask"]   = s["hard_mask"].tolist()
        if "hard_values" in s:
            row["hard_values"] = s["hard_values"].tolist()
        if "rel_matrix" in s:
            row["rel_matrix"]  = s["rel_matrix"].tolist()
        result.append(row)
    return result

save_json(stories_to_json_serializable(clutrr_train), "clutrr_train.json")
save_json(stories_to_json_serializable(clutrr_val),   "clutrr_val.json")
save_json(stories_to_json_serializable(clutrr_test),  "clutrr_test.json")

# Kinship composition table (ground truth for kinship A tensor)
kin_R = len(KIN2ID)
kin_comp_table = torch.zeros(kin_R, kin_R, kin_R)
for (r1_name, r2_name), r3_name in KINSHIP_COMPOSITION.items():
    r1 = KIN2ID.get(r1_name, -1)
    r2 = KIN2ID.get(r2_name, -1)
    r3 = KIN2ID.get(r3_name, -1)
    if r1 >= 0 and r2 >= 0 and r3 >= 0:
        kin_comp_table[r1, r2, r3] = 1.0

save_tensor(kin_comp_table, "kinship_composition_table.pt")
save_json({
    "relation2id":   KIN2ID,
    "id2relation":   {v: k for k, v in KIN2ID.items()},
    "composition":   {f"{r1},{r2}": r3
                      for (r1, r2), r3 in KINSHIP_COMPOSITION.items()},
    "n_relations":   kin_R,
    "source":        clutrr_source,
}, "kinship_vocab.json")

ok("All artifacts saved.")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 11: FINAL STATISTICS REPORT
# ═══════════════════════════════════════════════════════════════════════

section("11 / 11  │  FINAL STATISTICS REPORT")

report_lines = []
def rprint(line=""):
    print(f"  {line}")
    report_lines.append(line)

rprint("CTM RESEARCH DATA PIPELINE — STATISTICS REPORT")
rprint(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
rprint(f"Total runtime: {elapsed()}")
rprint("=" * 62)

rprint("\nDATASETS LOADED:")
rprint(f"  WN18RR      : {wn.n_entities:>7,} entities | "
       f"{wn.n_relations:>3} relations | {len(wn.all_triples):>7,} triples")
rprint(f"  FB15k-237   : {fb.n_entities:>7,} entities | "
       f"{fb.n_relations:>3} relations | {len(fb.all_triples):>7,} triples")
rprint(f"  CLUTRR      : {len(ctm_stories):>7,} stories  | "
       f"source: {clutrr_source:<12} | "
       f"depths: {min(s['hop_depth'] for s in ctm_stories)}–"
       f"{max(s['hop_depth'] for s in ctm_stories)}")

rprint("\nCOMPOSITION TABLES:")
wn_trans = wn_comp["transitivity"]
rprint(f"  WN18RR 11×11×11 table computed from training triples only.")
rprint(f"  Most transitive relation pairs (score > 0.3):")
for i in range(wn.n_relations):
    for j in range(wn.n_relations):
        sc = wn_trans[i,j].item()
        if sc > 0.3:
            r1n = rel_names_short.get(i, str(i))
            r2n = rel_names_short.get(j, str(j))
            r3n = rel_names_short.get(wn_comp["most_likely_r3"][i,j].item(), "?")
            rprint(f"    {r1n:>14} ∘ {r2n:<14} → {r3n:<14} (score={sc:.3f})")

rprint(f"\n  FB15k-237 composition table: top-{MAX_FB_RELATIONS} relations "
       f"({MAX_FB_RELATIONS}³ entries).")

rprint("\n2-HOP PATH DATASETS (A TENSOR TRAINING DATA):")
for name, paths_dict in [("WN18RR", wn_paths), ("FB15k-237", fb_paths)]:
    p2 = paths_dict["paths_2hop"]
    p3 = paths_dict["paths_3hop"]
    pos2 = (p2[:, 5] == 1).sum().item()
    neg2 = (p2[:, 5] == 0).sum().item()
    pos3 = (p3[:, 7] == 1).sum().item()
    neg3 = (p3[:, 7] == 0).sum().item()
    rprint(f"  {name}:")
    rprint(f"    2-hop: {len(p2):>7,} total "
           f"({pos2:,} pos, {neg2:,} neg, ratio 1:{NEG_SAMPLE_RATIO})")
    rprint(f"    3-hop: {len(p3):>7,} total "
           f"({pos3:,} pos, {neg3:,} neg)")

rprint("\nCONCEPT VOCABULARIES:")
rprint(f"  WN18RR   top-{TOP_K_ENTITIES}: min degree "
       f"{wn.degree[wn_top_entities[-1]]}, "
       f"max degree {wn.degree[wn_top_entities[0]]}")
rprint(f"  FB15k-237 top-{TOP_K_ENTITIES}: min degree "
       f"{fb.degree[fb_top_entities[-1]]}, "
       f"max degree {fb.degree[fb_top_entities[0]]}")

rprint("\nCLUTRR SPLIT (generalization by hop depth):")
rprint(f"  Train (depth ≤3): {len(clutrr_train):>5,} stories")
rprint(f"  Val  (depth  =4): {len(clutrr_val):>5,} stories")
rprint(f"  Test (depth ≥5):  {len(clutrr_test):>5,} stories")

rprint("\nKINSHIP COMPOSITION RULES (ground truth A tensor for kinship):")
for (r1, r2), r3 in sorted(KINSHIP_COMPOSITION.items()):
    rprint(f"  {r1:>16} ∘ {r2:<16} → {r3}")

rprint("\nARTIFACT DIRECTORY:")
for f_path in sorted(PROC_DIR.glob("*")):
    size_mb = f_path.stat().st_size / 1e6
    rprint(f"  {f_path.name:<45} {size_mb:>7.2f} MB")

rprint("\nNEXT STEPS FOR CTM RESEARCH:")
rprint("  1. Train A tensor using wn18rr_2hop_paths.pt")
rprint("     Loss: BCE on paths_2hop[:,5] (triangle closes or not)")
rprint("     Supervision: wn18rr_composition_prob.pt (relation-level targets)")
rprint("  2. Verify: A[r1,r2,r3] correlates with composition_prob[r1,r2,r3]")
rprint("     This validates the mechanism before building the full CTM")
rprint("  3. Benchmark on clutrr_test.json (depth ≥5)")
rprint("     Baseline: GPT-2 / T5 fine-tuned on depth≤3")
rprint("     CTM goal: generalize to depth 5-6 without re-training")
rprint("  4. Write the paper around that single benchmark number")

rprint("=" * 62)
rprint(f"All data saved to: {BASE_DIR}")
rprint(f"Visualizations:    {VIZ_DIR}")
rprint(f"Total runtime:     {elapsed()}")

with open(REPORT_DIR / "stats_report.txt", "w") as f:
    f.write("\n".join(report_lines))

ok("Report saved.")

print(f"\n{'╔' + '═'*60 + '╗'}")
print(f"║{'  PIPELINE COMPLETE — ALL ARTIFACTS READY FOR CTM TRAINING':^60}║")
print(f"╚{'═'*60}╝\n")