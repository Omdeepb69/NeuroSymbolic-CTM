import ast
from datasets import load_dataset

clutrr = load_dataset("kendrivp/CLUTRR_v1_extracted", split="test")

reasons = {"no_text": 0, "no_target": 0, "low_depth": 0, "no_names": 0, "bad_edges": 0, "success": 0}

for row in clutrr:
    story_text = row.get("story", row.get("clean_story", ""))
    if not story_text:
        reasons["no_text"] += 1
        continue

    target = str(row.get("target", "")).strip().lower()
    if not target:
        reasons["no_target"] += 1
        continue

    f_comb = row.get("f_comb", "")
    chain = f_comb.split("-") if isinstance(f_comb, str) and f_comb else []
    try: depth = int(row.get("num_hops", len(chain) if chain else 2))
    except Exception: depth = len(chain) if chain else 2

    if depth < 5:
        reasons["low_depth"] += 1
        continue

    genders = row.get("genders", "")
    story_edges = row.get("story_edges", [])
    edge_types = row.get("edge_types", [])

    names = []
    if isinstance(genders, str) and genders:
        names = [x.split(":")[0].strip() for x in genders.split(",") if ":" in x]
    elif isinstance(genders, list):
        names = [str(x).split(":")[0].strip() for x in genders]

    if not names:
        reasons["no_names"] += 1
        continue

    edges_parsed = story_edges
    if isinstance(story_edges, str):
        try: edges_parsed = ast.literal_eval(story_edges)
        except: pass

    types_parsed = edge_types
    if isinstance(edge_types, str):
        try: types_parsed = ast.literal_eval(edge_types)
        except: pass

    graph_edges = []
    if isinstance(edges_parsed, list) and isinstance(types_parsed, list):
        for item in zip(edges_parsed, types_parsed):
            graph_edges.append(item)
    
    if not graph_edges:
        reasons["bad_edges"] += 1
        continue

    reasons["success"] += 1

print(reasons)
