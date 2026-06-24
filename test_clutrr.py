from datasets import load_dataset
clutrr = load_dataset("kendrivp/CLUTRR_v1_extracted", split="test")
found = 0
for row in clutrr:
    f_comb = row.get("f_comb", "")
    chain = f_comb.split("-") if isinstance(f_comb, str) and f_comb else []
    depth = int(row.get("num_hops", len(chain) if chain else 2))
    
    if depth >= 5:
        print("=========")
        print(f"Depth: {depth}")
        print(f"Genders: {repr(row.get('genders'))}")
        print(f"Story Edges: {repr(row.get('story_edges'))}")
        print(f"Edge Types: {repr(row.get('edge_types'))}")
        print(f"Target: {repr(row.get('target'))}")
        print(f"Query: {repr(row.get('query'))}")
        found += 1
        if found >= 3:
            break
print(f"\nTotal >=5 in first scan: {found}")
