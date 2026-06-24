from datasets import load_dataset
clutrr = load_dataset("kendrivp/CLUTRR_v1_extracted", split="test")
found = 0
for row in clutrr:
    f_comb = row.get("f_comb", "")
    chain = f_comb.split("-") if isinstance(f_comb, str) and f_comb else []
    depth = int(row.get("num_hops", len(chain) if chain else 2))
    
    if depth >= 5:
        target = row.get('target', row.get('answer'))
        print(f"Target is: {target} (type: {type(target)})")
        found += 1
        if found >= 5: break
