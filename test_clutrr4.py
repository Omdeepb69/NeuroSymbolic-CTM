from datasets import load_dataset
clutrr = load_dataset("kendrivp/CLUTRR_v1_extracted", split="test")
for row in clutrr:
    f_comb = row.get("f_comb", "")
    chain = f_comb.split("-") if isinstance(f_comb, str) and f_comb else []
    depth = int(row.get("num_hops", len(chain) if chain else 2))
    if depth >= 5:
        print(f"Target Text: {row.get('target_text')}")
        print(f"Answer: {row.get('answer')}")
        break
