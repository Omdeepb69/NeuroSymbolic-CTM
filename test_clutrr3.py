from datasets import load_dataset
clutrr = load_dataset("kendrivp/CLUTRR_v1_extracted", split="test")
for row in clutrr:
    print(list(row.keys()))
    break
