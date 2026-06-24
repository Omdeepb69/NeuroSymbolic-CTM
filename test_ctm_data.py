import json
with open("ctm_artifacts/processed/clutrr_train.json") as f:
    data = json.load(f)
print(data[0])
