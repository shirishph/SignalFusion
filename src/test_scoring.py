import json

from scoring import compare


def load_first_two(path):
    with open(path) as f:
        records = [json.loads(line) for line in f]

    return records[0], records[1]


query, candidate = load_first_two(
    "data/fingerprints/originals.jsonl"
)

result = compare(query, candidate)

print(json.dumps(result, indent=2))
