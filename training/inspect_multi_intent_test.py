
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = (
    ROOT / "data" / "intent" / "encoded"
    / "test1_encoded.jsonl"
)

records = []

with TEST_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        encoded = json.loads(line)
        record = encoded["original_record"]

        if len(record["intent_labels"]) > 1:
            records.append(record)


print("===== MULTI-INTENT TEST1 INSPECTION =====")

print("Total multi-intent records:", len(records))


# 1. Intent combination distribution

combinations = Counter(
    tuple(sorted(record["intent_labels"]))
    for record in records
)

print("\n===== INTENT COMBINATIONS =====")

for combination, count in combinations.most_common():
    print(combination, ":", count)


# 2. Group distribution

groups = defaultdict(list)

for record in records:
    groups[record["group_id"]].append(record)

print("\n===== GROUP DISTRIBUTION =====")
print("Unique groups:", len(groups))
print("Largest group:", max(map(len, groups.values())))

print("\nRecords per group size:")
print(dict(Counter(map(len, groups.values()))))


# 3. Destination distribution

print("\n===== DESTINATIONS =====")

destinations = Counter(
    record["destination"]
    for record in records
)

for destination, count in destinations.most_common():
    print(destination, ":", count)


# 4. Target distribution

print("\n===== TARGETS =====")

targets = Counter(
    record["target"]
    for record in records
)

for target, count in targets.most_common():
    print(target, ":", count)


# 5. Print all multi-intent records

print("\n===== ALL MULTI-INTENT EXAMPLES =====")

for record in records:

    print("\n------------------------------")
    print("Example:", record["example_id"])
    print("Group:", record["group_id"])
    print("Intent:", record["intent_labels"])
    print("Action:", record["action"])
    print("Target:", record["target"])
    print("Destination:", record["destination"])
    print("Prompt:", record["prompt"])
