
import json
from collections import Counter
from pathlib import Path

DATASET = Path(
    "data/intent/sentra_train_clean.jsonl"
)

group_counts = Counter()
total_records = 0

with DATASET.open("r", encoding="utf-8") as file:

    for line in file:
        record = json.loads(line)

        group_counts[record["group_id"]] += 1
        total_records += 1

sizes = list(group_counts.values())

print("\n===== GROUP DISTRIBUTION =====")

print("Total records:", total_records)
print("Unique groups:", len(group_counts))
print("Smallest group:", min(sizes))
print("Largest group:", max(sizes))
print("Average records per group:", round(
    sum(sizes) / len(sizes), 2
))

print("\nGroup size distribution:")

for size, count in sorted(Counter(sizes).items()):
    print(f"{size} records: {count} groups")

print("\nFirst 10 groups:")

for group_id, count in list(group_counts.items())[:10]:
    print(group_id, "->", count)
