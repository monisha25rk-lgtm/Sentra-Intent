SENTRA CLEANED DATASET
======================

Primary supervised model files:
- sentra_train_clean.jsonl
- sentra_test1_clean.jsonl
- sentra_test2_clean.jsonl

Each file retains exactly the agreed 14 columns:
1. example_id
2. group_id
3. scenario_family
4. conversation_context
5. prompt
6. intent_labels
7. action
8. target
9. data_type
10. sensitivity
11. destination
12. authorization
13. context_dependency
14. user_role

IMPORTANT:
- Train, test1, and test2 remain separate. Do not combine the test files into training.
- The official three splits contained no duplicate (prompt, conversation_context, intent_labels) examples within each split.
- The older LLM-generated file is supplemental and is not merged into the official train split.
- Its exact duplicates were removed while preserving distinct user_role metadata.
- Unlabeled rows were placed in sentra_llm_unlabeled_quarantine.jsonl. Do not train on them until they are labeled by a qualified reviewer.
- No prompt or label was rewritten or invented.

See cleaning_report.json for counts, checks, and limitations.
