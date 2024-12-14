"""Preprocessing logic for datasets"""

from steering_bench.dataset.registry import list_datasets, build_dataset

print(list_datasets())
print(build_dataset("mwe_v2_xrisk/coordinate-itself"))

# NOTE: Preprocessing logic should not be baked into the dataset...


# NOTE: The prompt template formatting is part of the design space... So we shouldn't bake this in during preprocessing.
# Solution: Have standard utilities for formatting prompts as MCQ questions

# PROMPT_TEMPLATE = """\
# {question}

# Choices:
# {token_a} {choice_a}
# {token_b} {choice_b}""".strip()

# if __name__ == "__main__":
#     dir = RAW_DATASET_DIR / "mwe_v2_xrisk"
#     file = dir / "coordinate-itself.jsonl"

#     data = read_jsonl(file)
#     example = data[0]

#     example["token_a"] = "(A)"
#     example["token_b"] = "(B)"

#     answer_is_a = example["answer_matching_behavior"] == "(A)"

#     prompt = PROMPT_TEMPLATE.format(
#         question=example["question"],
#         choice_a=example["choice_a"],
#         choice_b=example["choice_b"],
#         token_a=example["token_a"],
#         token_b=example["token_b"],
#     )

#     response = example["token_a"] if answer_is_a else example["token_b"]
#     print(prompt)
#     print(response)
