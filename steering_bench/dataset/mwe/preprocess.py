def _maybe_swap(a, b, condition):
    if condition:
        return b, a
    return a, b


def preprocess_and_randomize_dataset(
    list_data: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Preprocess and randomize the dataset

    Randomization ensures that there are an equal number of samples where the correct answer is A and B respectively.
    """
    new_list_data = []
    for i, data in enumerate(list_data):
        question = data["question"]
        answer_a = data["answer_matching_behavior"]
        answer_b = data["answer_not_matching_behavior"]

        # Construct A/B formatted question / answer pair
        choices_str = "Choices:"
        new_answer_matching_behavior = "(A)"
        new_answer_not_matching_behavior = "(B)"

        # Swap A and B every 2 samples
        # The MWE data alternates every sample, so don't want to undo their alternating
        swap = int(i / 2) % 2 == 0
        answer_a, answer_b = _maybe_swap(answer_a, answer_b, swap)
        new_question = f"{question}\n\n{choices_str}\n(A): {answer_a}\n(B): {answer_b}"
        new_answer_matching_behavior, new_answer_not_matching_behavior = _maybe_swap(
            new_answer_matching_behavior, new_answer_not_matching_behavior, swap
        )

        new_list_data.append(
            {
                "question": new_question,
                "answer_matching_behavior": new_answer_matching_behavior,
                "answer_not_matching_behavior": new_answer_not_matching_behavior,
            }
        )

    return new_list_data
