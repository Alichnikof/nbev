# Shared by both architectures so they see the same text and only the encoding differs

# Short natural-language task hint
TASK_PROMPTS = {
    "choice": "Choose the best option.",
    "score": "Rate it on the given scale.",
    "noul": "Answer yes or no.",
}


def context_prefix(instructions, task_type):
    # Task hint and question come before the state, so every state token is read knowing what is asked
    return f"{TASK_PROMPTS[task_type]}\nQuestion: {instructions}\n\n"
