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


# A long state keeps its first STATE_HEAD_TOKENS tokens and its end; the middle is dropped.
# The end matters most: states look like "Prompt: <conversation> Response: <reply being judged>",
# so keeping only the beginning would cut off the very response the questions are about.
STATE_HEAD_TOKENS = 64


def keep_head_and_tail(ids, max_tokens, gap_ids=()):
    """Token ids -> at most max_tokens ids: the first STATE_HEAD_TOKENS, then gap_ids (e.g. "..."), then the end."""
    if len(ids) <= max_tokens:
        return list(ids)
    head = min(STATE_HEAD_TOKENS, max_tokens // 4)
    tail = max_tokens - head - len(gap_ids)
    return list(ids[:head]) + list(gap_ids) + list(ids[-tail:])
