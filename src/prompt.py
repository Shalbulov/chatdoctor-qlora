"""Prompt pieces shared by data prep, training and generation.

The source dataset stores this line in its `instruction` column, identical for all
112k rows, so it carries no per-example information. We lift it into a system prompt
instead of repeating it inside every training example.
"""

SYSTEM_PROMPT = (
    "If you are a doctor, please answer the medical questions "
    "based on the patient's description."
)


def build_messages(instruction: str, response: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
    return msgs
