"""Canonical command vocabulary for directionally identifiable locomotion."""

CANONICAL_COMMANDS = (
    "walk",
    "walk straight",
    "walk backward",
    "walk left",
    "walk right",
    "turn left while walking",
    "turn right while walking",
    "walk in a circle",
)

ALIASES = {
    "walking": "walk straight",
    "walk forward": "walk straight",
    "walking forward": "walk straight",
    "walking backwards": "walk backward",
    "walk in circle": "walk in a circle",
}

COMMAND_TO_ID = {command: index for index, command in enumerate(CANONICAL_COMMANDS)}


def canonicalize_command(text: str) -> str:
    normalized = " ".join(text.lower().strip().split())
    return ALIASES.get(normalized, normalized)


def command_id_for_text(text: str) -> int:
    command = canonicalize_command(text)
    if command not in COMMAND_TO_ID:
        raise ValueError(f"unsupported locomotion command: {text!r}")
    return COMMAND_TO_ID[command]
