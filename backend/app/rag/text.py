import re

BOUNDARIES = "。；！？\n"


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_at(text: str, start: int, end: int) -> str:
    left = max((text.rfind(char, 0, start) for char in BOUNDARIES), default=-1) + 1
    right = min(
        (index + 1 for index in (text.find(char, end) for char in BOUNDARIES) if index != -1),
        default=len(text),
    )
    return text[left:right].strip()[:400]