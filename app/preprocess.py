"""Text cleaning for short user utterances.

The usual "strip punctuation, drop stopwords" recipe hurts intent models: words
like "not" and "no" flip the meaning, and tokens such as "2nd" or "a/c" carry
the intent. So this keeps negations and digits and only normalises what is
safe to normalise.
"""
import re

_CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}

_KEEP = re.compile(r"[^a-z0-9/\- ]+")
_SPACES = re.compile(r"\s+")


def clean(text: str) -> str:
    text = text.lower().replace("’", "'")
    for short, full in _CONTRACTIONS.items():
        text = text.replace(short, full)
    text = _KEEP.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
