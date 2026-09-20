"""Text repair rules for the HealthCareMagic answers.

Every rule comes from something measured in src/explore.py; the counts are in the
README funnel. Two corruption families live in this dataset and they need different
treatment:

  1. Template noise the scraper added (brand name "Chat Doctor", greetings, sign-offs,
     words glued to punctuation, a stray ".*" marker). Repairable, handled below.
  2. Word substitution inside the medical text itself ("Phoenix nerve" for "phrenic
     nerve", "Cellophane You" for "Thank You"). Not repairable without a medical
     lexicon, see the limitations section of the README.
"""

import re

ARTIFACT = re.compile(r"chat\s*doctor", re.I)

# Leftover marker from whatever pipeline produced the corpus. Always sits next to
# damaged text, so rows carrying it are dropped rather than patched.
CORRUPTION_MARK = re.compile(r"\.\s?\*|\*\s?\.")

GREETING_WORDS = (
    r"hi|hello|hey|hai|dear|welcome|thanks|thank\s*you|greetings|"
    r"good\s+(?:morning|afternoon|evening|day)|respected\s+\w+|"
    # openers the anonymiser mangled: "Hallow"/"Howell" for Hello, "Cellophane You"
    # for Thank You
    r"hallow|howell|cellophane"
)
# A greeting clause may end in a comma just as often as in a full stop:
# "Hi, Thanks for asking the query, According to your symptoms ..."
GREETING_LEAD = re.compile(rf"^\W*(?:{GREETING_WORDS})\b[^.!?,]{{0,80}}[.!?,]+[\s.]*", re.I)
# "Hi I think you can take Sphere for 5 days": a bare greeting running straight into
# the answer, with no punctuation for GREETING_LEAD to anchor on.
GREETING_BARE = re.compile(r"^\W*(?:hi|hello|hey|hai|dear)\b[\s,]+(?=[A-Za-z])", re.I)

# Acknowledgement filler that carries no clinical content. Removing it is the point of
# the exercise: the style we want is the assessment, not the pleasantries around it.
BOILERPLATE_WORDS = (
    r"(?:i )?(?:can |could |do )?understand(?:ing)? your \w+(?: and \w+)?|"
    r"i (?:have |'ve )?(?:gone through|read|reviewed|evaluated|studied|seen) "
    r"your (?:query|question|problem|concern|history|message|post)|"
    r"after (?:going through|reading|reviewing) your|"
    r"noted your (?:query|concern)|"
    r"(?:it is |i am |i'm )?(?:good|glad|sorry) to (?:hear|know)|"
    r"as per your query|"
    r"thanks? for (?:asking|posting|the query|your query|writing|contacting)"
)
BOILERPLATE_LEAD = re.compile(rf"^\W*(?:{BOILERPLATE_WORDS})\b[^.!?]{{0,90}}[.!?,]+[\s.]*", re.I)

# Closing formulas, taken from the actual distribution of final sentences rather than
# guessed. The line drawn here: phrases about the conversation go, clinical
# reassurance addressed to the patient ("Do not worry, you will be alright") stays.
SIGNOFF_WORDS = (
    r"hope[^.!?]{0,40}\b(?:help(?:s|ful|ed)?|useful|answer(?:ed)?|solved|resolved|clarified)|"
    r"hope to have been|"
    r"i (?:will |would |shall |'ll )?be (?:happy|glad|pleased) to|"
    r"would be happy to|"
    r"(?:please )?(?:do not|don't|dont) hesitate|"
    r"if you have (?:any|additional|further|more)[^.!?]{0,60}"
    r"(?:question|quer|clarification|doubt|concern)|"
    r"feel free to|let me know if|do (?:let me know|write back|revert)|revert back|"
    r"thanks?|thank you|"
    r"take care|all the best|good luck|greetings|"
    r"best (?:wishes|regards)|kind regards|warm regards|regards|"
    r"wish(?:ing)? (?:you|for)[^.!?]{0,50}|get well|"
    r"please click|\d[\s-]?star rating|positive feedback|rate (?:me|this|us)|"
    r"chat\s*doctor"
)
SIGNOFF_TAIL = re.compile(rf"(?:\s*\b(?:{SIGNOFF_WORDS})\b[^.!?]{{0,110}}[.!?]*\s*)+$", re.I)

BAD_OPENERS = {
    "degree", "hit", "his", "hallow", "howell", "cellophane", "hai",
    "hellothanks", "hellowelcome", "helloyour", "hellodear", "thanksfor",
}

WS = re.compile(r"\s+")
# "Chat Doctor.come have evaluated" and "Hi there,What you are experiencing":
# punctuation glued to the next capitalised word. Numbers are safe, the lookahead
# only fires on a letter.
GLUED = re.compile(r"([.,;:!?])(?=[A-Z])")
MARKDOWN = re.compile(r"\*\*|^\s*[-*]\s|^\s*\d+[.)]\s", re.M)
# Enumerations survive normalisation glued to the preceding word ("issues.1. Age\n# (age more than 55)2. Strong family history"). 6.8% of answers, and they are a
# different answer shape from the flowing prose we are training towards.
# \s+ after the enumerator keeps decimals like "3.5 mg" out of the match.
INLINE_LIST = re.compile(r"\d\s*[.)]\s+[A-Za-z0-9].{0,200}?\d\s*[.)]\s+[A-Za-z0-9]", re.S)
SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s*$")

# Below this the answer has been stripped down to nothing useful; keep the original.
MIN_KEEP = 120


def normalise(text: str) -> str:
    return WS.sub(" ", GLUED.sub(r"\1 ", text.replace("\u00a0", " "))).strip()


def _strip_lead(text: str, pattern: re.Pattern, rounds: int) -> str:
    out = text
    for _ in range(rounds):
        candidate = pattern.sub("", out, count=1).strip()
        if candidate == out or len(candidate) < MIN_KEEP:
            break
        out = candidate
    return out


def strip_pleasantries(text: str) -> str:
    """Remove the greeting, the acknowledgement filler and the sign-off, leaving the
    clinical assessment. That assessment is the style we are fine-tuning for."""
    out = _strip_lead(text, GREETING_LEAD, rounds=3)
    out = _strip_lead(out, GREETING_BARE, rounds=1)
    out = _strip_lead(out, BOILERPLATE_LEAD, rounds=2)
    tail = SIGNOFF_TAIL.sub("", out, count=1).strip()
    return tail if len(tail) >= MIN_KEEP else out


def first_word(text: str) -> str:
    m = re.match(r"\W*([A-Za-z']+)", text)
    return m.group(1).lower() if m else ""
