"""
Label mapping from CFPB raw (Product, Issue) pairs to 4 business classes.

Design decision (Phase 2):
- We kept only products that align with our 4 target classes.
- We mapped the most common Issues to one of: fraud, billing_dispute,
  payment_issue, account_issue.
- Ambiguous or unrelated issues are dropped.

The mapping uses keyword matching on the Issue string (case-insensitive)
because the CFPB taxonomy has drifted over the years — the same real-world
issue appears under slightly different names.
"""

# Which raw Products we keep (they map to our 4 classes)
KEEP_PRODUCTS = [
    "Credit card",
    "Credit card or prepaid card",
    "Prepaid card",
    "Checking or savings account",
    "Bank account or service",
    "Money transfer, virtual currency, or money service",
    "Money transfers",
]


# Keyword rules: check the Issue text (lowercased) against these patterns.
# Order matters — first match wins.
LABEL_RULES = [
    # ----- FRAUD -----
    ("fraud", [
        "fraud", "scam", "identity theft", "embezzlement",
        "unauthorized", "lost or stolen",
    ]),

    # ----- BILLING DISPUTE -----
    ("billing_dispute", [
        "purchase shown on your statement",
        "billing dispute",
        "fees or interest",
        "wrong amount charged",
        "incorrect exchange rate",
    ]),

    # ----- PAYMENT ISSUE -----
    ("payment_issue", [
        "making payments",
        "struggling to pay",
        "transaction problem",
        "trouble using your card",
        "money was not available",
        "problem adding money",
        "problem with a purchase",   # often a payment/refund issue
    ]),

    # ----- ACCOUNT ISSUE -----
    ("account_issue", [
        "getting a credit card",
        "closing your account",
        "closing/cancelling",
        "opening or closing",
        "managing, opening",
        "trouble accessing funds",
        "customer service",
        "customer relations",
        "other features, terms",
    ]),
]


def map_issue_to_label(issue: str):
    """
    Return one of the 4 target labels for a given Issue text,
    or None if the issue doesn't fit any class.
    """
    if not isinstance(issue, str):
        return None
    issue_lower = issue.lower()
    for label, keywords in LABEL_RULES:
        for kw in keywords:
            if kw in issue_lower:
                return label
    return None