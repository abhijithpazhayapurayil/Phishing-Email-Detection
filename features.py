import re


# Suspicious words commonly found in phishing messages
SUSPICIOUS_KEYWORDS = [
    "urgent",
    "verify",
    "verification",
    "password",
    "account",
    "suspended",
    "click here",
    "winner",
    "won",
    "prize",
    "reward",
    "claim",
    "free",
    "bank",
    "login",
    "confirm",
    "security",
    "immediately",
    "limited time"
]


def extract_features(email):
    email_lower = email.lower()

    # Count URLs
    url_count = len(
        re.findall(r"https?://\S+|www\.\S+", email_lower)
    )

    # Count suspicious keywords
    keyword_count = 0

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in email_lower:
            keyword_count += 1

    # Count exclamation marks
    exclamation_count = email.count("!")

    # Email length
    email_length = len(email)

    # Check whether email contains a URL
    has_url = 1 if url_count > 0 else 0

    return [
        url_count,
        keyword_count,
        exclamation_count,
        email_length,
        has_url
    ]