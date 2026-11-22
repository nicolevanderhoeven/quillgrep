from quillgrep.config import Privacy
from quillgrep.privacy import clean_for_export, limit_quotes, redact

ALL = ["email", "phone", "number"]


def test_emails_are_redacted():
    assert redact("write to priya.nair@example.com today", ALL) == "write to [email] today"


def test_phone_numbers_are_redacted_in_common_formats():
    for number in ("+1 555 010 0199", "(555) 010-0199", "+44 20 7946 0958", "555.010.0199", "0612345678"):
        assert "[" in redact(f"call {number} now", ALL), number


def test_dates_times_and_ordinary_numbers_survive():
    text = "Due 2026-12-15 at 14:30, 17 volunteers, 8,000 steps, page 123, version 1.2.3, 12 euro"
    assert redact(text, ALL) == text


def test_long_digit_strings_are_redacted():
    assert redact("account 123456789012 ok", ["number"]) == "account [number] ok"
    assert redact("account 123456789012 ok", ["email"]) == "account 123456789012 ok"  # only what was asked for


def test_custom_patterns_and_bad_patterns():
    assert redact("secret-42 here", [], [r"secret-\d+"]) == "[redacted] here"
    assert redact("fine", [], ["("]) == "fine"  # an invalid regex must not crash the server


def test_long_verbatim_quotes_are_cut_and_short_ones_kept():
    note = "The quick brown fox jumps over the lazy dog while the garden slowly fills with rain and early frost"
    answer = ("The note says the quick brown fox jumps over the lazy dog while the garden slowly fills with rain "
              "and early frost, which I read as a metaphor. A lazy dog is mentioned.")
    out = limit_quotes(answer, [note], max_words=8)
    assert "[…]" in out and "garden slowly fills" not in out
    assert "which I read as a metaphor" in out and "A lazy dog is mentioned." in out
    assert limit_quotes(answer, [note], max_words=0) == answer


def test_quote_limit_works_across_newlines_and_punctuation():
    note = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    out = limit_quotes("Result:\nalpha, beta, gamma delta\nepsilon zeta eta theta iota kappa lambda!", [note], 5)
    assert out.count("[…]") == 1 and "gamma" not in out


def test_clean_for_export_applies_both():
    note = " ".join(f"word{i}" for i in range(40))
    p = Privacy(max_quote_words=10)
    out = clean_for_export("mail a@b.co then " + note, p, [note])
    assert "[email]" in out and "word20" not in out
