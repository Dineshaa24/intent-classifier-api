from app.preprocess import clean


def test_keeps_negation():
    assert clean("I don't want to cancel") == "i do not want to cancel"


def test_keeps_numbers_and_slashes():
    assert clean("Transfer $500 from my A/C!!") == "transfer 500 from my a/c"


def test_collapses_whitespace_and_curly_quotes():
    assert clean("  what’s   my   balance? ") == "what is my balance"
