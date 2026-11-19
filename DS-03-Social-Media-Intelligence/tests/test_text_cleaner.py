from src.preprocessing.text_cleaner import clean_text


def test_clean_text_removes_url_and_extra_whitespace():
    assert clean_text("Hello   https://example.com  world") == "Hello world"
