from app.utils.text_cleaning import clean_text


def test_clean_text_removes_control_chars() -> None:
    raw = "Hello\x00\x1f  world\r\n\r\n\r\nNext"
    cleaned = clean_text(raw)
    assert cleaned == "Hello world\n\nNext"


def test_clean_text_handles_empty_input() -> None:
    assert clean_text("") == ""
