import sys


def test_import_is_lightweight_and_names_the_subject() -> None:
    import openalpha_kronos

    assert openalpha_kronos.__name__ == "openalpha_kronos"
    assert "torch" not in sys.modules
    assert "huggingface_hub" not in sys.modules
    assert "yfinance" not in sys.modules
