from importlib.metadata import PackageNotFoundError, version


def test_phase3_scikit_learn_is_pinned() -> None:
    try:
        installed_version = version('scikit-learn')
    except PackageNotFoundError:
        installed_version = None

    assert installed_version == '1.7.2'
