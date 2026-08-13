from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_representation_has_no_execution_surface() -> None:
    forbidden = ("representation_probe", "frozen_representation")
    surfaces = (ROOT / "cloud", ROOT / ".github" / "workflows", ROOT / "scripts")
    offenders = []
    for surface in surfaces:
        for path in surface.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                if any(word in text for word in forbidden):
                    offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
