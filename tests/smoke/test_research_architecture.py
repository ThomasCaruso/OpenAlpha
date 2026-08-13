import ast
import re
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "cloud" / "modal" / "kronos_research.py"
ALLOWED_PUBLIC_MODAL_FUNCTIONS = {
    "inventory_base_artifacts",
    "inventory_zero_shot_artifacts",
    "run_base_structural_validity",
    "run_mini_structural_validity",
    "run_zero_shot_benchmark",
    "verify_base_runtime",
    "verify_mini_runtime",
    "verify_zero_shot_runtime",
}
_FORBIDDEN_RESEARCH_SHELL_IDENTITIES = (
    "control_api",
    "frozen_representation",
    "official_protocol",
    "representation_probe",
    "test_opening",
    "training",
)

_FORBIDDEN_IDENTITIES = ("representation_probe", "frozen_representation")
_TEXT_EXTENSIONS = frozenset(
    {
        ".bat",
        ".cfg",
        ".cmd",
        ".ini",
        ".json",
        ".ps1",
        ".py",
        ".pyi",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
    }
)
_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "cache",
        "dist",
        "generated",
        "node_modules",
    }
)
_ENTRY_POINT_KEYS = frozenset({"entry_points", "gui_scripts", "scripts"})
_INERT_HISTORICAL_LABEL = "PROCEED_TO_FROZEN_REPRESENTATION_PROBE"
_LOCAL_MODULE_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_EXECUTION_TARGET_MARKERS = frozenset(
    {
        "app",
        "callable",
        "command",
        "decorator",
        "entrypoint",
        "function",
        "handler",
        "job",
        "module",
        "route",
        "runner",
        "script",
        "target",
        "task",
    }
)


def _normalized(value: str) -> str:
    with_acronym_boundaries = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", value)
    with_word_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", with_acronym_boundaries)
    return with_word_boundaries.casefold().replace("-", "_")


def _references_dormant_study(value: str) -> bool:
    normalized = _normalized(value)
    return any(identity in normalized for identity in _FORBIDDEN_IDENTITIES)


def _python_imports_dormant_study(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_references_dormant_study(alias.name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _references_dormant_study(module):
                return True
            if any(_references_dormant_study(f"{module}.{alias.name}") for alias in node.names):
                return True
    return False


def _expression_references_dormant_study(
    node: ast.AST, *, allow_inert_historical_label: bool = False
) -> bool:
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and _references_dormant_study(item.id):
            return True
        if isinstance(item, ast.Attribute) and _references_dormant_study(item.attr):
            return True
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            if allow_inert_historical_label and item.value == _INERT_HISTORICAL_LABEL:
                continue
            if _references_dormant_study(item.value):
                return True
    return False


def _call_references_dormant_execution(node: ast.Call) -> bool:
    if _expression_references_dormant_study(node.func):
        return True
    called = ast.unparse(node.func)
    arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
    allow_inert_label = called not in {"__import__", "importlib.import_module"}
    return any(
        _expression_references_dormant_study(item, allow_inert_historical_label=allow_inert_label)
        for item in arguments
    )


def _assignment_is_execution_reference(node: ast.Assign | ast.AnnAssign) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = [
        item.id if isinstance(item, ast.Name) else item.attr
        for target in targets
        for item in ast.walk(target)
        if isinstance(item, ast.Name | ast.Attribute)
    ]
    if not any(
        marker in _normalized(name) for marker in _EXECUTION_TARGET_MARKERS for name in names
    ):
        return False
    return node.value is not None and _expression_references_dormant_study(node.value)


def _python_references_dormant_execution(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return any(
            _references_dormant_study(line)
            and any(
                marker in _normalized(line)
                for marker in (*_EXECUTION_TARGET_MARKERS, "import", "@")
            )
            for line in source.splitlines()
        )

    if _python_imports_dormant_study(source):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if _references_dormant_study(node.name):
                return True
            if any(_expression_references_dormant_study(item) for item in node.decorator_list):
                return True
        elif isinstance(node, ast.Call):
            if _call_references_dormant_execution(node):
                return True
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            if _assignment_is_execution_reference(node):
                return True
    return False


def _is_ignored(relative: Path) -> bool:
    return any(part.casefold() in _IGNORED_DIRECTORY_NAMES for part in relative.parts[:-1])


def _read_text(path: Path) -> str | None:
    if path.suffix.casefold() not in _TEXT_EXTENSIONS:
        return None
    content = path.read_bytes()
    if b"\x00" in content:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _surface_files(root: Path) -> Iterable[Path]:
    surfaces = (root / "cloud", root / ".github" / "workflows", root / "scripts")
    for surface in surfaces:
        if surface.is_dir():
            yield from (path for path in surface.rglob("*") if path.is_file())


def _entry_point_values(document: Mapping[str, Any]) -> Iterable[object]:
    project = document.get("project")
    if isinstance(project, Mapping):
        for key in ("scripts", "gui-scripts", "entry-points"):
            value = project.get(key)
            if value is not None:
                yield value

    tool = document.get("tool")
    if isinstance(tool, Mapping):
        pending: list[Mapping[str, Any]] = [tool]
        while pending:
            current = pending.pop()
            for key, value in current.items():
                if _normalized(str(key)) in _ENTRY_POINT_KEYS:
                    yield value
                if isinstance(value, Mapping):
                    pending.append(value)


def _string_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _string_values(item)


def _resolve_local_entrypoint(pyproject: Path, target: str) -> Path | None:
    module = target.partition(":")[0].strip()
    if not _LOCAL_MODULE_PATTERN.fullmatch(module):
        return None
    relative = Path(*module.split("."))
    project_root = pyproject.parent
    for source_root in (project_root, project_root / "src"):
        module_file = source_root / relative.with_suffix(".py")
        if module_file.is_file():
            return module_file
        package_file = source_root / relative / "__init__.py"
        if package_file.is_file():
            return package_file
    return None


def _entrypoint_targets_dormant_study(pyproject: Path, value: object) -> bool:
    for target in _string_values(value):
        module_path = _resolve_local_entrypoint(pyproject, target)
        if module_path is None:
            continue
        source = _read_text(module_path)
        if source is not None and _python_references_dormant_execution(source):
            return True
    return False


def _value_references_dormant_study(value: object) -> bool:
    if isinstance(value, str):
        return _references_dormant_study(value)
    if isinstance(value, Mapping):
        return any(
            _references_dormant_study(str(key)) or _value_references_dormant_study(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_value_references_dormant_study(item) for item in value)
    return False


def _pyproject_files(root: Path) -> Iterable[Path]:
    workspace = root / "pyproject.toml"
    if workspace.is_file():
        yield workspace
    packages = root / "packages"
    if packages.is_dir():
        yield from (
            path
            for path in packages.rglob("pyproject.toml")
            if path.is_file() and not _is_ignored(path.relative_to(root))
        )


def _find_dormant_surface_offenders(root: Path) -> list[str]:
    offenders: set[str] = set()
    for path in sorted(_surface_files(root)):
        relative = path.relative_to(root)
        if _is_ignored(relative):
            continue
        text = _read_text(path)
        if text is None:
            continue
        path_reference = _references_dormant_study(relative.as_posix())
        content_reference = (
            _python_references_dormant_execution(text)
            if path.suffix.casefold() in {".py", ".pyi"}
            else _references_dormant_study(text)
        )
        if path_reference or content_reference:
            offenders.add(relative.as_posix())

    for path in sorted(_pyproject_files(root)):
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        entry_points = tuple(_entry_point_values(document))
        if any(_value_references_dormant_study(value) for value in entry_points) or any(
            _entrypoint_targets_dormant_study(path, value) for value in entry_points
        ):
            offenders.add(path.relative_to(root).as_posix())

    return sorted(offenders)


def _write_text(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        parent = _decorator_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def test_completed_study_modal_shell_has_exactly_the_supported_surface() -> None:
    assert APP_PATH.is_file(), "the completed-study Modal shell is missing"
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    decorated_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(_decorator_name(item) == "app.function" for item in node.decorator_list)
    }
    assert decorated_functions == ALLOWED_PUBLIC_MODAL_FUNCTIONS

    decorator_names = {
        _decorator_name(item)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for item in node.decorator_list
    }
    assert "app.cls" not in decorator_names
    assert "app.local_entrypoint" not in decorator_names
    assert not {name for name in decorator_names if name.endswith("asgi_app")}

    normalized_source = source.casefold()
    offenders = {
        identity
        for identity in _FORBIDDEN_RESEARCH_SHELL_IDENTITIES
        if identity in normalized_source
    }
    assert offenders == set()


def test_dormant_surface_scan_ignores_cache_generated_and_binary_files(tmp_path: Path) -> None:
    _write_text(tmp_path, "cloud/__pycache__/worker.py", "frozen_representation")
    _write_text(tmp_path, "scripts/.cache/command.py", "representation_probe")
    _write_text(tmp_path, "scripts/generated/manifest.yml", "frozen-representation")
    binary = tmp_path / "cloud" / "FROZEN-REPRESENTATION.pyc"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\x00\x01representation_probe")

    assert _find_dormant_surface_offenders(tmp_path) == []


def test_dormant_surface_scan_catches_case_and_hyphen_variants(tmp_path: Path) -> None:
    _write_text(tmp_path, "cloud/worker.py", "command = 'FROZEN-REPRESENTATION'")
    _write_text(tmp_path, "scripts/REPRESENTATION-PROBE.py", "command = 'safe'")
    _write_text(tmp_path, ".github/workflows/research.yml", "name: REPRESENTATION-PROBE\n")

    assert _find_dormant_surface_offenders(tmp_path) == [
        ".github/workflows/research.yml",
        "cloud/worker.py",
        "scripts/REPRESENTATION-PROBE.py",
    ]


def test_python_import_scan_catches_dormant_namespace_behind_alias() -> None:
    source = (
        "from openalpha_kronos.studies.frozen_representation.worker "
        "import run_probe_fit_worker as run_frp\n"
    )

    assert _python_imports_dormant_study(source) is True


def test_dormant_surface_scan_catches_aliased_python_import(tmp_path: Path) -> None:
    _write_text(
        tmp_path,
        "scripts/run_research.py",
        (
            "from openalpha_kronos.studies.frozen_representation.worker "
            "import run_probe_fit_worker as run_frp\n"
        ),
    )

    assert _find_dormant_surface_offenders(tmp_path) == ["scripts/run_research.py"]


def test_python_surface_scan_catches_camel_case_execution_names(tmp_path: Path) -> None:
    _write_text(
        tmp_path,
        "cloud/classes.py",
        "@app.cls()\nclass FrozenRepresentationProbe: ...\n",
    )
    _write_text(
        tmp_path,
        "scripts/functions.py",
        "def runFrozenRepresentationProbe(): ...\n",
    )

    assert _find_dormant_surface_offenders(tmp_path) == [
        "cloud/classes.py",
        "scripts/functions.py",
    ]


def test_python_surface_scan_catches_dynamic_imports_and_decorators(tmp_path: Path) -> None:
    _write_text(
        tmp_path,
        "cloud/dynamic.py",
        "importlib.import_module('openalpha_kronos.studies.frozen-representation.worker')\n",
    )
    _write_text(tmp_path, "scripts/decorated.py", "@REPRESENTATION_PROBE\ndef run(): ...\n")

    assert _find_dormant_surface_offenders(tmp_path) == [
        "cloud/dynamic.py",
        "scripts/decorated.py",
    ]


def test_local_entrypoint_alias_resolves_to_a_dormant_import(tmp_path: Path) -> None:
    _write_text(
        tmp_path,
        "packages/example/pyproject.toml",
        """\
[project]
name = "example"
version = "0.1.0"

[project.scripts]
frp = "example.cli:main"
""",
    )
    _write_text(
        tmp_path,
        "packages/example/src/example/cli.py",
        (
            "from openalpha_kronos.studies.frozen_representation.worker "
            "import run_probe_fit_worker as main\n"
        ),
    )

    assert _find_dormant_surface_offenders(tmp_path) == ["packages/example/pyproject.toml"]


def test_unresolved_entrypoint_alias_is_not_flagged(tmp_path: Path) -> None:
    _write_text(
        tmp_path,
        "pyproject.toml",
        """\
[project]
name = "root"
version = "0.1.0"

[project.scripts]
frp = "external_package.cli:main"
""",
    )

    assert _find_dormant_surface_offenders(tmp_path) == []


def test_historical_decision_labels_are_not_execution_surfaces(tmp_path: Path) -> None:
    _write_text(
        tmp_path,
        "cloud/metadata.py",
        "decision_outcomes = ['PROCEED_TO_FROZEN_REPRESENTATION_PROBE']\n",
    )

    assert _find_dormant_surface_offenders(tmp_path) == []


def test_historical_decision_label_is_inert_in_calls(tmp_path: Path) -> None:
    _write_text(
        tmp_path,
        "cloud/metadata_calls.py",
        """\
record = Decision(outcome="PROCEED_TO_FROZEN_REPRESENTATION_PROBE")
logger.info("PROCEED_TO_FROZEN_REPRESENTATION_PROBE")
""",
    )

    assert _find_dormant_surface_offenders(tmp_path) == []


def test_dormant_surface_scan_checks_root_and_package_entry_points(tmp_path: Path) -> None:
    _write_text(
        tmp_path,
        "pyproject.toml",
        """\
[project]
name = "root"
version = "0.1.0"

[project.scripts]
FROZEN-REPRESENTATION = "safe.cli:main"
""",
    )
    _write_text(
        tmp_path,
        "packages/example/pyproject.toml",
        """\
[project]
name = "example"
version = "0.1.0"

[project.entry-points."openalpha.commands"]
run_frp = "openalpha_bridge.representation_probe.worker:main"
""",
    )

    assert _find_dormant_surface_offenders(tmp_path) == [
        "packages/example/pyproject.toml",
        "pyproject.toml",
    ]


def test_non_entry_point_pyproject_mentions_are_not_execution_surfaces(tmp_path: Path) -> None:
    _write_text(
        tmp_path,
        "pyproject.toml",
        """\
[project]
name = "root"
version = "0.1.0"
description = "Documentation about the frozen-representation study"
""",
    )

    assert _find_dormant_surface_offenders(tmp_path) == []


def test_frozen_representation_has_no_execution_surface() -> None:
    assert _find_dormant_surface_offenders(ROOT) == []
