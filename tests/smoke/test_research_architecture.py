import ast
import hashlib
import re
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pytest

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
_RETIRED_OPERATIONAL_DOCS = frozenset(
    {
        "docs/BRIDGE_CLOUD_API.md",
        "docs/BRIDGE_CLOUD_ARCHITECTURE.md",
        "docs/BRIDGE_CLOUD_STORAGE.md",
        "docs/BRIDGE_GPU_RUNBOOK.md",
        "docs/BRIDGE_MATHEMATICAL_REPRESENTATION.md",
        "docs/BRIDGE_MODAL_DEPLOYMENT.md",
        "docs/BRIDGE_NUMERICAL_CONTRACT.md",
        "docs/BRIDGE_PHASE2_EXECUTION.md",
        "docs/BRIDGE_TEST_OPENING_POLICY.md",
        "docs/OPENALPHA_KRONOS_BRIDGE.md",
    }
)
_RETIRED_LIVE_DOC_IDENTIFIERS = (
    "packages/bridge",
    "openalpha_bridge",
    "bridge_phase2_app.py",
)
_ATX_HEADING = re.compile(
    r"^(?P<marks>#{1,6})[ \t]+(?P<title>.*?)(?:[ \t]+#+)?[ \t]*$"
)
_HISTORICAL_SECTION_HEADING = re.compile(
    r"\b(?:historical|retired|archive|archived|obsolete)\b", re.IGNORECASE
)
_CURRENT_IMPLEMENTATION_HEADING = re.compile(
    r"\b(?:implementation|next|scope)\b", re.IGNORECASE
)
_LIVE_BRIDGE_PHASE2_CLAIMS = (
    re.compile(
        r"\bbridge[\W_]*phase[\W_]*2\b.{0,160}"
        r"\b(?:is|remains|continues?\s+to\s+be)\s+(?:an?\s+)?"
        r"(?:active|available|deployed|executable|implemented|live|operational)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbridge[\W_]*phase[\W_]*2\b.{0,160}"
        r"\b(?:can\s+be\s+)?(?:deployed|executed|exposed|provided|run|served|supported)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbridge[\W_]*phase[\W_]*2\b.{0,80}"
        r"\bhas\s+(?:an?\s+)?(?:api|cli|control[\W_]+plane|endpoint|service|worker)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?bridge\s+(?:phase\s+2\s+)?"
        r"(?:adapter|api|checkpoint|cli|control[\W_]+plane|service|worker)\b.{0,80}\b"
        r"(?:can(?!\s+(?:never|no|not)\b)(?!\s+be\s+(?:absent|inactive|not\s+authorized|retired|stopped|unavailable)\b)|"
        r"executes?|must(?!\s+not\b)|provides?|produces?|runs?|supports?|will\b|"
        r"has(?!\s+(?:been\s+(?:removed|retired)|never|no|not)\b)|"
        r"is\s+(?:an?\s+)?(?:active|available|deployed|executable|implemented|live|loadable|"
        r"operational|service|system|worker))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:active|deployed|executable|live|loadable|operational)\s+"
        r"bridge\s+(?:adapter|api|checkpoint|cli|service|worker)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bbridge\s+produces?\b", re.IGNORECASE),
    re.compile(
        r"\bbridge(?:[\W_]*(?:2k|base))?\b.{0,100}"
        r"\b(?:is\s+prohibited\s+until|must\s+pass|passes)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:modal\s+(?:deploy|run)|python\s+-m)\b.{0,120}"
        r"\bbridge[\W_]*(?:phase[\W_]*2)?\b",
        re.IGNORECASE,
    ),
)
_LIVE_IMPLEMENTATION_SCOPE_DIRECTIVE = re.compile(
    r"\b(?:implement\s+phase\s+1\b|"
    r"(?:implement|add|build|deploy|run|fetch|materialize|apply|generate|create|produce)\b"
    r".{0,120}\b(?:adapter|bridge|checkpoint|corpus|head|holdout|kronos(?:\s+requests?)?)\b)",
    re.IGNORECASE,
)
_MARKDOWN_LINK = re.compile(
    r"!?\[[^\]\n]*\]\(\s*"
    r"(?:<(?P<angle>[^>\n]+)>|(?P<plain>[^\s()]+))"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^\)\n]*\)))?\s*\)"
)
_EXTERNAL_LINK_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
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


def _current_document_paths(root: Path) -> tuple[Path, ...]:
    package_readmes = sorted((root / "packages").glob("*/README.md"))
    return (
        root / "README.md",
        root / "docs" / "ARCHITECTURE.md",
        root / "docs" / "DATA_POLICY.md",
        root / "docs" / "KRONOS_COMPATIBILITY_BOUNDARY.md",
        root / "docs" / "MASTER_PLAN.md",
        root / "docs" / "SENTINEL_DIRECTION.md",
        root / "docs" / "STATUS.md",
        *package_readmes,
    )


def _markdown_atx_sections(text: str) -> list[tuple[str, str, bool, bool]]:
    sections: list[tuple[str, str, bool, bool]] = []
    heading_stack: list[tuple[int, bool, bool]] = []
    current_heading = ""
    current_body: list[str] = []
    current_historical = False
    current_implementation_scope = False
    fence_character = ""
    fence_length = 0

    def flush() -> None:
        if current_heading or current_body:
            sections.append(
                (
                    current_heading,
                    "\n".join(current_body),
                    current_historical,
                    current_implementation_scope,
                )
            )

    for line in text.splitlines():
        stripped = line.lstrip()
        fence_match = re.match(r"(?P<fence>`{3,}|~{3,})", stripped)
        if fence_match:
            fence = fence_match.group("fence")
            if not fence_character:
                fence_character = fence[0]
                fence_length = len(fence)
            elif fence[0] == fence_character and len(fence) >= fence_length:
                fence_character = ""
                fence_length = 0
            current_body.append(line)
            continue
        if fence_character:
            current_body.append(line)
            continue

        heading_match = _ATX_HEADING.match(line)
        if not heading_match:
            current_body.append(line)
            continue

        flush()
        level = len(heading_match.group("marks"))
        title = heading_match.group("title").strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        inherited_historical = heading_stack[-1][1] if heading_stack else False
        inherited_implementation_scope = heading_stack[-1][2] if heading_stack else False
        current_historical = inherited_historical or bool(
            _HISTORICAL_SECTION_HEADING.search(title)
        )
        current_implementation_scope = inherited_implementation_scope or bool(
            _CURRENT_IMPLEMENTATION_HEADING.search(title)
        )
        heading_stack.append(
            (level, current_historical, current_implementation_scope)
        )
        current_heading = title
        current_body = []

    flush()
    return sections

def _live_bridge_phase2_claims(text: str) -> list[str]:
    offenders: set[str] = set()
    for heading, body, historical, implementation_scope in _markdown_atx_sections(text):
        if historical:
            continue
        normalized = re.sub(r"\s+", " ", f"{heading} {body}")
        offenders.update(
            pattern.pattern
            for pattern in _LIVE_BRIDGE_PHASE2_CLAIMS
            if pattern.search(normalized)
        )
        if implementation_scope and _LIVE_IMPLEMENTATION_SCOPE_DIRECTIVE.search(
            normalized
        ):
            offenders.add(_LIVE_IMPLEMENTATION_SCOPE_DIRECTIVE.pattern)
    return sorted(offenders)


def _markdown_without_code(text: str) -> str:
    visible_lines: list[str] = []
    fence_character = ""
    fence_length = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        fence_match = re.match(r"(?P<fence>`{3,}|~{3,})", stripped)
        if fence_match:
            fence = fence_match.group("fence")
            if not fence_character:
                fence_character = fence[0]
                fence_length = len(fence)
            elif fence[0] == fence_character and len(fence) >= fence_length:
                fence_character = ""
                fence_length = 0
            visible_lines.append("")
            continue
        if fence_character:
            visible_lines.append("")
            continue
        visible_lines.append(
            re.sub(r"(?P<ticks>`+)[^`\n]*?(?P=ticks)", "", line)
        )
    return "\n".join(visible_lines)


def _github_heading_base(title: str) -> str:
    title = re.sub(r"!?\[([^\]]+)\]\([^\)]+\)", r"\1", title)
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"[`*_~]", "", title).casefold()
    title = re.sub(r"[^\w\s-]", "", title)
    title = re.sub(r"\s+", "-", title.strip())
    return re.sub(r"-+", "-", title).strip("-")


def _markdown_heading_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    fence_character = ""
    fence_length = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        fence_match = re.match(r"(?P<fence>`{3,}|~{3,})", stripped)
        if fence_match:
            fence = fence_match.group("fence")
            if not fence_character:
                fence_character = fence[0]
                fence_length = len(fence)
            elif fence[0] == fence_character and len(fence) >= fence_length:
                fence_character = ""
                fence_length = 0
            continue
        if fence_character:
            continue
        heading_match = _ATX_HEADING.match(line)
        if not heading_match:
            continue
        base = _github_heading_base(heading_match.group("title"))
        if not base:
            continue
        duplicate_index = counts.get(base, 0)
        counts[base] = duplicate_index + 1
        anchors.add(base if duplicate_index == 0 else f"{base}-{duplicate_index}")
    return anchors


def _broken_current_document_links(root: Path) -> list[str]:
    broken: list[str] = []
    for path in _current_document_paths(root):
        text = path.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(_markdown_without_code(text)):
            target = (match.group("angle") or match.group("plain")).strip()
            if not target or _EXTERNAL_LINK_SCHEME.match(target):
                continue
            encoded_path, has_fragment, encoded_fragment = target.partition("#")
            relative_target = unquote(encoded_path)
            target_path = path if not relative_target else path.parent / relative_target
            problem = not target_path.exists()
            if not problem and has_fragment and encoded_fragment and target_path.is_file():
                fragment = unquote(encoded_fragment).casefold()
                anchors = _markdown_heading_anchors(
                    target_path.read_text(encoding="utf-8")
                )
                problem = fragment not in anchors
            if problem:
                broken.append(f"{path.relative_to(root).as_posix()}: {target}")
    return sorted(broken)

def _current_documentation_offenders(root: Path) -> dict[str, list[str]]:
    offenders: dict[str, list[str]] = {}
    for path in _current_document_paths(root):
        text = path.read_text(encoding="utf-8")
        normalized = text.replace("\\", "/").casefold()
        problems = [
            retired_path
            for retired_path in sorted(_RETIRED_OPERATIONAL_DOCS)
            if any(
                candidate in normalized
                for candidate in (retired_path.casefold(), Path(retired_path).name.casefold())
            )
        ]
        problems.extend(
            identifier
            for identifier in _RETIRED_LIVE_DOC_IDENTIFIERS
            if identifier.casefold() in normalized
        )
        problems.extend(_live_bridge_phase2_claims(text))
        if problems:
            offenders[path.relative_to(root).as_posix()] = problems
    return offenders


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


def _python_imports_namespace(source: str, namespace: str) -> bool:
    tree = ast.parse(source)
    prefix = f"{namespace}."
    importlib_aliases: set[str] = set()
    import_module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == namespace or alias.name.startswith(prefix):
                    return True
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == namespace or module.startswith(prefix):
                return True
            if module == "importlib":
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        target = node.args[0]
        if not isinstance(target, ast.Constant) or not isinstance(target.value, str):
            continue
        if target.value != namespace and not target.value.startswith(prefix):
            continue
        function = node.func
        if isinstance(function, ast.Name) and (
            function.id == "__import__" or function.id in import_module_aliases
        ):
            return True
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and isinstance(function.value, ast.Name)
            and function.value.id in importlib_aliases
        ):
            return True
    return False


@pytest.mark.parametrize(
    "source",
    (
        'import importlib\nimportlib.import_module("openalpha_bridge.phase2")\n',
        'import importlib as il\nil.import_module("openalpha_bridge")\n',
        'from importlib import import_module as load\nload("openalpha_bridge.cloud")\n',
        '__import__("openalpha_bridge.phase2")\n',
    ),
)
def test_python_namespace_scan_catches_dynamic_imports(source: str) -> None:
    assert _python_imports_namespace(source, "openalpha_bridge") is True


@pytest.mark.parametrize(
    "source",
    (
        'import importlib\nimportlib.import_module("openalpha_bridgework")\n',
        'import importlib as il\ntarget = "openalpha_bridge"\nil.import_module(target)\n',
        'from importlib import import_module as load\nload("other_package")\n',
        'loader.import_module("openalpha_bridge")\n',
        'value = "openalpha_bridge.phase2"\n',
    ),
)
def test_python_namespace_scan_ignores_near_names_and_inert_strings(source: str) -> None:
    assert _python_imports_namespace(source, "openalpha_bridge") is False


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


def test_abandoned_bridge_package_is_absent() -> None:
    assert not (ROOT / "packages" / "bridge").exists()


def test_active_python_has_no_openalpha_bridge_imports() -> None:
    namespace = "openalpha_" + "bridge"
    offenders: list[str] = []
    for surface in ("packages", "cloud", "scripts", "tests"):
        root = ROOT / surface
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            relative = path.relative_to(ROOT)
            if _is_ignored(relative):
                continue
            if _python_imports_namespace(path.read_text(encoding="utf-8"), namespace):
                offenders.append(relative.as_posix())
    assert sorted(offenders) == []


def test_research_core_knows_no_research_subject() -> None:
    source_root = ROOT / "packages" / "research-core" / "src" / "openalpha_research"
    forbidden = ("openalpha_kronos", "openalpha_sentinel", "kronos", "sentinel", "bridge")
    offenders = {
        path.relative_to(ROOT).as_posix(): sorted(
            token for token in forbidden if token in path.read_text(encoding="utf-8").casefold()
        )
        for path in source_root.rglob("*.py")
        if any(
            token in path.read_text(encoding="utf-8").casefold() for token in forbidden
        )
    }
    assert offenders == {}


def test_research_core_distribution_has_no_study_dependency() -> None:
    document = tomllib.loads(
        (ROOT / "packages" / "research-core" / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = document["project"]
    dependencies = [*project.get("dependencies", [])]
    for values in project.get("optional-dependencies", {}).values():
        dependencies.extend(values)
    normalized = "\n".join(dependencies).casefold()
    assert not any(subject in normalized for subject in ("kronos", "sentinel", "bridge"))


def test_no_official_protocol_implementation_exists() -> None:
    source_root = ROOT / "packages" / "kronos-research" / "src"
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in source_root.rglob("*")
        if "official_protocol" in path.relative_to(source_root).as_posix().casefold()
    ]
    assert offenders == []


def test_active_ci_has_no_bridge_phase2_identity() -> None:
    identity = re.compile(r"bridge[\W_]*phase[\W_]*2", re.IGNORECASE)
    workflow_root = ROOT / ".github" / "workflows"
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((*workflow_root.glob("*.yml"), *workflow_root.glob("*.yaml")))
        if identity.search(path.name) or identity.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_workspace_metadata_has_no_live_bridge_distribution_or_cli() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8").casefold()
    assert "openalpha-bridge" not in metadata
    assert "bridge-phase2" not in metadata


@pytest.mark.parametrize(
    "text",
    [
        "Bridge Phase 2\nis operational.",
        "Bridge Phase 2 has an API.",
        "Bridge Phase 2 can be deployed.",
        "Bridge Phase 2 continues to be operational.",
        "Bridge Phase 2 is an active service.",
        "The Bridge adapter must preserve token order.",
        "The Bridge adapter can preserve token order.",
        "The Bridge adapter executes production forecasts.",
        "Bridge produces constrained forecasts.",
        "This is a loadable Bridge checkpoint.",
        "Bridge-2K reconstruction feasibility must pass.",
        "## Next justified scope\n\nImplement Phase 1 only.",
        "## Next justified scope\n\nImplement the corpus builder.",
        "## Next justified scope\n\nAdd the causal adapter.",
        "## Next justified scope\n\nGenerate a loadable checkpoint.",
        "## Current work\n\n### Next implementation\n\nImplement the corpus builder.",
        "## Next implementation\n\n### Phase 1\n\nImplement the corpus builder.",
    ],
)
def test_current_doc_guard_rejects_live_system_claims(text: str) -> None:
    assert _live_bridge_phase2_claims(text)


@pytest.mark.parametrize(
    "text",
    [
        "Bridge Phase 2 was retired.",
        "Bridge Phase 2 is inactive.",
        "The Bridge adapter is absent.",
        "The Bridge adapter is not authorized.",
        "The Bridge service was stopped.",
        "The Bridge checkpoint has been retired.",
        "The historical contract required ordered tokens.",
        "Bridge was never trained.",
        "## Historical next justified scope\n\nImplement Phase 1 only.",
        (
            "## Historical Sentinel execution stages (retired)\n\n"
            "### Next implementation\n\nImplement the corpus builder."
        ),
        (
            "# Archive\n\n## Current notes\n\n### Next implementation\n\n"
            "Build the Bridge checkpoint."
        ),
    ],
)
def test_current_doc_guard_allows_explicit_historical_status(text: str) -> None:
    assert _live_bridge_phase2_claims(text) == []


def test_readme_has_no_hardcoded_repository_metrics() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "1,549 passing" not in text
    assert not re.search(
        r"\|\s*(?:Source|Tests)\s*\|[^|\n]*\b\d[\d,]*\s+lines\b",
        text,
        re.IGNORECASE,
    )
    assert not re.search(r"#\s*\d[\d,]*\s+tests\b", text, re.IGNORECASE)


def test_readme_uses_the_approved_research_flow_image() -> None:
    image = ROOT / "docs" / "assets" / "openalpha-research-flow.png"
    assert image.is_file()
    assert image.stat().st_size == 1_419_345
    assert hashlib.sha256(image.read_bytes()).hexdigest() == (
        "4782c0251fe9a21bd6984b4b03b39ffe6d2549cb34fac6b5f14a11afece88929"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    marker = "![OpenAlpha research flow"
    assert marker in readme
    assert readme.index(marker) < readme.index("| Study | Question | Conclusion |")


def test_master_plan_marks_retired_execution_stages_as_historical() -> None:
    text = (ROOT / "docs" / "MASTER_PLAN.md").read_text(encoding="utf-8")
    assert "## Historical Sentinel execution stages (retired)" in text
    for stage in range(1, 6):
        assert re.search(rf"^### Stage {stage}\b", text, re.MULTILINE)
        assert not re.search(rf"^## Stage {stage}\b", text, re.MULTILINE)


def test_sentinel_direction_has_separate_compatible_reconstruction_clause() -> None:
    text = (ROOT / "docs" / "SENTINEL_DIRECTION.md").read_text(encoding="utf-8")
    assert "and every separately labeled compatible reconstruction" in re.sub(
        r"\s+", " ", text
    )


def _write_current_doc_fixture(
    root: Path, readme: str, extra_files: Mapping[str, str | bytes]
) -> None:
    required_docs = (
        "ARCHITECTURE.md",
        "DATA_POLICY.md",
        "KRONOS_COMPATIBILITY_BOUNDARY.md",
        "MASTER_PLAN.md",
        "SENTINEL_DIRECTION.md",
        "STATUS.md",
    )
    (root / "docs").mkdir(parents=True)
    (root / "README.md").write_text(readme, encoding="utf-8")
    for name in required_docs:
        (root / "docs" / name).write_text(f"# {name}\n", encoding="utf-8")
    for relative, content in extra_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")


def test_current_doc_links_accept_commonmark_destinations(tmp_path: Path) -> None:
    readme = """# Overview

[file](docs/guide.md)
![image](assets/pixel.png)
[same document](#overview)
[cross document](docs/guide.md#target-heading)
[first duplicate](docs/guide.md#repeat)
[second duplicate](docs/guide.md#repeat-1)
[with title](docs/guide.md "Guide")
[angle path](<docs/path with spaces.md>)
[encoded path](docs/path%20with%20spaces.md)
[http](http://example.com)
[https](https://example.com)
[email](mailto:research@example.com)
[external scheme](ftp://example.com/archive)
`[code example](missing.md)`
"""
    _write_current_doc_fixture(
        tmp_path,
        readme,
        {
            "docs/guide.md": (
                "# Guide\n\n## Target *Heading*!\n\n## Repeat\n\n## Repeat\n"
            ),
            "docs/path with spaces.md": "# Spaced path\n",
            "assets/pixel.png": b"not-a-real-image",
        },
    )

    assert _broken_current_document_links(tmp_path) == []


def test_current_doc_links_report_missing_files_and_anchors(tmp_path: Path) -> None:
    readme = """# Overview

[missing file](docs/missing.md)
[missing same-document anchor](#missing-anchor)
[missing cross-document anchor](docs/guide.md#missing-anchor)
"""
    _write_current_doc_fixture(
        tmp_path,
        readme,
        {"docs/guide.md": "# Guide\n\n## Existing anchor\n"},
    )

    assert _broken_current_document_links(tmp_path) == [
        "README.md: #missing-anchor",
        "README.md: docs/guide.md#missing-anchor",
        "README.md: docs/missing.md",
    ]


def test_master_plan_states_current_research_direction() -> None:
    text = (ROOT / "docs" / "MASTER_PLAN.md").read_text(encoding="utf-8")
    assert "## Current next direction" in text
    assert "official-protocol replication is the next research direction" in text.casefold()
    assert (
        "protocol must be designed and its preregistration cryptographically sealed "
        "before code is written"
    ) in re.sub(r"\s+", " ", text)
    assert "No official-protocol implementation exists" in text


def test_current_documentation_has_no_retired_bridge_operations() -> None:
    assert _current_documentation_offenders(ROOT) == {}


def test_current_documentation_local_links_exist() -> None:
    assert _broken_current_document_links(ROOT) == []
