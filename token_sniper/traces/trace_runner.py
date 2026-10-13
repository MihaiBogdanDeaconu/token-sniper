from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TraceRunConfig:
    instance_id: str
    repo: str
    base_commit: str
    fail_to_pass_tests: list[str]
    gold_patch: str
    test_patch: str | None
    ssg_path: Path
    output_path: Path
    repo_checkout: Path
    python_executable: str = field(default_factory=lambda: sys.executable)
    timeout_seconds: int = 1800
    pytest_args: list[str] = field(default_factory=lambda: ["-p", "no:xdist", "-x", "--no-header"])


def _apply_patch(repo_dir: Path, patch_text: str) -> None:
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=repo_dir,
        input=patch_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git apply failed: {proc.stderr.strip()}")


def _checkout_commit(repo_dir: Path, commit: str) -> None:
    subprocess.run(["git", "reset", "--hard"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "clean", "-fdx", "-e", "*.egg-info"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "checkout", commit], cwd=repo_dir, check=True, capture_output=True)


_CHILD_RUNNER_TEMPLATE = """
import json, sys, os, traceback, re, subprocess, glob
from pathlib import Path

# Python 3.12 removed distutils; setuptools provides a shim.
try:
    import setuptools  # noqa: F401  -- activates distutils compatibility
except ImportError:
    pass

# Python 3.10+ moved ABCs from collections to collections.abc.
# Python 3.12 removed the aliases entirely. Old repos still use them.
import collections, collections.abc
for _abc_name in ('Mapping', 'MutableMapping', 'Sequence', 'MutableSequence',
                  'Callable', 'Iterable', 'Iterator', 'Set', 'MutableSet',
                  'Hashable', 'Sized', 'Container', 'OrderedDict',
                  'defaultdict', 'deque'):
    if not hasattr(collections, _abc_name) and hasattr(collections.abc, _abc_name):
        setattr(collections, _abc_name, getattr(collections.abc, _abc_name))

# Numpy 1.24+ removed deprecated aliases (np.int, np.float, etc.)
# Numpy 2.0 removed np.unicode_, np.str_, np.bool_, etc.
# Old repos (astropy 4.x, scikit-learn, sympy, xarray) still use them.
try:
    import numpy as _np
    for _alias, _builtin in [('int', int), ('float', float), ('complex', complex),
                              ('object', object), ('bool', bool), ('str', str),
                              ('long', int), ('unicode', str),
                              ('unicode_', str), ('string_', bytes),
                              ('int0', _np.intp if hasattr(_np, 'intp') else int),
                              ('uint0', _np.uintp if hasattr(_np, 'uintp') else int)]:
        if not hasattr(_np, _alias):
            setattr(_np, _alias, _builtin)
except ImportError:
    pass

PROJECT_ROOT = Path({project_root!r}).resolve()
OUTPUT_PATH = Path({output_path!r})
TRACER_PKG_PATH = {tracer_pkg_path!r}
TEST_IDS = {test_ids!r}
PYTEST_ARGS = {pytest_args!r}
MODE = {mode!r}

sys.path.insert(0, TRACER_PKG_PATH)
from token_sniper.traces.tracer import CallEdgeTracer  # noqa: E402

tracer = CallEdgeTracer(project_root=PROJECT_ROOT)
rc = 1
err = ""

def _run_pytest(test_ids, pytest_args):
    import pytest
    return pytest.main(list(pytest_args) + list(test_ids))

def _run_django(test_ids):
    \"\"\"Run Django tests using the same setup as Django's runtests.py.

    Replicates the INSTALLED_APPS configuration from Django's tests/runtests.py
    so that model imports during test loading succeed.
    \"\"\"
    test_labels = []
    for tid in test_ids:
        m = re.match(r'^(\\S+)\\s+\\(([^)]+)\\)$', tid)
        if m:
            test_method, dotted_path = m.group(1), m.group(2)
            test_labels.append(f"{{dotted_path}}.{{test_method}}")
        elif '.' in tid and ' ' not in tid:
            test_labels.append(tid)
        else:
            test_labels.append(tid)

    tests_dir = str(PROJECT_ROOT / 'tests')
    os.chdir(tests_dir)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)

    # Derive test module names from labels (e.g. "auth_tests.test_validators.Foo" -> "auth_tests")
    test_modules = set()
    for label in test_labels:
        parts = label.split('.')
        if parts:
            test_modules.add(parts[0])

    ALWAYS_INSTALLED_APPS = [
        "django.contrib.contenttypes",
        "django.contrib.auth",
        "django.contrib.sites",
        "django.contrib.sessions",
        "django.contrib.messages",
        "django.contrib.admin.apps.SimpleAdminConfig",
        "django.contrib.staticfiles",
    ]

    # Remove DJANGO_SETTINGS_MODULE so settings.configure() takes precedence
    os.environ.pop('DJANGO_SETTINGS_MODULE', None)

    from django.conf import settings
    settings.configure(
        DATABASES={{
            "default": {{
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }},
        }},
        INSTALLED_APPS=ALWAYS_INSTALLED_APPS + sorted(test_modules),
        ROOT_URLCONF="urls",
        DEFAULT_AUTO_FIELD="django.db.models.AutoField",
        USE_TZ=False,
        SECRET_KEY="test-secret-key-for-tracing",
        PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"],
    )

    import django
    django.setup()

    from django.test.runner import DiscoverRunner
    runner = DiscoverRunner(verbosity=0, parallel=1)
    suite = runner.build_suite(test_labels)
    result = runner.run_suite(suite)
    return 0 if result.wasSuccessful() else 1

def _run_sympy(test_ids, pytest_args):
    \"\"\"Find sympy test files by function name, then run via pytest.
    NOTE: uses grep — assumes Linux (trace collection runs on K8s).\"\"\"
    import pytest
    converted = []
    keyword_fallbacks = []
    for tid in test_ids:
        if '::' in tid or '/' in tid or tid.endswith('.py'):
            converted.append(tid)
            continue
        result = subprocess.run(
            ['grep', '-rl', f'def {{tid}}', str(PROJECT_ROOT / 'sympy')],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        files = [f.strip() for f in result.stdout.strip().split('\\n') if f.strip()]
        if files:
            rel_path = str(Path(files[0]).relative_to(PROJECT_ROOT))
            converted.append(f"{{rel_path}}::{{tid}}")
        else:
            keyword_fallbacks.append(tid)
    if keyword_fallbacks and not converted:
        return pytest.main(list(pytest_args) + ['sympy/', '-k', ' or '.join(keyword_fallbacks)])
    if keyword_fallbacks:
        converted.extend(['-k', ' or '.join(keyword_fallbacks)])
    return pytest.main(list(pytest_args) + converted)

try:
    with tracer.trace():
        if MODE == "django":
            rc = _run_django(TEST_IDS)
        elif MODE == "sympy":
            rc = _run_sympy(TEST_IDS, PYTEST_ARGS)
        else:
            rc = _run_pytest(TEST_IDS, PYTEST_ARGS)
except Exception:
    err = traceback.format_exc()

payload = {{
    "executed_nodes": sorted(tracer.result.executed_nodes),
    "edges": [list(e) for e in tracer.result.edges],
    "call_events": tracer.result.call_events,
    "dropped_out_of_repo": tracer.result.dropped_out_of_repo,
    "pytest_returncode": int(rc),
    "error": err,
}}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
"""


_EXTRA_DEPS: dict[str, list[str]] = {
    "astropy/astropy": ["hypothesis", "pytest-astropy", "pytest-xdist",
                        "scipy", "setuptools", "pyerfa"],
    "django/django": ["pytz", "asgiref", "sqlparse", "setuptools"],
    "matplotlib/matplotlib": ["numpy<2", "setuptools", "setuptools-scm",
                              "certifi", "cycler", "fonttools", "kiwisolver",
                              "packaging", "pillow", "pyparsing",
                              "python-dateutil"],
    "psf/requests": ["urllib3", "chardet", "certifi", "idna", "setuptools"],
    "pylint-dev/pylint": ["astroid", "isort", "mccabe", "toml", "tomlkit",
                          "setuptools"],
    "pytest-dev/pytest": ["attrs", "pluggy", "py", "packaging", "setuptools",
                          "atomicwrites", "iniconfig", "tomli"],
    "sphinx-doc/sphinx": ["docutils", "Jinja2", "alabaster", "imagesize",
                          "packaging", "snowballstemmer", "babel",
                          "sphinxcontrib-applehelp", "sphinxcontrib-devhelp",
                          "sphinxcontrib-jsmath", "sphinxcontrib-htmlhelp",
                          "sphinxcontrib-serializinghtml", "sphinxcontrib-qthelp",
                          "setuptools"],
    "scikit-learn/scikit-learn": ["scipy", "joblib", "threadpoolctl",
                                  "setuptools", "Cython", "numpy<2"],
    "pydata/xarray": ["pandas", "setuptools", "numpy<2"],
    "sympy/sympy": ["setuptools", "mpmath"],
}

_REPO_TEST_MODE: dict[str, str] = {
    "django/django": "django",
    "sympy/sympy": "sympy",
}

_extra_deps_installed: set[str] = set()


def _ensure_extra_deps(repo: str, python_exe: str, repo_dir: Path | None = None) -> None:
    if repo in _extra_deps_installed:
        return
    extra = _EXTRA_DEPS.get(repo, [])
    if extra:
        logger.info("[%s] Installing extra deps: %s", repo, extra)
        subprocess.run(
            [python_exe, "-m", "pip", "install", "--quiet"] + extra,
            capture_output=True, check=False,
        )

    if repo_dir is not None:
        _install_repo_requirements(repo, python_exe, repo_dir)

    _extra_deps_installed.add(repo)


def _install_repo_requirements(repo: str, python_exe: str, repo_dir: Path) -> None:
    req_files = [
        repo_dir / "requirements.txt",
        repo_dir / "requirements" / "testing.txt",
        repo_dir / "requirements" / "test.txt",
    ]
    for req_file in req_files:
        if req_file.exists():
            logger.info("[%s] Installing from %s", repo, req_file.name)
            subprocess.run(
                [python_exe, "-m", "pip", "install", "--quiet",
                 "-r", str(req_file)],
                capture_output=True, check=False,
                timeout=300,
            )


def _build_pythonpath(repo: str, repo_dir: Path) -> str:
    paths = [str(repo_dir)]

    if repo == "django/django":
        paths.append(str(repo_dir / "tests"))

    src_dir = repo_dir / "src"
    if src_dir.is_dir():
        paths.append(str(src_dir))

    if repo == "matplotlib/matplotlib":
        paths.append(str(repo_dir / "lib"))

    return os.pathsep.join(paths)


_C_EXT_REPOS: dict[str, str] = {
    "scikit-learn/scikit-learn": "sklearn",
    "astropy/astropy": "astropy",
    "matplotlib/matplotlib": "matplotlib",
}

_RELAXED_CFLAGS = (
    "-Wno-error=incompatible-pointer-types "
    "-Wno-error=int-conversion "
    "-Wno-error=implicit-function-declaration "
    "-Wno-error"
)

_IMP_SHIM = '''\
"""Compatibility shim: the imp module was removed in Python 3.12."""
import importlib, importlib.machinery, importlib.util, types

reload = importlib.reload
PY_SOURCE, PY_COMPILED, C_EXTENSION = 1, 2, 3
PKG_DIRECTORY, C_BUILTIN, PY_FROZEN = 5, 6, 7

def find_module(name, path=None):
    spec = importlib.util.find_spec(name, path)
    if spec is None:
        raise ImportError(f"No module named {name!r}")
    return (None, spec.origin, ("", "", PY_SOURCE))

def load_module(name, file=None, pathname=None, description=None):
    return importlib.import_module(name)

def new_module(name):
    return types.ModuleType(name)

def lock_held():
    return False

def acquire_lock():
    pass

def release_lock():
    pass

class NullImporter:
    def __init__(self, path):
        raise ImportError

def get_suffixes():
    exts = []
    for s in importlib.machinery.EXTENSION_SUFFIXES:
        exts.append((s, "rb", C_EXTENSION))
    for s in importlib.machinery.SOURCE_SUFFIXES:
        exts.append((s, "r", PY_SOURCE))
    for s in importlib.machinery.BYTECODE_SUFFIXES:
        exts.append((s, "rb", PY_COMPILED))
    return exts
'''


def _ensure_imp_shim(python_exe: str) -> None:
    check = subprocess.run(
        [python_exe, "-c", "import imp"],
        capture_output=True, check=False,
    )
    if check.returncode == 0:
        return
    site_pkg = subprocess.run(
        [python_exe, "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if not site_pkg:
        return
    imp_path = Path(site_pkg) / "imp.py"
    imp_path.write_text(_IMP_SHIM)
    logger.info("Installed imp shim at %s", imp_path)


def _generate_pytest_version_stub(repo_dir: Path) -> None:
    version_file = repo_dir / "src" / "_pytest" / "_version.py"
    if version_file.exists():
        return
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version = "0.0.0+traced"
    setup_cfg = repo_dir / "setup.cfg"
    if setup_cfg.exists():
        for line in setup_cfg.read_text(errors="replace").splitlines():
            if line.strip().startswith("version"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    version = parts[1].strip() or version
                    break
    version_file.write_text(
        f'version = "{version}"\n'
        f'__version__ = version\n'
    )
    logger.info("[pytest-dev] Generated version stub: %s", version_file)


def _prepare_repo_for_trace(repo: str, repo_dir: Path, python_exe: str) -> None:
    if repo == "pytest-dev/pytest":
        _generate_pytest_version_stub(repo_dir)

    pkg = _C_EXT_REPOS.get(repo)
    if pkg is None:
        return

    _ensure_imp_shim(python_exe)

    check_pythonpath = _build_pythonpath(repo, repo_dir)
    check = subprocess.run(
        [python_exe, "-c", f"import {pkg}"],
        capture_output=True, check=False,
        env={**os.environ, "PYTHONPATH": check_pythonpath},
    )
    if check.returncode != 0:
        build_env = {**os.environ, "CFLAGS": _RELAXED_CFLAGS}
        logger.info("[%s] Building with pip install -e . (C extensions for %s)...", repo, pkg)
        proc = subprocess.run(
            [python_exe, "-m", "pip", "install", "-e", ".", "--quiet", "--no-build-isolation"],
            cwd=repo_dir, capture_output=True, text=True, check=False,
            timeout=600, env=build_env,
        )
        if proc.returncode != 0:
            logger.warning("[%s] pip install -e . failed (rc=%d), trying without --no-build-isolation...",
                           repo, proc.returncode)
            proc2 = subprocess.run(
                [python_exe, "-m", "pip", "install", "-e", ".", "--quiet"],
                cwd=repo_dir, capture_output=True, text=True, check=False,
                timeout=600, env=build_env,
            )
            if proc2.returncode != 0:
                logger.warning("[%s] pip install -e . also failed, trying setup.py build_ext...",
                               repo)
                proc3 = subprocess.run(
                    [python_exe, "setup.py", "build_ext", "--inplace"],
                    cwd=repo_dir, capture_output=True, text=True, check=False,
                    timeout=600, env=build_env,
                )
                if proc3.returncode != 0:
                    logger.warning("[%s] All C extension build attempts failed: %s",
                                   repo, proc3.stderr[:300])


def run_trace_for_task(cfg: TraceRunConfig) -> dict[str, Any]:
    repo_dir = cfg.repo_checkout.resolve()
    if not (repo_dir / ".git").exists():
        raise FileNotFoundError(f"{repo_dir} is not a git checkout")

    _checkout_commit(repo_dir, cfg.base_commit)
    _apply_patch(repo_dir, cfg.gold_patch)
    if cfg.test_patch:
        _apply_patch(repo_dir, cfg.test_patch)

    _ensure_extra_deps(cfg.repo, cfg.python_executable, repo_dir)

    _prepare_repo_for_trace(cfg.repo, repo_dir, cfg.python_executable)

    mode = _REPO_TEST_MODE.get(cfg.repo, "pytest")

    tracer_pkg_path = str(Path(__file__).resolve().parents[2])

    output_path_abs = cfg.output_path.resolve()
    runner_file = output_path_abs.with_suffix(".runner.py")
    runner_file.parent.mkdir(parents=True, exist_ok=True)

    runner_script = _CHILD_RUNNER_TEMPLATE.format(
        project_root=str(repo_dir),
        output_path=str(output_path_abs),
        tracer_pkg_path=tracer_pkg_path,
        test_ids=cfg.fail_to_pass_tests,
        pytest_args=cfg.pytest_args,
        mode=mode,
    )
    runner_file.write_text(runner_script, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    repo_pythonpath = _build_pythonpath(cfg.repo, repo_dir)
    existing = env.get("PYTHONPATH", "")

    env["PYTHONPATH"] = f"{repo_pythonpath}{os.pathsep}{existing}" if existing else repo_pythonpath

    logger.info("[%s] Running tracer subprocess (mode=%s)", cfg.instance_id, mode)
    proc = subprocess.run(
        [cfg.python_executable, str(runner_file)],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=cfg.timeout_seconds,
        check=False,
    )
    stdout_log = output_path_abs.with_suffix(".stdout.log")
    stderr_log = output_path_abs.with_suffix(".stderr.log")
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")

    if not output_path_abs.exists():
        raise RuntimeError(
            f"[{cfg.instance_id}] trace file not produced. "
            f"stderr={stderr_log}, stdout={stdout_log}"
        )

    payload = json.loads(output_path_abs.read_text(encoding="utf-8"))
    logger.info(
        "[%s] trace: rc=%s edges=%d nodes=%d mode=%s",
        cfg.instance_id,
        payload.get("pytest_returncode"),
        len(payload.get("edges", [])),
        len(payload.get("executed_nodes", [])),
        mode,
    )
    return payload
