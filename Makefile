PYTHON := python
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
PYTEST := $(PYTHON) -m pytest
PRE_COMMIT := $(PYTHON) -m pre_commit
PY_FILES := $(shell git ls-files '*.py')

.PHONY: format lint typecheck test pre-commit ci

format:
	$(RUFF) format $(PY_FILES)

lint:
	$(RUFF) check $(PY_FILES)
	$(RUFF) format --check $(PY_FILES)

typecheck:
	$(MYPY) $(PY_FILES)

test:
	$(PYTEST)

pre-commit:
	$(PRE_COMMIT) run --all-files --show-diff-on-failure

ci: lint typecheck test
