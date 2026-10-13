"""Unqualified call matching multiple definitions."""

# mypy: ignore-errors


def ambiguous_caller() -> None:
    dup()
