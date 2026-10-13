"""Entry module with imports, inheritance, and calls."""

from pack.base import Base
from pack.utils import helper
from pack import dup_a

x = 1  # not a call site


class Child(Base):
    """Subclass."""

    def run(self) -> None:
        helper(x)
        self.save()
        dup_a.dup()
