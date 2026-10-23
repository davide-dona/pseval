from collections.abc import Mapping, Sequence
from types import MappingProxyType

# The vocabulary entry an activity unseen in the train split is scored as.
UNK = 'UNK'

# Start of the Unicode private use area, where activity codes are drawn from.
_FIRST_CODE = 0xE000

# Boundaries the sequence metrics pad spelled suffixes with, distinct and below every activity
# code.
START_CODE = '\x02'
END_CODE = '\x03'


class ActivityCodes:
    """Spell activity names as one character each, `chr(0xE000 + i)` for vocabulary entry `i`.

    A suffix is then one string, which is what the sequence metrics compare and what the
    generations file stores.
    """

    def __init__(self, vocabulary: Sequence[str]) -> None:
        """
        Args:
            vocabulary: The benchmark's activity names in order, `UNK` among them.
        """
        self._codes = {name: chr(_FIRST_CODE + index) for index, name in enumerate(vocabulary)}

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """The names in code order."""
        return tuple(self._codes)

    @property
    def codes(self) -> Mapping[str, str]:
        """Each name to its character, read-only."""
        return MappingProxyType(self._codes)

    def spell(self, names: Sequence[str], *, unseen_as_unk: bool = False) -> str:
        """Spell a sequence of activity names.

        Args:
            names: The names, in order.
            unseen_as_unk: Spell a name outside the vocabulary as `UNK`, as an observed event of
                a later split is scored, instead of refusing it.

        Raises:
            KeyError: If a name is outside the vocabulary and `unseen_as_unk` is false.
        """
        if unseen_as_unk:
            unknown = self._codes[UNK]
            return ''.join(self._codes.get(name, unknown) for name in names)
        return ''.join(self._codes[name] for name in names)
