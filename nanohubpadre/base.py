"""
Base classes for PADRE input deck components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class PadreCommand(ABC):
    """Base class for all PADRE input deck commands."""

    # Command name in PADRE syntax (e.g., "MESH", "DOPING", "SOLVE")
    command_name: str = ""

    def __init__(self):
        self._parameters: Dict[str, Any] = {}

    @abstractmethod
    def to_padre(self) -> str:
        """Convert this command to PADRE input deck syntax."""
        pass

    def _format_value(self, value: Any) -> str:
        """Format a Python value for PADRE syntax."""
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        elif isinstance(value, float):
            # Handle zero specially
            if value == 0.0:
                return "0"
            # Use scientific notation for very small/large numbers
            if abs(value) < 1e-3 or abs(value) > 1e6:
                return f"{value:.6e}".replace("e+", "e").replace("e0", "e")
            return str(value)
        elif isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        elif isinstance(value, str):
            return value
        else:
            return str(value)

    def _format_param(self, name: str, value: Any) -> str:
        """Format a parameter as NAME=VALUE."""
        if isinstance(value, bool):
            if value:
                return name.upper()
            return ""  # Don't include false booleans unless explicit
        return f"{name.upper()}={self._format_value(value)}"

    def _build_command(self, params: Dict[str, Any],
                       flags: Optional[List[str]] = None,
                       lowercase: bool = True) -> str:
        """Build a PADRE command string from parameters."""
        cmd_name = self.command_name.lower() if lowercase else self.command_name
        parts = [cmd_name]

        # Add flags (boolean parameters that are just keywords)
        if flags:
            for f in flags:
                if f:
                    parts.append(f.lower() if lowercase else f.upper())

        # Add key=value parameters
        for name, value in params.items():
            if value is not None:
                formatted = self._format_param(name, value, lowercase)
                if formatted:
                    parts.append(formatted)

        # Handle line continuation for long lines
        line = " ".join(parts)
        if len(line) > 72:
            # Split into multiple lines with continuation
            lines = []
            current = cmd_name
            for part in parts[1:]:
                if len(current) + len(part) + 1 > 70:
                    lines.append(current)
                    current = "+     " + part
                else:
                    current += " " + part
            lines.append(current)
            return "\n".join(lines)

        return line

    def _format_param(self, name: str, value: Any, lowercase: bool = True) -> str:
        """Format a parameter as NAME=VALUE."""
        param_name = name.lower() if lowercase else name.upper()
        if isinstance(value, bool):
            if value:
                return param_name
            return ""  # Don't include false booleans unless explicit
        return f"{param_name}={self._format_value(value)}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class Comment(PadreCommand):
    """A comment line in the PADRE input deck."""

    command_name = "$"

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def to_padre(self) -> str:
        return f"$  {self.text}"


class Title(PadreCommand):
    """Title line for PADRE simulation."""

    command_name = "TITLE"

    def __init__(self, title: str):
        super().__init__()
        self.title = title[:60]  # Max 60 characters

    def to_padre(self) -> str:
        return f"  TITLE  {self.title}"
