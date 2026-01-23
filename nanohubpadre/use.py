"""
Environment module loader for nanoHUB tools.

This module provides functionality to load nanoHUB environment modules (similar to
the Unix 'module' or 'use' commands) directly from Python. It modifies the current
process environment by reading module configuration files and setting/prepending
environment variables like PATH, LD_LIBRARY_PATH, etc.

The module supports three directives from configuration files:
- setenv: Set an environment variable to a value
- prepend: Prepend a value to an existing environment variable (colon-separated)
- use: Recursively load another module

This is particularly useful for loading the PADRE simulator environment in
Jupyter notebooks or Python scripts running on nanoHUB.

Example
-------
>>> from nanohubpadre import use, load_padre
>>> # Load any nanoHUB module
>>> use("padre-2.4E-r15")
>>> # Or use the convenience function for PADRE
>>> load_padre()  # Loads default version

In Jupyter/IPython, you can also use the magic command:
>>> %use padre-2.4E-r15
"""

import sys
import os
import subprocess
from string import Template
from typing import Dict, List, Optional

# Try to import IPython magic registration (optional dependency)
try:
    from IPython.core.magic import register_line_magic
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Search paths for environment module configuration files
# These are read from the ENVIRON_CONFIG_DIRS environment variable
try:
    EPATH: List[str] = os.environ.get('ENVIRON_CONFIG_DIRS', '').split()
except Exception:
    EPATH = []

# Dictionary to store variable substitutions during module loading
# Used for $VAR substitution in module configuration files
_substitutions: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

def _expand_shell_value(value: str) -> str:
    """
    Expand shell variables and expressions in a value using bash.

    Parameters
    ----------
    value : str
        The value potentially containing shell expressions

    Returns
    -------
    str
        The expanded value, or original value if expansion fails
    """
    try:
        result = subprocess.run(
            ['/bin/bash', '-c', f'echo {value}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return value


def _set_substitution(name: str, value: str) -> None:
    """
    Store a variable substitution for later use in module loading.

    This handles $VAR style substitutions in module configuration files.

    Parameters
    ----------
    name : str
        Variable name
    value : str
        Variable value (may contain $VAR references to substitute)
    """
    global _substitutions
    # Apply existing substitutions to the value
    expanded = Template(value).safe_substitute(_substitutions)
    _substitutions[name] = expanded


def _setenv(args: List[str]) -> None:
    """
    Set an environment variable.

    Implements the 'setenv' directive from module configuration files.
    Format: setenv NAME value [more values...]

    Parameters
    ----------
    args : List[str]
        List where first element is variable name, rest is the value
    """
    if not args:
        return

    name = args[0]
    value = ' '.join(args[1:]) if len(args) > 1 else ''

    # Expand shell expressions in the value
    expanded_value = _expand_shell_value(value)

    # Set the environment variable
    os.environ[name] = expanded_value

    # Also store for substitution in subsequent directives
    _set_substitution(name, expanded_value)


def _prepend(args: List[str]) -> None:
    """
    Prepend a value to an environment variable.

    Implements the 'prepend' directive from module configuration files.
    Format: prepend NAME value

    The value is prepended to the existing variable with a colon separator.
    If the variable doesn't exist, it's set to just the value.

    Special handling for PYTHONPATH: also updates sys.path.

    Parameters
    ----------
    args : List[str]
        List of [variable_name, value]
    """
    global _substitutions

    if len(args) < 2:
        return

    name, value = args[0], args[1]

    # Apply substitutions to the value
    value = Template(value).safe_substitute(_substitutions)

    # Expand shell expressions
    value = _expand_shell_value(value)

    # Prepend to existing value (colon-separated) or set new
    if name in os.environ:
        os.environ[name] = f'{value}:{os.environ[name]}'
    else:
        os.environ[name] = value

    # Special handling for PYTHONPATH: update sys.path
    if name == 'PYTHONPATH':
        for path in reversed(value.split(':')):
            if path and path not in sys.path:
                sys.path.insert(1, path)


def _use(name: str) -> None:
    """
    Load an environment module by name.

    Searches for the module configuration file in ENVIRON_CONFIG_DIRS,
    then processes its directives to modify the environment.

    Parameters
    ----------
    name : str
        Module name (e.g., "padre-2.4E-r15")

    Raises
    ------
    ValueError
        If the module cannot be found in any search path
    RuntimeError
        If ENVIRON_CONFIG_DIRS is not set or empty
    """
    if not EPATH:
        raise RuntimeError(
            "ENVIRON_CONFIG_DIRS environment variable is not set. "
            "This function is designed to run on nanoHUB where environment "
            "modules are configured. Set ENVIRON_CONFIG_DIRS to the directory "
            "containing module configuration files."
        )

    # Search for the module configuration file
    module_path: Optional[str] = None
    for search_dir in EPATH:
        candidate = os.path.join(search_dir, name)
        if os.path.isfile(candidate):
            module_path = candidate
            break

    if module_path is None:
        available_hint = ""
        # Try to list available modules for better error message
        try:
            if EPATH:
                available = []
                for search_dir in EPATH:
                    if os.path.isdir(search_dir):
                        available.extend(os.listdir(search_dir))
                padre_modules = [m for m in available if 'padre' in m.lower()]
                if padre_modules:
                    available_hint = f" Available PADRE modules: {', '.join(sorted(set(padre_modules)))}"
        except Exception:
            pass

        raise ValueError(
            f"Could not find module '{name}' in search paths: {EPATH}.{available_hint}"
        )

    # Parse and execute the module configuration file
    with open(module_path, 'r') as fp:
        for line in fp:
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            tokens = stripped.split()
            directive = tokens[0].lower()

            if directive == 'prepend':
                _prepend(tokens[1:])
            elif directive == 'setenv':
                _setenv(tokens[1:])
            elif directive == 'use':
                # Recursively load another module
                if len(tokens) > 1:
                    _use(tokens[-1])
            elif '=' in line:
                # Handle simple variable assignments: VAR=value
                parts = line.split('=', 1)
                if len(parts) == 2:
                    _set_substitution(parts[0].strip(), parts[1].strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def use(name: str) -> None:
    """
    Load a nanoHUB environment module.

    This function modifies the current process environment by reading module
    configuration files and setting/prepending environment variables. Use this
    to load tools like the PADRE simulator into your PATH.

    Parameters
    ----------
    name : str
        The module name to load (e.g., "padre-2.4E-r15")

    Raises
    ------
    ValueError
        If the module cannot be found
    RuntimeError
        If ENVIRON_CONFIG_DIRS is not configured

    Example
    -------
    >>> from nanohubpadre import use
    >>> use("padre-2.4E-r15")
    >>> # Now 'padre' executable is available in PATH
    >>> import subprocess
    >>> subprocess.run(["padre", "input.inp"])
    """
    _use(name)


def load_padre(version: str = "2.4E-r15") -> None:
    """
    Load the PADRE simulator environment.

    Convenience function to load PADRE into the current environment.
    This modifies PATH and other environment variables so the PADRE
    executable can be found by sim.run().

    Parameters
    ----------
    version : str, optional
        PADRE version to load (default: "2.4E-r15")

    Raises
    ------
    ValueError
        If the specified PADRE version cannot be found
    RuntimeError
        If not running on nanoHUB (ENVIRON_CONFIG_DIRS not set)

    Example
    -------
    >>> from nanohubpadre import load_padre, create_mosfet
    >>> load_padre()  # Load default PADRE version
    >>> sim = create_mosfet()
    >>> sim.add_solve(Solve(initial=True))
    >>> result = sim.run()  # Will find 'padre' in PATH
    """
    module_name = f"padre-{version}"
    _use(module_name)


def list_available_modules(pattern: Optional[str] = None) -> List[str]:
    """
    List available environment modules.

    Parameters
    ----------
    pattern : str, optional
        Filter modules containing this pattern (case-insensitive)

    Returns
    -------
    List[str]
        List of available module names

    Example
    -------
    >>> from nanohubpadre import list_available_modules
    >>> padre_modules = list_available_modules("padre")
    >>> print(padre_modules)
    ['padre-2.4E-r15', 'padre-2.4E-r14', ...]
    """
    available = []
    for search_dir in EPATH:
        if os.path.isdir(search_dir):
            try:
                for name in os.listdir(search_dir):
                    if pattern is None or pattern.lower() in name.lower():
                        available.append(name)
            except OSError:
                pass
    return sorted(set(available))


# ---------------------------------------------------------------------------
# IPython/Jupyter magic command registration
# ---------------------------------------------------------------------------

# Register the %use magic command if running in IPython/Jupyter
try:
    # Check if we're in an IPython environment
    _ipython = get_ipython()  # noqa: F821 - defined in IPython

    if _IPYTHON_AVAILABLE:
        @register_line_magic
        def use_magic(line: str) -> None:
            """
            IPython magic to load environment modules.

            Usage: %use module_name

            Example:
                %use padre-2.4E-r15
            """
            module_name = line.strip()
            if module_name:
                _use(module_name)
            else:
                print("Usage: %use <module_name>")
                print("Example: %use padre-2.4E-r15")

        # Register with the preferred name '%use'
        _ipython.register_magic_function(use_magic, magic_name='use')

except NameError:
    # Not running in IPython - this is fine, magic just won't be available
    pass
except Exception:
    # Some other error during registration - silently ignore
    pass
