"""
yaml_config.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24

Loads instructions and tools config files over their bundled defaults.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ValidationError

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_\-]*)\}")
SUPPORTED_CONFIG_VERSIONS = {"1.0"}

ConfigModel = TypeVar("ConfigModel", bound=BaseModel)


class ConfigFileError(ValueError):
    """Raised when an instructions or tools config file is malformed."""


def check_placeholders(text: str, allowed: Set[str], where: str) -> str:
    """Reject {name} placeholders that are not filled in for this text.

    Args:
        text (str, required): Text to check.
        allowed (set, required): Placeholder names filled in at runtime.
        where (str, required): Key name used in the error message.

    Returns:
        str: the unchanged text.

    Raises:
        ValueError: If the text uses a placeholder outside `allowed`.
    """
    unknown = sorted(set(PLACEHOLDER.findall(text)) - allowed)
    if unknown:
        supported = ", ".join(f"{{{name}}}" for name in sorted(allowed)) or "none"
        raise ValueError(f"{where} uses unsupported placeholder(s) {unknown}; supported: {supported}")
    return text


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read a config YAML file into a mapping.

    Args:
        path (Path, required): File to read.

    Returns:
        dict: the parsed mapping, empty for an empty file.

    Raises:
        ConfigFileError: If the file is missing or is not a YAML mapping.
    """
    if not path.is_file():
        raise ConfigFileError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigFileError(f"{path}: config must be a YAML mapping")
    return raw


def _merge_sections(defaults: Dict[str, Any], user: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Overlay the keys of a user file on the defaults, one section at a time.

    Args:
        defaults (dict, required): Bundled default values.
        user (dict, required): Values from the user file.
        path (Path, required): User file, for error messages.

    Returns:
        dict: the merged mapping.

    Raises:
        ConfigFileError: If the user file has an unknown section or a section that is not a mapping.
    """
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in defaults.items()}
    for key, value in user.items():
        if key not in merged:
            raise ConfigFileError(f"{path}: unknown top-level key '{key}'; expected one of {sorted(merged)}")
        if isinstance(merged[key], dict):
            if not isinstance(value, dict):
                raise ConfigFileError(f"{path}: '{key}' must be a mapping")
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def _dotted_keys(raw: Dict[str, Any]) -> Set[str]:
    """List the keys of a config mapping as "section.key" (or "key" at the top level).

    Args:
        raw (dict, required): Parsed config mapping.

    Returns:
        set: the dotted keys.
    """
    keys = set()
    for key, value in raw.items():
        if isinstance(value, dict):
            keys.update(f"{key}.{sub}" for sub in value)
        else:
            keys.add(key)
    return keys


def load_config_file(
    model: Type[ConfigModel], default_path: Path, config_path: Optional[Union[str, Path]] = None
) -> ConfigModel:
    """Load a config file over its bundled defaults, or the defaults alone, and validate it.

    Args:
        model (type, required): Pydantic model of the config, with a `version` and a `user_keys` field.
        default_path (Path, required): Bundled default file.
        config_path (str | Path, optional): User file. Defaults to None (defaults only).

    Returns:
        BaseModel: the validated config, with `user_keys` holding the dotted keys the user file set.

    Raises:
        ConfigFileError: If a file is missing or malformed, or has unknown keys or invalid values.
    """
    raw = _read_yaml(default_path)
    path = default_path
    user_keys: Set[str] = set()
    if config_path is not None:
        path = Path(config_path)
        user = _read_yaml(path)
        raw = _merge_sections(raw, user, path)
        user_keys = _dotted_keys(user)
    version = str(raw.get("version", ""))
    if version not in SUPPORTED_CONFIG_VERSIONS:
        raise ConfigFileError(
            f"{path}: unsupported config version {version!r}; supported: {sorted(SUPPORTED_CONFIG_VERSIONS)}"
        )
    raw["version"] = version
    raw["user_keys"] = user_keys
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors())
        raise ConfigFileError(f"{path}: {problems}") from None
