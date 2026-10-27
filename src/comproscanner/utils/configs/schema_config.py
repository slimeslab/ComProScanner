"""
schema_config.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

import copy
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from ..logger import setup_logger

logger = setup_logger("comproscanner.log", module_name="schema_config")

DEFAULT_SCHEMA_PATH = Path(__file__).parent / "default_extraction_schema.yaml"
SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
SCHEMA_DOMAINS = ("composition", "synthesis")

# Safe identifiers only (used as dict keys and model attributes)
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SCALAR_TYPES = {"str", "int", "float", "bool", "None", "object"}
_MISSING = object()

# Comparison rules used by the evaluators
MATCH_MODES = {"exact", "semantic", "item_list", "joined_text", "keyed_values", "numeric"}
# Default thresholds per match mode
DEFAULT_SIMILARITY_THRESHOLDS = {
    "semantic": 0.9,
    "item_list": 0.9,
    "joined_text": 0.8,
    "keyed_values": 0.9,
}
PLOT_KINDS = {"categorical_distribution", "numeric_distribution", "none"}
GRAPH_ROLES = {"node_property", "node_cluster", "joined_node", "skip"}
GRAPH_EDGE_SOURCES = {"paper", "composition"}
# Safe Neo4j labels and relationship types (interpolated into Cypher)
_SAFE_LABEL = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_SAFE_RELATIONSHIP = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SchemaValidationError(ValueError):
    """Raised when an extraction schema file is malformed."""


@dataclass
class ParsedType:
    """Parsed schema type string, e.g. "dict[str, int|float|None]"."""

    kind: str
    args: List["ParsedType"] = field(default_factory=list)

    def contains(self, kind: str) -> bool:
        return self.kind == kind or any(arg.contains(kind) for arg in self.args)

    def is_nullable(self) -> bool:
        return self.kind == "None" or (
            self.kind == "union" and any(a.kind == "None" for a in self.args)
        )


def _split_top_level(text: str, sep: str) -> List[str]:
    parts, depth, current = [], 0, ""
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == sep and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return parts


def parse_type(type_str: str) -> ParsedType:
    """Parse a schema type string into a ParsedType."""
    text = re.sub(r"\s+", "", str(type_str))
    members = _split_top_level(text, "|")
    if len(members) > 1:
        return ParsedType("union", [parse_type(m) for m in members])
    if text in _SCALAR_TYPES:
        return ParsedType(text)
    match = re.fullmatch(r"(list|dict)\[(.*)\]", text)
    if match:
        container, inner = match.groups()
        if container == "list":
            return ParsedType("list", [parse_type(inner)])
        key_and_value = _split_top_level(inner, ",")
        if len(key_and_value) != 2 or key_and_value[0] != "str":
            raise SchemaValidationError(
                f"Unsupported dict type '{type_str}': only dict[str, <value type>] is supported"
            )
        return ParsedType("dict", [parse_type(key_and_value[1])])
    raise SchemaValidationError(f"Unsupported field type '{type_str}'")


@dataclass
class SchemaField:
    """
    One extracted field of the schema.

    Args:
        name (str): Key in the extracted record.
        type (str): Type string, e.g. "str", "list[str]", "dict[str, object]".
        description (str, optional): Guidance shown to the LLM.
        required (bool, optional): Shown as required/optional in the prompt (default: True).
        default (Any, optional): Value used when the field is absent (default: derived from type).
        allowed_values (list, optional): Inline allowed values for the formatter.
        allowed_values_ref (str, optional): Name of a runtime allowed-values list.
        value_fields (list, optional): Sub-fields for "object" / "dict[str, object]" types.
        extraction_only (bool, optional): Dropped by the format crew (default: False).
        format_note (str, optional): Field-specific rule for the format crew.
        example (Any, optional): Example value for the prompt's example JSON.
        variable_example (Any, optional): Example value for the variable-composition example.
        weight (float, optional): Evaluation weight relative to sibling fields (default: equal share).
        match_mode (str, optional): Comparison rule for evaluation (default: derived from type).
        similarity_threshold (float, optional): Matching threshold (default: per match mode).
        similarity_threshold_mode (str, optional): "relative" or "absolute" tolerance for numeric fields (default: "relative").
        plot_kind (str, optional): "categorical_distribution", "numeric_distribution" or "none" (default: "none").
        evaluation_note (str, optional): Field-specific judging rule for the agentic evaluator.
        graph_role (str, optional): Knowledge-graph role of a leaf: "node_property", "node_cluster", "joined_node" or "skip" (default: "node_property").
        graph_label (str, optional): Neo4j label of the field's nodes (default: field name in CamelCase).
        relationship_type (str, optional): Relationship from the parent node (default: derived from the field name).
        graph_edge_sources (dict, optional): For a composition-domain cluster, relationship type per source node ("paper", "composition").
        graph_key_property (str, optional): Property holding a keyed entry's key (default: "key").
        graph_value_property (str, optional): Property holding a keyed entry's scalar value (default: "value").
        graph_default (Any, optional): Graph value used when the field is missing or empty (default: None, not added).
        cluster_similarity_threshold (float, optional): Semantic clustering threshold for node_cluster values (default: 0.85).
    """

    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = _MISSING
    allowed_values: Optional[List[str]] = None
    allowed_values_ref: Optional[str] = None
    value_fields: Optional[List["SchemaField"]] = None
    extraction_only: bool = False
    format_note: Optional[str] = None
    example: Any = None
    variable_example: Any = None
    weight: Optional[float] = None
    match_mode: Optional[str] = None
    similarity_threshold: Optional[float] = None
    similarity_threshold_mode: str = "relative"
    plot_kind: str = "none"
    evaluation_note: Optional[str] = None
    graph_role: str = "node_property"
    graph_label: Optional[str] = None
    relationship_type: Optional[str] = None
    graph_edge_sources: Optional[Dict[str, str]] = None
    graph_key_property: str = "key"
    graph_value_property: str = "value"
    graph_default: Any = None
    cluster_similarity_threshold: float = 0.85
    parsed_type: ParsedType = field(init=False, repr=False)
    effective_weight: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.parsed_type = parse_type(self.type)
        if self.default is _MISSING:
            self.default = self._derive_default()
        if self.match_mode is None:
            self.match_mode = self._derive_match_mode()
        elif self.match_mode not in MATCH_MODES:
            raise SchemaValidationError(
                f"Unknown match_mode '{self.match_mode}'; supported: {sorted(MATCH_MODES)}"
            )
        if self.similarity_threshold is None:
            self.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLDS.get(self.match_mode)
        if self.similarity_threshold_mode not in ("relative", "absolute"):
            raise SchemaValidationError(
                f"similarity_threshold_mode must be 'relative' or 'absolute', got '{self.similarity_threshold_mode}'"
            )
        if self.plot_kind not in PLOT_KINDS:
            raise SchemaValidationError(
                f"Unknown plot_kind '{self.plot_kind}'; supported: {sorted(PLOT_KINDS)}"
            )
        if self.weight is not None and not 0 <= self.weight <= 1:
            raise SchemaValidationError(f"weight must be between 0 and 1, got {self.weight}")
        self._validate_graph_settings()

    def _validate_graph_settings(self):
        if self.graph_role not in GRAPH_ROLES:
            raise SchemaValidationError(
                f"Unknown graph_role '{self.graph_role}'; supported: {sorted(GRAPH_ROLES)}"
            )
        if self.graph_label is None:
            self.graph_label = "".join(part.capitalize() for part in self.name.split("_"))
        if not _SAFE_LABEL.match(self.graph_label):
            raise SchemaValidationError(
                f"graph_label '{self.graph_label}' must match {_SAFE_LABEL.pattern}"
            )
        if self.relationship_type is None:
            prefix = "USES" if self.graph_role == "node_cluster" else "HAS"
            self.relationship_type = f"{prefix}_{self.name.upper()}"
        relationship_types = [self.relationship_type] + list(
            (self.graph_edge_sources or {}).values()
        )
        for relationship_type in relationship_types:
            if not _SAFE_RELATIONSHIP.match(str(relationship_type)):
                raise SchemaValidationError(
                    f"relationship type '{relationship_type}' must match {_SAFE_RELATIONSHIP.pattern}"
                )
        if self.graph_edge_sources is not None:
            unknown = set(self.graph_edge_sources) - GRAPH_EDGE_SOURCES
            if unknown or not self.graph_edge_sources:
                raise SchemaValidationError(
                    f"graph_edge_sources keys must be among {sorted(GRAPH_EDGE_SOURCES)}"
                )
            if self.graph_role != "node_cluster":
                raise SchemaValidationError("graph_edge_sources requires graph_role 'node_cluster'")
        for attr in ("graph_key_property", "graph_value_property"):
            if not _SAFE_NAME.match(str(getattr(self, attr))):
                raise SchemaValidationError(
                    f"{attr} '{getattr(self, attr)}' must match {_SAFE_NAME.pattern}"
                )
        if self.graph_role == "joined_node" and self.parsed_type.kind != "list":
            raise SchemaValidationError("graph_role 'joined_node' requires a list type")

    @property
    def is_object(self) -> bool:
        return self.parsed_type.kind == "object"

    @property
    def is_object_dict(self) -> bool:
        return self.parsed_type.kind == "dict" and self.parsed_type.args[0].contains(
            "object"
        )

    def _derive_default(self) -> Any:
        kind = self.parsed_type.kind
        if self.parsed_type.is_nullable():
            return None
        if kind == "str":
            return ""
        if kind == "list":
            return []
        if kind == "dict":
            return {}
        if kind == "bool":
            return False
        # Numbers and objects default to null
        return None

    def _derive_match_mode(self) -> Optional[str]:
        parsed = self.parsed_type
        if parsed.kind == "union":
            members = [a for a in parsed.args if a.kind != "None"]
            if members and all(a.kind in ("int", "float") for a in members):
                return "numeric"
            parsed = members[0] if members else parsed
        kind = parsed.kind
        if kind == "object":
            return None
        if kind in ("int", "float"):
            return "numeric"
        if kind == "bool":
            return "exact"
        if kind == "list":
            element = parsed.args[0]
            numeric = [a for a in (element.args if element.kind == "union" else [element]) if a.kind != "None"]
            if numeric and all(a.kind in ("int", "float") for a in numeric):
                return "numeric"
            return "item_list"
        if kind == "dict":
            return "keyed_values"
        return "semantic"

    def get_default(self) -> Any:
        """Return a copy of the default value."""
        return copy.deepcopy(self.default)


_FIELD_KEYS = {
    "name",
    "type",
    "description",
    "required",
    "default",
    "allowed_values",
    "allowed_values_ref",
    "value_fields",
    "extraction_only",
    "format_note",
    "example",
    "variable_example",
    "weight",
    "match_mode",
    "similarity_threshold",
    "similarity_threshold_mode",
    "plot_kind",
    "evaluation_note",
    "graph_role",
    "graph_label",
    "relationship_type",
    "graph_edge_sources",
    "graph_key_property",
    "graph_value_property",
    "graph_default",
    "cluster_similarity_threshold",
}


def _parse_fields(raw_fields: Any, location: str) -> List[SchemaField]:
    if not isinstance(raw_fields, list) or not raw_fields:
        raise SchemaValidationError(f"{location} must be a non-empty list of fields")
    fields, seen = [], set()
    for index, raw in enumerate(raw_fields):
        where = f"{location}[{index}]"
        if not isinstance(raw, dict):
            raise SchemaValidationError(f"{where} must be a mapping")
        unknown = set(raw) - _FIELD_KEYS
        if unknown:
            raise SchemaValidationError(f"{where} has unknown keys: {sorted(unknown)}")
        for key in ("name", "type"):
            if key not in raw:
                raise SchemaValidationError(f"{where} is missing '{key}'")
        name = raw["name"]
        if not isinstance(name, str) or not _SAFE_NAME.match(name):
            raise SchemaValidationError(
                f"{where} has invalid name {name!r}: must match {_SAFE_NAME.pattern}"
            )
        if name in seen:
            raise SchemaValidationError(f"{location} declares '{name}' more than once")
        seen.add(name)

        kwargs = {k: v for k, v in raw.items() if k != "value_fields"}
        if raw.get("value_fields") is not None:
            kwargs["value_fields"] = _parse_fields(
                raw["value_fields"], f"{where}.value_fields"
            )
        try:
            schema_field = SchemaField(**kwargs)
        except SchemaValidationError as e:
            raise SchemaValidationError(f"{where}: {e}") from None

        needs_value_fields = schema_field.is_object or schema_field.is_object_dict
        if needs_value_fields and not schema_field.value_fields:
            raise SchemaValidationError(
                f"{where} ('{name}') has type '{schema_field.type}' but no value_fields"
            )
        if schema_field.value_fields and not needs_value_fields:
            raise SchemaValidationError(
                f"{where} ('{name}') declares value_fields, which require type 'object' or 'dict[str, object]'"
            )
        if (needs_value_fields or schema_field.parsed_type.kind == "dict") and schema_field.graph_role != "node_property":
            raise SchemaValidationError(
                f"{where} ('{name}'): graph_role applies to leaf fields only; nested and keyed fields are always nodes"
            )
        if schema_field.parsed_type.contains("object") and not needs_value_fields:
            raise SchemaValidationError(
                f"{where} ('{name}'): 'object' is only supported as 'object' or 'dict[str, object]'"
            )
        fields.append(schema_field)
    return fields


# Allowed-values lists supplied at runtime by the API
ALLOWED_VALUES_REFS = {"allowed_synthesis_methods", "allowed_characterization_techniques"}
# Placeholders allowed per domain and attribute (CrewAI fails on unknown ones)
_PROMPT_PLACEHOLDERS = {
    "composition": {"main_extraction_keyword"},
    "synthesis": set(),
}
_EVALUATION_PLACEHOLDERS = {
    "composition": {"value_error_threshold_instructions"},
    "synthesis": set(),
}
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_\-]*)\}")


def _iter_fields(fields: List[SchemaField]):
    for schema_field in fields:
        yield schema_field
        if schema_field.value_fields:
            yield from _iter_fields(schema_field.value_fields)


def _validate_prompt_text(fields: List[SchemaField], domain: str) -> None:
    for schema_field in _iter_fields(fields):
        for attr in ("description", "format_note", "evaluation_note"):
            if attr == "evaluation_note":
                allowed = _EVALUATION_PLACEHOLDERS.get(domain, set())
            else:
                allowed = _PROMPT_PLACEHOLDERS.get(domain, set())
            used = set(_PLACEHOLDER.findall(getattr(schema_field, attr) or ""))
            if used - allowed:
                supported = sorted(allowed) or "none"
                raise SchemaValidationError(
                    f"{domain} field '{schema_field.name}' {attr} uses unsupported placeholder(s) "
                    f"{sorted(used - allowed)}; supported in {domain} fields: {supported}"
                )
        ref = schema_field.allowed_values_ref
        if ref is not None and (domain != "synthesis" or ref not in ALLOWED_VALUES_REFS):
            raise SchemaValidationError(
                f"{domain} field '{schema_field.name}' has allowed_values_ref '{ref}'; "
                f"only synthesis fields may use {sorted(ALLOWED_VALUES_REFS)} (use allowed_values for an inline list)"
            )
        if ref is not None:
            message = (
                f"{domain} field '{schema_field.name}': allowed_values_ref is deprecated; "
                "list the values in allowed_values instead."
            )
            logger.warning(message)
            warnings.warn(message, DeprecationWarning, stacklevel=2)


def normalize_weights(fields: List[SchemaField], parent_effective_weight: float = 1.0) -> None:
    """Fill unspecified weights within each sibling group and cascade effective weights."""
    evaluated = [f for f in fields if not f.extraction_only]
    specified_sum = sum(f.weight for f in evaluated if f.weight is not None)
    if specified_sum > 1.0 + 1e-9:
        raise SchemaValidationError(
            f"Weights of {[f.name for f in evaluated]} sum to {specified_sum} > 1.0"
        )
    unspecified = [f for f in evaluated if f.weight is None]
    for schema_field in unspecified:
        schema_field.weight = (1.0 - specified_sum) / len(unspecified)
    for schema_field in fields:
        if schema_field.extraction_only:
            schema_field.weight = 0.0
        schema_field.effective_weight = schema_field.weight * parent_effective_weight
        if schema_field.value_fields:
            normalize_weights(schema_field.value_fields, schema_field.effective_weight)


BUILTIN_ABBREVIATIONS_FIELD = "abbreviations"


def _attach_builtin_abbreviations(
    composition_fields: List[SchemaField], primary: Optional[str]
) -> List[SchemaField]:
    """Attach the default abbreviations field if the schema doesn't declare its own."""
    if any(f.name == BUILTIN_ABBREVIATIONS_FIELD for f in composition_fields):
        return composition_fields
    with open(DEFAULT_SCHEMA_PATH, "r", encoding="utf-8") as f:
        default_raw = yaml.safe_load(f)
    builtin = next(
        f
        for f in _parse_fields(
            default_raw["composition_fields"], "default composition_fields"
        )
        if f.name == BUILTIN_ABBREVIATIONS_FIELD
    )
    names = [f.name for f in composition_fields]
    if primary not in names:
        primary = next(
            (f.name for f in composition_fields if f.parsed_type.kind == "dict"), None
        )
    insert_at = names.index(primary) + 1 if primary in names else len(names)
    return composition_fields[:insert_at] + [builtin] + composition_fields[insert_at:]


@dataclass
class ExtractionSchemaConfig:
    """
    Extraction schema loaded from a YAML file.

    Args:
        version (str): Schema format version.
        composition_fields (list): Fields of the composition_data section.
        synthesis_fields (list): Fields of the synthesis_data section.
        primary_composition_field (str): Composition-keyed field used for cleaning and relevance checks.
        extra_domains (dict, optional): Further field groups (parsed, not yet extracted).
        source_path (Path, optional): File the schema was loaded from.
    """

    version: str
    composition_fields: List[SchemaField]
    synthesis_fields: List[SchemaField]
    primary_composition_field: str
    extra_domains: Dict[str, List[SchemaField]] = field(default_factory=dict)
    source_path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExtractionSchemaConfig":
        """Load and validate a schema file."""
        path = Path(path)
        if not path.is_file():
            raise SchemaValidationError(f"Schema file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        try:
            return cls.from_dict(raw, source_path=path)
        except SchemaValidationError as e:
            raise SchemaValidationError(f"{path}: {e}") from None

    @classmethod
    def from_dict(
        cls, raw: Any, source_path: Optional[Path] = None
    ) -> "ExtractionSchemaConfig":
        if not isinstance(raw, dict):
            raise SchemaValidationError("Schema must be a YAML mapping")
        version = str(raw.get("version", ""))
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaValidationError(
                f"Unsupported schema version {version!r}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        unknown = set(raw) - {
            "version",
            "composition_fields",
            "synthesis_fields",
            "primary_composition_field",
            "extra_domains",
        }
        if unknown:
            raise SchemaValidationError(f"Unknown top-level keys: {sorted(unknown)}")

        composition_fields = _parse_fields(
            raw.get("composition_fields"), "composition_fields"
        )
        composition_fields = _attach_builtin_abbreviations(
            composition_fields, raw.get("primary_composition_field")
        )
        synthesis_fields = _parse_fields(raw.get("synthesis_fields"), "synthesis_fields")
        _validate_prompt_text(composition_fields, "composition")
        _validate_prompt_text(synthesis_fields, "synthesis")
        for domain_fields in (composition_fields, synthesis_fields):
            try:
                normalize_weights(domain_fields)
            except SchemaValidationError as e:
                raise SchemaValidationError(f"Invalid weights: {e}") from None
        extra_domains = {}
        for domain, domain_fields in (raw.get("extra_domains") or {}).items():
            if not _SAFE_NAME.match(str(domain)) or domain in SCHEMA_DOMAINS:
                raise SchemaValidationError(f"Invalid extra domain name {domain!r}")
            extra_domains[domain] = _parse_fields(
                domain_fields, f"extra_domains.{domain}"
            )

        primary = raw.get("primary_composition_field")
        if primary is None:
            primary = next(
                (f.name for f in composition_fields if f.parsed_type.kind == "dict"),
                None,
            )
        primary_field = next(
            (f for f in composition_fields if f.name == primary), None
        )
        if primary_field is None:
            raise SchemaValidationError(
                f"primary_composition_field {primary!r} is not declared in composition_fields"
            )
        if primary_field.parsed_type.kind != "dict" or primary_field.extraction_only:
            raise SchemaValidationError(
                f"primary_composition_field '{primary}' must be a dict-typed field kept in the final output"
            )

        return cls(
            version=version,
            composition_fields=composition_fields,
            synthesis_fields=synthesis_fields,
            primary_composition_field=primary,
            extra_domains=extra_domains,
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "ExtractionSchemaConfig":
        """Load `path`, or the bundled default schema when `path` is None."""
        return cls.from_yaml(path if path is not None else DEFAULT_SCHEMA_PATH)

    def fields(self, domain: str, final_only: bool = False) -> List[SchemaField]:
        """Fields of a domain; final_only drops extraction-only fields."""
        if domain == "composition":
            domain_fields = self.composition_fields
        elif domain == "synthesis":
            domain_fields = self.synthesis_fields
        else:
            raise ValueError(f"Unknown schema domain '{domain}'")
        if final_only:
            return [f for f in domain_fields if not f.extraction_only]
        return list(domain_fields)

    def field_names(self, domain: str, final_only: bool = True) -> List[str]:
        return [f.name for f in self.fields(domain, final_only=final_only)]

    def default_shape(self, domain: str, final_only: bool = True) -> Dict[str, Any]:
        """Empty record for a domain."""
        return {f.name: f.get_default() for f in self.fields(domain, final_only)}


DEFAULT_SCHEMA_CONFIG = ExtractionSchemaConfig.load()


def load_schema_config(
    schema_config_path: Optional[Union[str, Path]] = None,
) -> ExtractionSchemaConfig:
    """Load the schema at the given path, or the default schema if None."""
    if schema_config_path is None:
        return DEFAULT_SCHEMA_CONFIG
    return ExtractionSchemaConfig.from_yaml(schema_config_path)


# Synthesis field each deprecated runtime list fills when no field references it
LEGACY_ALLOWED_VALUES_FIELDS = {
    "allowed_synthesis_methods": "method",
    "allowed_characterization_techniques": "characterization_techniques",
}


def apply_legacy_allowed_values(
    schema_config: ExtractionSchemaConfig, legacy_args: Dict[str, List[str]]
) -> Tuple[ExtractionSchemaConfig, List[str]]:
    """Copy deprecated runtime allowed-values lists into the matching schema fields.

    A list goes to the synthesis field whose allowed_values_ref names it, else to the field named in LEGACY_ALLOWED_VALUES_FIELDS. A field that already lists allowed_values keeps them.

    Args:
        schema_config (ExtractionSchemaConfig, required): Loaded extraction schema.
        legacy_args (dict, required): Deprecated keys from LEGACY_ALLOWED_VALUES_FIELDS with their lists.

    Returns:
        tuple: two items:
            - ExtractionSchemaConfig: the schema to use (a copy when anything changed).
            - list: deprecation messages to report.

    Raises:
        SchemaValidationError: If the schema has no field for a given list.
    """
    if not legacy_args:
        return schema_config, []
    schema_config = copy.deepcopy(schema_config)
    messages = []
    for arg, values in legacy_args.items():
        fields = schema_config.synthesis_fields
        target = next((f for f in fields if f.allowed_values_ref == arg), None) or next(
            (f for f in fields if f.name == LEGACY_ALLOWED_VALUES_FIELDS[arg]), None
        )
        if target is None:
            raise SchemaValidationError(
                f"{arg} was given, but the schema has no synthesis field "
                f"'{LEGACY_ALLOWED_VALUES_FIELDS[arg]}' or field with allowed_values_ref: {arg}; "
                "list the values in allowed_values of the field they belong to"
            )
        if target.allowed_values:
            messages.append(
                f"{arg} is deprecated and was ignored because the schema field "
                f"'{target.name}' sets allowed_values."
            )
            continue
        target.allowed_values = list(values) or None
        messages.append(
            f"{arg} is deprecated; list the values in allowed_values of the schema field '{target.name}'."
        )
    return schema_config, messages
