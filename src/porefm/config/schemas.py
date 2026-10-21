"""JSON Schema documents for supported PoreFM configuration formats."""

from __future__ import annotations

from copy import deepcopy


SCHEMA_VERSION = 1
CONFIG_TYPES = ("pretraining", "bam-sharding", "supervised")


def _header(config_type: str) -> dict:
    return {
        "config_type": {
            "type": "string",
            "const": config_type,
            "default": config_type,
            "description": "Configuration discriminator.",
        },
        "schema_version": {
            "type": "integer",
            "const": SCHEMA_VERSION,
            "default": SCHEMA_VERSION,
            "description": "PoreFM schema version for this configuration type.",
        },
    }


_POLISHING_PROPERTIES = {
    "polish": {
        "type": "boolean",
        "default": False,
        "description": "Enable reference-guided sequence polishing for this sample.",
    },
    "polish_indel_limit": {
        "type": "integer",
        "minimum": 0,
        "default": 5,
        "description": "Largest unmasked insertion or deletion corrected by polishing.",
    },
    "polish_vcf_path": {
        "type": ["string", "null"],
        "default": None,
        "description": "Optional VCF/BCF whose listed variants polishing preserves.",
    },
}

_POLISHING_CONSTRAINT = {
    "if": {
        "required": ["polish_vcf_path"],
        "properties": {
            "polish_vcf_path": {"type": "string", "minLength": 1},
        },
    },
    "then": {
        "required": ["polish"],
        "properties": {"polish": {"const": True}},
    },
}


PRETRAINING_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:porefm:schema:pretraining:1",
    "title": "PoreFM pretraining BAM manifest",
    "description": (
        "Unlabelled signal-BAM inputs for pretraining and canonical-null workflows. "
        "Polishing remains a run-wide CLI setting rather than a per-sample field."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": ["samples"],
    "properties": {
        **_header("pretraining"),
        "samples": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["bam_path"],
                "properties": {
                    "bam_path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Signal BAM file consumed with sequential until-EOF streaming.",
                    },
                    "sampling_probability": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                        "default": 1.0,
                        "description": "Independent probability of retaining each eligible BAM record.",
                    },
                },
            },
        },
    },
}


BAM_SHARDING_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:porefm:schema:bam-sharding:1",
    "title": "PoreFM BAM sharding manifest",
    "description": "BAM inputs and optional contig restrictions for coordinate sharding.",
    "type": "object",
    "additionalProperties": False,
    "required": ["inputs"],
    "properties": {
        **_header("bam-sharding"),
        "inputs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["bam_path"],
                "properties": {
                    "bam_path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Input BAM file to shard.",
                    },
                    "contigs": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "uniqueItems": True,
                        "default": [],
                        "description": "Contigs to fetch; an empty list streams all records.",
                    },
                },
            },
        },
    },
}


_LABEL_SMOOTHING_SCHEMA = {
    "type": "number",
    "minimum": 0,
    "exclusiveMaximum": 1,
    "description": (
        "Classification uniform-target mixing epsilon. Sample overrides config, "
        "then --label_smoothing; an explicit zero disables smoothing."
    ),
}


_LABELLED_SAMPLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["bam_path"],
    "properties": {
        "bam_path": {"type": "string", "minLength": 1},
        "pod5": {
            "type": "string",
            "minLength": 1,
            "description": "porefm detect-modification (alias: porefm infer) only: POD5 file or recursively searched directory supplying raw signal instead of SG.",
        },
        "bam_index": {
            "type": "string", "minLength": 1,
            "description": "POD5 inference only: saved unfiltered index from porefm utils index-bam.",
        },
        "contigs": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
            "default": [],
        },
        "ref": {"type": ["string", "null"], "default": None},
        "pos_file": {"type": ["string", "null"], "default": None},
        "motif": {"type": ["string", "null"], "default": None},
        "motif_label": {
            "type": ["string", "integer"],
            "default": 0,
            "description": (
                "Semicolon-delimited class indices. If fewer labels than motifs "
                "are supplied, the final label applies to the remaining motifs."
            ),
        },
        "label_smoothing": _LABEL_SMOOTHING_SCHEMA,
        "soft_tags": {"type": "number", "minimum": 0, "maximum": 1, "deprecated": True},
        "flag": {"type": "integer", "minimum": 0, "default": 0},
        "sample_labels": {"type": "number", "minimum": 0, "maximum": 1, "default": 1.0},
        "read_based": {"type": "boolean", "default": False},
        **_POLISHING_PROPERTIES,
    },
    "allOf": [
        {
            "if": {
                "required": ["read_based"],
                "properties": {"read_based": {"const": True}},
            },
            "then": {
                "required": ["motif"],
                "properties": {
                    "motif": {"type": "string", "minLength": 1},
                    "pos_file": {"type": "null"},
                },
            },
            "else": {
                "required": ["ref"],
                "properties": {
                    "ref": {"type": "string", "minLength": 1},
                },
                "oneOf": [
                    {
                        "required": ["motif"],
                        "properties": {
                            "motif": {"type": "string", "minLength": 1},
                            "pos_file": {"type": "null"},
                        },
                    },
                    {
                        "required": ["pos_file"],
                        "properties": {
                            "pos_file": {"type": "string", "minLength": 1},
                            "motif": {"type": "null"},
                        },
                    },
                ],
            },
        },
        _POLISHING_CONSTRAINT,
    ],
}


_TAGGED_SAMPLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["bam_path"],
    "properties": {
        "bam_path": {"type": "string", "minLength": 1},
        "contigs": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
            "default": [],
        },
        "ref": {"type": ["string", "null"], "minLength": 1},
        "label_coordinates": {"enum": ["read", "ref"]},
        "read_label_dict_path": {"type": ["string", "null"], "default": None},
        "label_smoothing": _LABEL_SMOOTHING_SCHEMA,
        "soft_tags": {"type": "number", "minimum": 0, "maximum": 1, "deprecated": True},
        "label": {"type": "integer", "minimum": 0, "default": 1},
        "negative_label": {"type": ["integer", "null"], "minimum": 0, "default": None},
        "ML_tag": {"type": "boolean", "default": False},
        "MM_code": {
            "oneOf": [
                {"type": "string", "pattern": "^([A-Za-z]|[0-9]*[1-9][0-9]*)$"},
                {"type": "integer", "minimum": 1},
            ],
            "description": "Exact MM modification code to select; requires ML_tag=true.",
        },
        "ML_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
        "min_read_frac": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.0},
        "flag": {"type": "integer", "minimum": 0, "default": 0},
        **_POLISHING_PROPERTIES,
    },
    "allOf": [
        {
            "if": {
                "required": ["ML_tag"],
                "properties": {"ML_tag": {"const": True}},
            },
            "then": {
                "required": ["ref"],
                "properties": {
                    "ref": {"type": "string", "minLength": 1},
                    "read_label_dict_path": {"type": "null"},
                },
                "not": {"required": ["label_coordinates"]},
            },
            "else": {
                "required": ["read_label_dict_path", "label_coordinates"],
                "properties": {
                    "read_label_dict_path": {"type": "string", "minLength": 1},
                },
            },
        },
        {
            "if": {
                "required": ["label_coordinates"],
                "properties": {"label_coordinates": {"const": "ref"}},
            },
            "then": {
                "required": ["ref"],
                "properties": {"ref": {"type": "string", "minLength": 1}},
            },
        },
        _POLISHING_CONSTRAINT,
    ],
}


_SUPERVISION_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"const": "classification", "default": "classification"},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "excluded_class"],
            "properties": {
                "type": {"const": "exclusion"},
                "excluded_class": {"type": "string", "minLength": 1},
                "expected_allowed_proportions": {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0},
                },
                "proportion_tolerance": {
                    "type": "number",
                    "minimum": 0,
                    "exclusiveMaximum": 1,
                },
            },
            "dependentRequired": {
                "expected_allowed_proportions": ["proportion_tolerance"],
                "proportion_tolerance": ["expected_allowed_proportions"],
            },
        },
    ],
    "default": {"type": "classification"},
}


SUPERVISED_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:porefm:schema:supervised:1",
    "title": "PoreFM supervised dataset configuration",
    "description": "Labelled or tagged BAM samples sharing one supervision policy.",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **_header("supervised"),
        "supervision": _SUPERVISION_SCHEMA,
        "label_smoothing": _LABEL_SMOOTHING_SCHEMA,
        "labelled_samples": {
            "type": "array",
            "minItems": 1,
            "items": _LABELLED_SAMPLE_SCHEMA,
        },
        "tagged_samples": {
            "type": "array",
            "minItems": 1,
            "items": _TAGGED_SAMPLE_SCHEMA,
        },
    },
    "oneOf": [
        {"required": ["labelled_samples"], "not": {"required": ["tagged_samples"]}},
        {"required": ["tagged_samples"], "not": {"required": ["labelled_samples"]}},
    ],
    "allOf": [
        {
            "if": {
                "required": ["supervision"],
                "properties": {
                    "supervision": {
                        "required": ["type"],
                        "properties": {"type": {"const": "exclusion"}},
                    },
                },
            },
            "then": {"required": ["labelled_samples"]},
        },
    ],
}


SCHEMAS = {
    "pretraining": PRETRAINING_SCHEMA,
    "bam-sharding": BAM_SHARDING_SCHEMA,
    "supervised": SUPERVISED_SCHEMA,
}


def get_schema(config_type: str) -> dict:
    """Return an independent JSON Schema document for ``config_type``.

    Parameters
    ----------
    config_type : str
        One of :data:`CONFIG_TYPES`.

    Returns
    -------
    dict
        Deep copy safe for caller mutation or JSON serialization.

    Raises
    ------
    KeyError
        If the configuration type is unknown.
    """
    return deepcopy(SCHEMAS[config_type])
