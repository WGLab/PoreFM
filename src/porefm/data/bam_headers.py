import copy
import os
import shlex
import sys
from pathlib import Path

import pysam


def current_command_line():
    """Return the current Python invocation as a shell-escaped command line."""
    argv = [sys.executable] + sys.argv
    return shlex.join(str(arg) for arg in argv)


def add_pg_to_header(header, program_name=None, program_id=None, command_line=None):
    """Append a uniquely identified ``@PG`` record to a BAM header.

    Parameters
    ----------
    header : pysam.AlignmentHeader, dict, str, or header-like object
        Source header. Dictionaries are deep-copied; the input is not mutated.
    program_name : str, optional
        Value for ``PN``. Defaults to the executable basename.
    program_id : str, optional
        Preferred ``ID``. A numeric suffix is added when it collides with an
        existing program record.
    command_line : str, optional
        Value for ``CL``. Defaults to :func:`current_command_line`.

    Returns
    -------
    pysam.AlignmentHeader
        New header whose appended program record points to the previous final
        program through ``PP`` when that record has an ID.
    """
    header_dict = _header_to_dict(header)
    pg_records = list(header_dict.get("PG", []))

    program_name = program_name or Path(sys.argv[0]).name or Path(sys.executable).name
    base_id = program_id or program_name
    existing_ids = {str(record.get("ID")) for record in pg_records if record.get("ID")}
    pg_id = _unique_pg_id(base_id, existing_ids)

    pg_record = {
        "ID": pg_id,
        "PN": program_name,
        "CL": command_line or current_command_line(),
    }
    if pg_records and pg_records[-1].get("ID"):
        pg_record["PP"] = str(pg_records[-1]["ID"])

    header_dict["PG"] = pg_records + [pg_record]
    return pysam.AlignmentHeader.from_dict(header_dict)


def _header_to_dict(header):
    """Convert supported header representations to an independent mapping."""
    if isinstance(header, dict):
        return copy.deepcopy(header)
    if isinstance(header, str):
        return pysam.AlignmentHeader.from_text(header).to_dict()
    if hasattr(header, "to_dict"):
        return header.to_dict()
    raise TypeError(f"Unsupported BAM header type: {type(header)!r}")


def _unique_pg_id(base_id, existing_ids):
    """Return ``base_id`` or its first unused dot-numbered variant."""
    base_id = str(base_id)
    if base_id not in existing_ids:
        return base_id

    index = 1
    while True:
        candidate = f"{base_id}.{index}"
        if candidate not in existing_ids:
            return candidate
        index += 1
