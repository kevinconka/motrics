"""Config/validation helpers, mirroring ``trackeval.utils``."""

from __future__ import annotations

from typing import Any


class TrackEvalException(Exception):  # noqa: N818 - mirrors trackeval.utils.TrackEvalException
    """Raised for a config or data error."""


def init_config(
    config: dict[str, Any] | None,
    default_config: dict[str, Any],
    name: str | None = None,
) -> dict[str, Any]:
    """Fill missing keys in `config` with defaults; print if `PRINT_CONFIG` is set."""
    if config is None:
        config = default_config
    else:
        for k, v in default_config.items():
            config.setdefault(k, v)
    if name and config["PRINT_CONFIG"]:
        print(f"\n{name} Config:")
        for c, v in config.items():
            print(f"{c:<20} : {v!s:<30}")
    return config


def validate_metrics_list(metrics_list: list[Any]) -> list[str]:
    """Ensure metric class names and their combined field names are unique."""
    metric_names = [type(m).__name__ for m in metrics_list]
    if len(metric_names) != len(set(metric_names)):
        raise TrackEvalException(
            "Code being run with multiple metrics of the same name"
        )
    fields = [field for m in metrics_list for field in m.fields]
    if len(fields) != len(set(fields)):
        raise TrackEvalException(
            "Code being run with multiple metrics with fields of the same name"
        )
    return metric_names
