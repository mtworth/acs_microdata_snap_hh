"""Summarize CPS Food Security Supplement household food security status.

This module fetches household-level microdata from the CPS Food Security
Supplement (FSS) via the Census microdata API and aggregates the weighted
number of households by food security status for a given state.

The implementation intentionally avoids third-party dependencies so that it
can run in constrained environments.  It mirrors the structure of the
ACS-focused notebook in this repository while adapting it to the CPS FSS
dataset.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Iterable, List, Mapping, MutableMapping, Sequence
from urllib import error, parse, request


DEFAULT_YEAR = 2023
DATASET = "fss"
BASE_URL_TEMPLATE = "https://api.census.gov/data/{year}/cps/{dataset}"
DEFAULT_FIELDS: Sequence[str] = ("FSSTATUS", "FSWGT", "HRHHID")
STATE_FIELD = "state"
STATUS_FIELD = "FSSTATUS"
WEIGHT_FIELD = "FSWGT"
HOUSEHOLD_ID_FIELD = "HRHHID"

STATUS_LABELS = OrderedDict([
    ("1", "High food security"),
    ("2", "Marginal food security"),
    ("3", "Low food security"),
    ("4", "Very low food security"),
])


class CensusApiError(RuntimeError):
    """Raised when the Census API request fails."""


@dataclass
class FoodSecuritySummaryRow:
    """Represents a single summarized row for reporting."""

    status_code: str
    status_label: str
    weighted_households: int

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status_code": self.status_code,
            "status_label": self.status_label,
            "weighted_households": self.weighted_households,
        }


@dataclass
class FoodSecuritySummary:
    """Container for a grouped food security summary."""

    rows: List[FoodSecuritySummaryRow]
    total_households: int
    suppressed_weight: int

    def to_table(self) -> str:
        """Render the summary as a simple table."""

        lines = [
            f"{'Food Security Status':<35} {'Weighted Households':>20}",
            f"{'-' * 35} {'-' * 20}",
        ]
        for row in self.rows:
            lines.append(
                f"{row.status_label:<35} {row.weighted_households:>20,}"
            )
        lines.append(f"{'-' * 35} {'-' * 20}")
        lines.append(f"{'Total':<35} {self.total_households:>20,}")
        if self.suppressed_weight:
            lines.append(
                "Note: "
                f"{self.suppressed_weight:,} households had unclassified "
                "food security codes and were excluded from the rows above."
            )
        return "\n".join(lines)


def build_request_url(
    api_key: str,
    *,
    year: int = DEFAULT_YEAR,
    dataset: str = DATASET,
    fields: Sequence[str] = DEFAULT_FIELDS,
    state: str = "06",
) -> str:
    """Construct the Census API query URL for the CPS FSS dataset."""

    if not api_key:
        raise ValueError("An API key is required to query the Census API.")

    cleaned_state = state.strip()
    if not cleaned_state:
        raise ValueError("State FIPS code must be provided.")

    query_params = {
        "get": ",".join(fields),
        "for": f"{STATE_FIELD}:{cleaned_state}",
        "key": api_key,
    }
    encoded_params = parse.urlencode(query_params, safe=",:")
    base_url = BASE_URL_TEMPLATE.format(year=year, dataset=dataset)
    return f"{base_url}?{encoded_params}"


def fetch_cps_fss_records(
    api_key: str,
    *,
    year: int = DEFAULT_YEAR,
    state: str = "06",
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> List[MutableMapping[str, str]]:
    """Fetch raw CPS FSS records for the requested state."""

    url = build_request_url(api_key, year=year, state=state, fields=fields)

    try:
        with request.urlopen(url) as response:
            if response.status != 200:
                raise CensusApiError(
                    f"Census API request failed with HTTP {response.status}: {url}"
                )
            payload = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise CensusApiError(
            f"Census API request failed with HTTP {exc.code}: {exc.reason}"
        ) from exc
    except error.URLError as exc:
        raise CensusApiError(f"Unable to reach the Census API: {exc.reason}") from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CensusApiError("Received malformed JSON from the Census API") from exc

    if not data:
        return []

    headers = data[0]
    records: List[MutableMapping[str, str]] = []
    for row in data[1:]:
        record = {headers[i]: value for i, value in enumerate(row)}
        records.append(record)
    return records


def _parse_weight(value: str) -> Decimal:
    """Convert a Census weight to a Decimal, ignoring invalid entries."""

    if value in (None, ""):
        raise InvalidOperation
    return Decimal(value)


def summarize_food_security(
    records: Iterable[Mapping[str, str]],
    *,
    status_field: str = STATUS_FIELD,
    weight_field: str = WEIGHT_FIELD,
    status_labels: Mapping[str, str] = STATUS_LABELS,
) -> FoodSecuritySummary:
    """Aggregate weighted household counts by food security status."""

    totals: MutableMapping[str, Decimal] = OrderedDict(
        (code, Decimal(0)) for code in status_labels
    )
    suppressed = Decimal(0)

    for record in records:
        status_code = record.get(status_field)
        weight_value = record.get(weight_field)
        if status_code is None:
            continue
        try:
            weight = _parse_weight(weight_value)
        except (InvalidOperation, TypeError):
            continue

        if status_code in totals:
            totals[status_code] += weight
        else:
            suppressed += weight

    rows: List[FoodSecuritySummaryRow] = []
    total_weight = Decimal(0)
    for code, label in status_labels.items():
        weight = totals[code]
        rounded = weight.to_integral_value(rounding=ROUND_HALF_UP)
        rows.append(
            FoodSecuritySummaryRow(
                status_code=code,
                status_label=label,
                weighted_households=int(rounded),
            )
        )
        total_weight += weight

    total_households = int(
        total_weight.to_integral_value(rounding=ROUND_HALF_UP)
    )
    suppressed_households = int(
        suppressed.to_integral_value(rounding=ROUND_HALF_UP)
    )

    return FoodSecuritySummary(
        rows=rows,
        total_households=total_households,
        suppressed_weight=suppressed_households,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize weighted household counts by food security status for "
            "the CPS Food Security Supplement."
        )
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Census API key with access to CPS microdata.",
    )
    parser.add_argument(
        "--state",
        default="06",
        help="State FIPS code to summarize (default: 06 for California).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="CPS FSS reference year (default: 2023).",
    )

    args = parser.parse_args(argv)

    try:
        records = fetch_cps_fss_records(
            args.api_key, year=args.year, state=args.state
        )
    except CensusApiError as exc:
        parser.error(str(exc))
        return 1

    summary = summarize_food_security(records)

    state_label = args.state
    print(
        f"CPS Food Security Supplement household summary for state {state_label}"
    )
    print(summary.to_table())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
