"""Offline integrity and provenance checks for the shipped synthetic PII corpus.

These checks verify the reference data, not AIGate's detection accuracy. Identifier
checksums are mathematical format checks; no ownership/registry lookup is made.
"""

from __future__ import annotations

import collections
import ipaddress
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "benchmarks" / "pii").glob("curated_*.jsonl"))
EXPECTED_TYPES = {
    "PII.EMAIL", "PII.PHONE", "PII.PERSON", "PII.PASSPORT.RU",
    "PII.PASSPORT.FOREIGN", "PII.SNILS", "PII.INN", "PII.DRIVER_LICENSE",
    "PII.BANK_CARD", "PII.CVC", "PII.BANK_ACCOUNT", "PII.ADDRESS",
    "PII.DATE_OF_BIRTH", "PII.LOGIN", "NETWORK.IP", "NETWORK.MAC",
    "PII.BIK", "PII.KPP", "PII.OGRN", "PII.OGRNIP", "PII.IBAN",
    "PII.SWIFT_BIC", "PII.OMS", "PII.VEHICLE_REGISTRATION", "PII.VIN",
    "PII.VEHICLE_DOCS", "SECRET.PASSWORD", "SECRET.API_KEY",
}


def corpus():
    return [
        json.loads(line)
        for path in FILES
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def entities():
    return [entity for row in corpus() for entity in row["entities"]]


def digits(value):
    return "".join(char for char in value if char.isdigit())


def test_shipped_corpus_has_broad_coverage_and_negative_controls():
    rows = corpus()
    assert len(FILES) >= 8
    assert len(rows) >= 350
    counts = collections.Counter(entity["type"] for row in rows for entity in row["entities"])
    assert set(counts) == EXPECTED_TYPES
    assert min(counts.values()) >= 4
    assert sum(not row["entities"] for row in rows) >= len(rows) // 5
    assert sum(len(row["entities"]) > 1 for row in rows) >= 40
    languages = collections.Counter(row["language"] for row in rows)
    assert languages["ru"] >= 100 and languages["en"] >= 100
    assert len({row["category"] for row in rows}) >= 30
    assert sum(row["difficulty"] == "boundary" for row in rows) >= 30


def test_ids_text_and_spans_are_unique_exact_non_overlapping_and_fully_labeled():
    rows = corpus()
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["text"] for row in rows}) == len(rows)
    for row in rows:
        assert row["synthetic"] is True
        assert row["provenance"] == "aitalon-authored-synthetic-v1"
        assert row["source"].startswith("curated-pii-")
        assert row["language"] in {"ru", "en"}
        assert isinstance(row["tags"], list) and row["tags"]
        assert row["text"].strip() and "[[" not in row["text"]
        covered = set()
        for entity in row["entities"]:
            start, end = entity["start"], entity["end"]
            assert type(start) is int and type(end) is int
            assert 0 <= start < end <= len(row["text"]), row["id"]
            assert row["text"][start:end] == entity["text"], row["id"]
            span = set(range(start, end))
            assert not covered.intersection(span), row["id"]
            covered.update(span)
        # Every complete e-mail literal, including repeats and mailto targets,
        # must be annotated. Fully labeled documents cannot hide omitted copies.
        found_mail = {
            (match.start(), match.end())
            for match in re.finditer(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", row["text"])
        }
        annotated_mail = {
            (entity["start"], entity["end"])
            for entity in row["entities"]
            if entity["type"] == "PII.EMAIL"
        }
        assert found_mail == annotated_mail, row["id"]


def test_contact_network_and_secret_values_are_safe_fixtures():
    for row in corpus():
        for entity in row["entities"]:
            kind, value = entity["type"], entity["text"]
            if kind == "PII.EMAIL":
                assert value.rsplit("@", 1)[1].casefold() in {
                    "example.com", "example.org", "example.net"
                }
            elif kind == "PII.PHONE":
                compact = digits(value)
                if "ru_phone" in row["tags"]:
                    assert re.fullmatch(r"[78]495555017[1-8]", compact)
                    assert "invented_unassigned_not_asserted" in row["tags"]
                else:
                    if len(compact) == 10:
                        compact = "1" + compact
                    # NANP fictional-use exchange: 555-0100 through 555-0199.
                    assert re.fullmatch(r"1(?:202|212|415|650)55501\d{2}", compact)
            elif kind == "NETWORK.IP":
                address = ipaddress.ip_address(value)
                networks = (
                    "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24",
                    "2001:db8::/32",
                )
                assert any(address in ipaddress.ip_network(network) for network in networks)
            elif kind == "NETWORK.MAC":
                assert value.startswith(("02:", "02-"))
            elif kind == "SECRET.PASSWORD":
                assert value.startswith("EvalOnly!")
                assert row["policy_dependent"] is True
            elif kind == "SECRET.API_KEY":
                assert value.startswith("eval_synthetic_api_key_do_not_use_")
                assert row["policy_dependent"] is True


def test_checksum_based_identifiers_are_mathematically_valid():
    for entity in entities():
        kind, value = entity["type"], entity["text"]
        number = digits(value)
        if kind == "PII.SNILS":
            assert len(number) == 11
            checksum = sum(int(number[index]) * (9 - index) for index in range(9)) % 101
            assert int(number[-2:]) == (0 if checksum == 100 else checksum)
        elif kind == "PII.INN":
            assert len(number) == 12
            c1 = sum(int(number[i]) * weight for i, weight in enumerate((7, 2, 4, 10, 3, 5, 9, 4, 6, 8))) % 11 % 10
            c2 = sum(int(number[i]) * weight for i, weight in enumerate((3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8))) % 11 % 10
            assert number[-2:] == str(c1) + str(c2)
        elif kind in {"PII.OGRN", "PII.OGRNIP"}:
            assert len(number) == (13 if kind == "PII.OGRN" else 15)
            modulus = 11 if kind == "PII.OGRN" else 13
            assert int(number[-1]) == int(number[:-1]) % modulus % 10
        elif kind == "PII.BANK_CARD":
            transformed = [int(char) for char in number[::-1]]
            total = sum(char if index % 2 == 0 else sum(divmod(char * 2, 10)) for index, char in enumerate(transformed))
            assert total % 10 == 0, value
        elif kind == "PII.IBAN":
            compact = value.replace(" ", "")
            assert len(compact) == 22 and compact.startswith("DE")
            rearranged = compact[4:] + compact[:4]
            expanded = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
            assert int(expanded) % 97 == 1
            assert compact[4:12] == "00000000"  # deliberately unassigned bank code


def test_clean_controls_contain_no_contact_or_secret_literals():
    clean = [row for row in corpus() if not row["entities"]]
    assert all("negative_control" in row["tags"] for row in clean)
    assert {"numbers", "technical", "dates_not_birth", "topic", "format"} <= {
        row["category"] for row in clean
    }
    for row in clean:
        assert not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", row["text"])
        assert "EvalOnly!" not in row["text"]
        assert "eval_synthetic_api_key_do_not_use_" not in row["text"]
