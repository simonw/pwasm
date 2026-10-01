"""Tests for the spec runner's own literal parsing."""

import pytest

from spec_runner import parse_float_bits, parse_int, parse_sexprs, parse_string


@pytest.mark.parametrize(
    "text,bits,expected",
    [
        ("1.1", 32, 0x3F8CCCCD),
        ("-0x0p+0", 32, 0x80000000),
        ("0x1p-149", 32, 0x00000001),
        ("0x1.fffffep+127", 32, 0x7F7FFFFF),
        ("0x1.fffffefffffffffffp+127", 32, 0x7F7FFFFF),
        ("0x1.ffffffp+127", 32, 0x7F800000),  # tie rounds to even: infinity
        ("nan:0x200000", 32, 0x7FA00000),
        ("-inf", 32, 0xFF800000),
        ("1e39", 32, 0x7F800000),
        ("0x1p-1074", 64, 0x0000000000000001),
        ("0.1", 64, 0x3FB999999999999A),
        ("1_000.5", 64, 0x408F440000000000),
        ("-nan", 64, 0xFFF8000000000000),
    ],
)
def test_parse_float_bits(text, bits, expected):
    assert parse_float_bits(text, bits) == expected


def test_parse_int():
    assert parse_int("-1", 32) == 0xFFFFFFFF
    assert parse_int("0x7fff_ffff", 32) == 0x7FFFFFFF
    assert parse_int("0123", 32) == 123


def test_parse_sexprs_skips_comments_and_keeps_spans():
    text = '(module (; a (; nested ;) comment ;) (func)) ;; line\n(invoke "f" (i32.const 1))'
    exprs = parse_sexprs(text)
    assert exprs[0][0] == "module"
    assert text[exprs[0].start : exprs[0].end] == text.split(" ;;")[0]
    assert exprs[1][2] == ["i32.const", "1"]


def test_parse_string_escapes():
    assert parse_string(r'"a\00\ff\n\u{263a}"') == b"a\x00\xff\n" + "☺".encode()
