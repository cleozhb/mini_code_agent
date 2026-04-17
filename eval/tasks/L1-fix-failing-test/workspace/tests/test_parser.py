from __future__ import annotations

from parser import parse_key_value_lines


def test_simple_pairs():
    result = parse_key_value_lines("a=1\nb=2")
    assert result == {"a": "1", "b": "2"}


def test_ignores_comments_and_blank():
    result = parse_key_value_lines("# this is a comment\n\nk=v")
    assert result == {"k": "v"}


def test_value_contains_equals_sign():
    # URL 的 query string 里带 '='，value 应保持原样
    result = parse_key_value_lines("url=http://example.com?a=b")
    assert result == {"url": "http://example.com?a=b"}


def test_value_can_be_empty_string():
    result = parse_key_value_lines("empty=")
    assert result == {"empty": ""}
