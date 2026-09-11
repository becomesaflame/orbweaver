from orbweaver.editmatch import (
    TIER_EXACT,
    find_replacement,
    nearest_lines,
    reindent,
    strip_line_numbers,
    unified_diff,
)


def test_strip_line_numbers_pipe_and_colon():
    assert strip_line_numbers("  12|foo\n  13|bar") == "foo\nbar"
    assert strip_line_numbers("12: foo\n13: bar") == "foo\nbar"
    # Only strips when every non-blank line carries a prefix.
    assert strip_line_numbers("12|foo\nbar") == "12|foo\nbar"


def test_reindent_shifts_by_first_line_delta():
    assert reindent("    a\n        b\n", "    ", "\t") == "\ta\n\t    b\n"
    assert reindent("a\n  b", "", "    ") == "    a\n      b"
    assert reindent("    a\n    b", "    ", "") == "a\nb"
    assert reindent("x", "  ", "  ") == "x"


def test_reindent_uses_per_level_map_from_matched_lines():
    mapping = {"    ": "\t", "        ": "\t\t"}
    assert reindent("    a\n        b\n            c", "    ", "\t", mapping) == (
        "\ta\n\t\tb\n\t        c"
    )


def test_find_replacement_exact_wins_over_fallbacks():
    res = find_replacement("a\n  b\nc\n", "  b\n", "  B\n")
    assert res is not None
    assert res.tier == TIER_EXACT
    assert res.count == 1
    assert res.apply("a\n  b\nc\n", replace_all=False) == "a\n  B\nc\n"


def test_find_replacement_trimmed_keeps_file_indentation():
    content = "def f():\n\treturn 1\n"
    res = find_replacement(content, "    return 1\n", "    return 2\n")
    assert res is not None
    assert "line-trimmed" in res.tier
    assert res.apply(content, replace_all=False) == "def f():\n\treturn 2\n"


def test_find_replacement_anchor_requires_three_lines():
    content = "a\nx\nb\n"
    assert find_replacement(content, "a\nb\n", "z\n") is None
    res = find_replacement("a\nx\ny\nb\n", "a\nq\nr\nb\n", "z\n")
    assert res is not None
    assert "anchor" in res.tier
    assert res.apply("a\nx\ny\nb\n", replace_all=False) == "z\n"


def test_find_replacement_replace_all_reports_every_span():
    content = "\tk = 1\nsep\n\tk = 1\n"
    res = find_replacement(content, "    k = 1", "    k = 2")
    assert res is not None
    assert res.count == 2
    assert [s.line for s in res.spans] == [1, 3]
    assert res.apply(content, replace_all=True) == "\tk = 2\nsep\n\tk = 2\n"


def test_nearest_lines_reports_line_numbers():
    near = nearest_lines("alpha\ndef compute(x):\nomega\n", "def computes(x):")
    assert near == ["2|def compute(x):"]
    assert nearest_lines("alpha\n", "   \n") == []


def test_unified_diff_is_capped():
    before = "\n".join(str(i) for i in range(500))
    after = "\n".join(str(i + 1000) for i in range(500))
    out = unified_diff(before, after, "f.txt", cap=50)
    lines = out.splitlines()
    assert lines[0] == "--- a/f.txt"
    assert lines[1] == "+++ b/f.txt"
    assert len(lines) == 51
    assert lines[-1].startswith("[diff truncated; ")
