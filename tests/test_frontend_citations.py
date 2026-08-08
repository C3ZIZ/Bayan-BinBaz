"""Execute the frontend's citation-linking logic in Node against the real
index.html.

This exists because of a bug found in review: the `done` SSE event replaces the
positional `sources` array with `citations`, which is ordered by FIRST
APPEARANCE. A positional `sources[marker - 1]` lookup therefore linked each
marker to the wrong fatwa whenever the model wrote [2] before [1] — silently
attributing a ruling to a fatwa that does not contain it, which is the exact
failure this project exists to prevent.

The functions are extracted from index.html and run under a minimal DOM shim,
so the assertions are against the shipped source rather than a copy.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX_HTML = Path("frontend/index.html")

pytestmark = [
    pytest.mark.skipif(shutil.which("node") is None, reason="node not available"),
    pytest.mark.skipif(not INDEX_HTML.exists(), reason="frontend not present"),
]

DOM_SHIM = """
class El {
  constructor(tag) {
    this.tag = tag; this.children = []; this.attrs = {};
    this._text = ""; this.className = "";
  }
  get textContent() {
    return this.children.length
      ? this.children.map((c) => c.textContent).join("")
      : this._text;
  }
  set textContent(v) { this._text = v; this.children = []; }
  appendChild(c) { this.children.push(c); return c; }
}
const document = {
  createElement: (tag) => new El(tag),
  createTextNode: (t) => { const e = new El("#text"); e._text = t; return e; },
};
"""


def _extract(*names: str) -> str:
    src = INDEX_HTML.read_text(encoding="utf-8")
    out = []
    for name in names:
        m = re.search(rf"\n      function {name}\(", src)
        assert m, f"{name} not found in index.html"
        start = m.start()
        depth, i, started = 0, start, False
        while i < len(src):
            if src[i] == "{":
                depth += 1
                started = True
            elif src[i] == "}":
                depth -= 1
                if started and depth == 0:
                    break
            i += 1
        out.append(src[start : i + 1])
    return "\n".join(out)


def _run(script: str) -> dict:
    code = DOM_SHIM + _extract("markerMap", "renderAnswer") + script
    proc = subprocess.run(
        ["node", "-e", code], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_markers_link_to_the_right_fatwa_when_written_out_of_order():
    """[2] appearing before [1] must still resolve each marker correctly."""
    result = _run("""
      const container = document.createElement("div");
      // citations ordered by first appearance, as the `done` event sends them
      const citations = [
        { marker: 2, id: 202, title: "ب", link: "https://x/202" },
        { marker: 1, id: 101, title: "أ", link: "https://x/101" },
      ];
      renderAnswer(container, "أولا [2] ثم [1].", citations);
      const links = container.children.filter((c) => c.tag === "a");
      console.log(JSON.stringify({
        hrefs: links.map((l) => l.href),
        labels: links.map((l) => l.textContent),
      }));
    """)
    assert result["labels"] == ["2", "1"]
    # The decisive assertion: marker 2 -> fatwa 202, marker 1 -> fatwa 101.
    assert result["hrefs"] == ["https://x/202", "https://x/101"]


def test_markers_link_correctly_in_natural_order():
    result = _run("""
      const container = document.createElement("div");
      const citations = [
        { marker: 1, id: 101, title: "أ", link: "https://x/101" },
        { marker: 2, id: 202, title: "ب", link: "https://x/202" },
      ];
      renderAnswer(container, "حكم [1] وكذلك [2].", citations);
      const links = container.children.filter((c) => c.tag === "a");
      console.log(JSON.stringify({ hrefs: links.map((l) => l.href) }));
    """)
    assert result["hrefs"] == ["https://x/101", "https://x/202"]


def test_marker_without_a_matching_source_stays_plain_text():
    result = _run("""
      const container = document.createElement("div");
      const citations = [{ marker: 1, id: 101, title: "أ", link: "https://x/101" }];
      renderAnswer(container, "حكم [1] و [9].", citations);
      const links = container.children.filter((c) => c.tag === "a");
      console.log(JSON.stringify({
        n_links: links.length, text: container.textContent,
      }));
    """)
    assert result["n_links"] == 1
    assert "[9]" in result["text"]


def test_positional_sources_without_marker_field_still_work():
    """The `meta` event sends sources that already carry marker, but fall back
    to position if a payload ever omits it."""
    result = _run("""
      const container = document.createElement("div");
      const sources = [
        { id: 101, title: "أ", link: "https://x/101" },
        { id: 202, title: "ب", link: "https://x/202" },
      ];
      renderAnswer(container, "حكم [2].", sources);
      const links = container.children.filter((c) => c.tag === "a");
      console.log(JSON.stringify({ hrefs: links.map((l) => l.href) }));
    """)
    assert result["hrefs"] == ["https://x/202"]


def test_answer_text_is_preserved_around_markers():
    result = _run("""
      const container = document.createElement("div");
      const citations = [{ marker: 1, id: 101, title: "أ", link: "https://x/101" }];
      renderAnswer(container, "قرر الشيخ التحريم [1] والله أعلم.", citations);
      console.log(JSON.stringify({ text: container.textContent }));
    """)
    assert "قرر الشيخ التحريم" in result["text"]
    assert "والله أعلم" in result["text"]
