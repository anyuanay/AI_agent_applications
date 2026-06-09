"""
Text helpers for the memory layer (Part 6).

`chunk` is the right-sized, overlapping window from the chunking section.
`reorder_for_window` makes "lost in the middle" (Part 2 / Part 6) operational:
given items sorted best → worst, it places the strongest at the edges where the
model attends most, and buries the weakest in the middle.
"""

from __future__ import annotations


def chunk(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """~1 paragraph, sliding window, keeping `overlap` chars of context."""
    out: list[str] = []
    i = 0
    if not text:
        return out
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap  # keep context across the boundary
    return out


def reorder_for_window(items: list) -> list:
    """Place the best matches where the model attends most.

    `items` arrive sorted best → worst. The two strongest go to the front,
    the next two strongest to the back, and the weak middle stays buried.
    """
    if len(items) <= 4:
        return items
    edges, middle = [], []
    for i, c in enumerate(items):
        (edges if i < 2 or i >= len(items) - 2 else middle).append(c)
    head = edges[:2]
    tail = edges[2:]
    return head + middle + tail
