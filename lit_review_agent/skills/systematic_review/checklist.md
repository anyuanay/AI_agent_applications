# PRISMA flow reference (abbreviated)

A systematic review reports how the candidate set was narrowed. Fill these
counts in the synthesized review:

- **Identified**: records found by `search_papers` across all subtopics.
- **Screened**: records whose abstracts were read (`fetch_paper`).
- **Excluded**: screened records that failed the inclusion criteria, with a
  one-line reason category (off-topic, wrong year, not peer-reviewed).
- **Included**: records cited in the final review.

Inclusion criteria for this agent:
- Published in the requested year or range.
- Directly about the subtopic (not a passing mention).
- Has a retrievable abstract.

Report the four counts as a single line at the top of the review, e.g.:
`Identified 124 · screened 28 · excluded 19 · included 9.`
