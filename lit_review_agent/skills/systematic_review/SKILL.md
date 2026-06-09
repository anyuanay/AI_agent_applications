---
name: systematic_review
description: Run a PRISMA-style systematic review — screen, dedupe, and format to
  checklist. Use when the user asks for a formal or systematic review, not a
  casual lookup.
tools: [screen_abstract, dedupe, format_prisma]
---
Procedure:
  1. screen_abstract() each candidate against the inclusion criteria.
  2. dedupe() the surviving set on title + DOI.
  3. format_prisma() the result using the flow counts in ./checklist.md.

Synthesis guidance for this review:
- Group the cited papers by subfield, then by influence (citation count).
- Every claim must carry a [S2:<id>] tag traceable to a screened source.
- State inclusion/exclusion counts at the top (identified, screened, included).
- If a subfield contributed zero included papers, say so explicitly. Do not pad.
