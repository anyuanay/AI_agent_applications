---
name: cluster-classes
description: Group induced leaf classes into families and check that the clustering
  agrees with the LLM-proposed parents. Stage 3 of ontology extraction. Use to
  sanity-check a proposed hierarchy's shape before expert review, or to suggest
  families when induction returned a flat list.
tools: [cluster_into_families]
---
Procedure:
  1. Take the `ProposedClass` list from `induce-schema`.
  2. Split it into families (parent == the root concept) and leaves.
  3. cluster_into_families() to assign each leaf to its most similar family by
     label-token overlap (a stand-in for sentence-embedding clustering).
  4. Compare each leaf's clustered family against its LLM-proposed parent. Agreement
     is a coherence signal; disagreement is a flag for `refine-ontology`.

Guidance for this stage:
- This is a coherence check, not a decision maker. When the cluster and the
  LLM-proposed parent agree, confidence in the hierarchy rises. When they disagree,
  do not silently re-parent; surface it for review.
- Token overlap is the dependency-light default. In production swap it for
  sentence-BERT embeddings plus agglomerative clustering; the interface (leaves,
  families -> assignment) stays the same.
- Evaluate cluster quality before trusting it. See `references/coherence_metrics.md`
  for silhouette and manual-inspection criteria.

Run:
    python scripts/cluster_families.py          # cluster leaves and report agreement
