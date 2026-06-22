"""Deterministic run-name slug so each distinct submission config gets its own
output folder instead of overwriting the previous run.

Re-running the SAME config reuses (overwrites) the same folder — idempotent.
Different configs (different retriever / top_k_final / dense_weight /
expand_query) land in different folders, so multiple experiments coexist and
you can tell from the folder name which config produced which submission.zip.
"""


def run_slug(
    mode: str,
    top_k_final: int,
    dense_weight: float = 1.0,
    expand_query: bool = False,
    segment: bool = False,
) -> str:
    parts = [mode, f"kf{top_k_final}"]
    if mode == "hybrid":
        parts.append(f"dw{dense_weight}")
    if expand_query:
        parts.append("expand")
    if segment:
        parts.append("seg")
    return "_".join(parts)
