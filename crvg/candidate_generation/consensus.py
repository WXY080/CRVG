"""Paper Eq. (3): provenance-preserving consensus and weighted-IoU medoids."""
import copy

from crvg.utils.bbox import iou_xywh, valid_box
from crvg.utils.data import current_bbox


def connected_clusters(candidates, threshold=.45):
    parent = list(range(len(candidates)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(candidates)):
        for j in range(i):
            if iou_xywh(candidates[i]["bbox"], candidates[j]["bbox"]) >= threshold:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(candidates)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def consensus_update(row, transformed, dedup_iou=.95, cluster_iou=.45,
                     transform_weight=.5, current_bias=1., min_support=2, score_gate=.25):
    """Deduplicate boxes but retain all original-sample and view votes."""
    original = row["candidates"]
    pool = []
    observations = [(dict(bbox=current_bbox(row), source="current_system"), "current", None)]
    observations += [(c, "original", i) for i, c in enumerate(original)]
    observations += [(c, "view", c["view"]) for c in transformed]
    for candidate, kind, identity in observations:
        if kind != "current" and not valid_box(candidate["bbox"]):
            continue
        index = next((i for i, c in enumerate(pool)
                      if c["bbox"] == candidate["bbox"] or iou_xywh(c["bbox"], candidate["bbox"]) >= dedup_iou), None)
        if index is None:
            index = len(pool)
            pool.append({**copy.deepcopy(candidate), "ece_original_ids": [], "ece_views": [], "ece_current": False})
        target = pool[index]
        if kind == "current":
            target["ece_current"] = True
        elif kind == "original":
            target["ece_original_ids"].append(identity)
        elif identity not in target["ece_views"]:
            target["ece_views"].append(identity)
    count = max(len(original), 1)
    weights = [len(c["ece_original_ids"])/count + transform_weight*len(c["ece_views"])
               + current_bias*c["ece_current"] for c in pool]
    clusters = []
    for indices in connected_clusters(pool, cluster_iou):
        views = {v for i in indices for v in pool[i]["ece_views"]}
        has_current = any(pool[i]["ece_current"] for i in indices)
        score = (transform_weight*len(views) + sum(len(pool[i]["ece_original_ids"]) for i in indices)/count
                 + current_bias*has_current)
        clusters.append({"indices": indices, "score": score, "views": len(views), "current": has_current})
    current = next(c for c in clusters if c["current"])
    winner = max(clusters, key=lambda c: (c["score"], c["views"]))
    chosen = None
    if winner is not current and winner["views"] >= min_support and winner["score"]-current["score"] >= score_gate:
        index = max(winner["indices"], key=lambda i:
                    sum(weights[j]*iou_xywh(pool[i]["bbox"], pool[j]["bbox"]) for j in winner["indices"]))
        chosen = pool[index]
    return pool, chosen, {"current_score": current["score"], "winner_score": winner["score"],
                          "winner_view_support": winner["views"], "cluster_count": len(clusters)}
