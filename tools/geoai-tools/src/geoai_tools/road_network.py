"""도로망(선형 객체) 처리: 세선화, 골격 → 그래프, 중심선 → 그래프, 위상·연결성 지표.

용어
- 골격(skeleton): 노면 마스크를 폭 1화소 선으로 깎은 것. Zhang-Suen 세선화 [Zhang 1984].
- 도로망 그래프: 노드 = 위치(m 좌표), 간선 = 노드 사이 선분(길이 m). 교차점은 차수 3 이상, 막다른 끝은 차수 1.
- 제어 노드(control node): 교차점·끝점과, 그 사이 간선 위에 일정 간격(spacing)으로 둔 점. APLS 경로의 양 끝이다.

지표
- 완전성·정확성·품질 [Wiedemann 1998]: 버퍼(buffer_m) 안에서 대응되는 중심선 길이 비율.
  완전성 = 예측 버퍼 안 참조 길이 / 참조 길이, 정확성 = 참조 버퍼 안 예측 길이 / 예측 길이,
  품질 = 완전성·정확성 / (완전성 − 완전성·정확성 + 정확성).
- APLS(Average Path Length Similarity) [Van Etten 2018]: 참조망 제어 노드 쌍의 최단 경로 길이 L 과 예측망에서
  대응 노드 쌍의 경로 길이 L' 를 비교해 1 − 평균(min(1, |L − L'| / L)).
  대응 노드가 버퍼 밖이거나 경로가 끊기면 1(최대 벌점).
  참조→예측과 예측→참조 두 방향 값의 조화 평균. 노면 화소가 맞아도 도로가 끊기면 크게 떨어진다.
- 경로 연결 비율: 참조망에서 이어진 제어 노드 쌍 중 예측망에서도 경로가 있는 비율.
"""

from __future__ import annotations

from collections.abc import Iterator

import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# 형태 연산(외부 영상 처리 라이브러리 없이 numpy 로)
# ---------------------------------------------------------------------------


def _shift_or(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    p = np.pad(m, r)
    h, w = m.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                out |= p[r + dy : r + dy + h, r + dx : r + dx + w]
    return out


def dilate(m: np.ndarray, r: int = 1) -> np.ndarray:
    return _shift_or(m.astype(bool), r) if r > 0 else m.astype(bool)


def erode(m: np.ndarray, r: int = 1) -> np.ndarray:
    return ~_shift_or(~m.astype(bool), r) if r > 0 else m.astype(bool)


def clean_mask(m: np.ndarray, r: int = 1) -> np.ndarray:
    """닫기(작은 틈 메움) 뒤 열기(가는 돌기 제거). 세선화 전에 잔가지를 줄인다."""
    closed = erode(dilate(m, r), r)
    return dilate(erode(closed, r), r)


def thin(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang-Suen 세선화. 연결성을 유지하며 폭 1화소 골격을 만든다."""
    img = np.pad(mask.astype(np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            seq = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            cond = (img[1:-1, 1:-1] == 1) & (b >= 2) & (b <= 6) & (a == 1)
            if step == 0:
                cond &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if cond.any():
                img[1:-1, 1:-1][cond] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


# ---------------------------------------------------------------------------
# 그래프 만들기
# ---------------------------------------------------------------------------


def skeleton_graph(skel: np.ndarray, x0: float, y0: float, gsd: float) -> nx.Graph:
    """골격 화소 → 그래프. 노드 키 (행, 열), 속성 xy = 화소 중심의 지도 좌표(m). 8방향 연결이되,
    직각으로 돌아가는 화소가 있으면 대각선 간선을 두지 않아 가짜 삼각형(가짜 교차점)을 막는다."""
    g = nx.Graph()
    rows, cols = np.nonzero(skel)
    h, w = skel.shape
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        g.add_node((r, c), xy=(x0 + (c + 0.5) * gsd, y0 - (r + 0.5) * gsd))

    def on(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and bool(skel[r, c])

    diag = gsd * float(np.sqrt(2))
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        if on(r, c + 1):
            g.add_edge((r, c), (r, c + 1), length=gsd)
        if on(r + 1, c):
            g.add_edge((r, c), (r + 1, c), length=gsd)
        if on(r + 1, c + 1) and not on(r, c + 1) and not on(r + 1, c):
            g.add_edge((r, c), (r + 1, c + 1), length=diag)
        if on(r + 1, c - 1) and not on(r, c - 1) and not on(r + 1, c):
            g.add_edge((r, c), (r + 1, c - 1), length=diag)
    return g


def _key(x: float, y: float) -> tuple[float, float]:
    return (round(x, 2), round(y, 2))


def lines_graph(lines: list[LineString], step: float) -> nx.Graph:
    """중심선 목록 → 그래프. 교차점에서 선을 나누고(noding), 간선을 step(m) 간격 점으로 잘게 나눈다."""
    g = nx.Graph()
    if not lines:
        return g
    merged = unary_union(lines)
    for seg in getattr(merged, "geoms", [merged]):
        if seg.geom_type != "LineString" or seg.length <= 0:
            continue
        n = max(1, int(np.ceil(seg.length / step)))
        inner = [seg.interpolate(i * seg.length / n) for i in range(1, n)]
        pts = [seg.coords[0], *((p.x, p.y) for p in inner), seg.coords[-1]]  # 끝점은 원래 좌표 그대로(교차점 공유)
        keys = [_key(x, y) for x, y, *_ in pts]
        for k, (x, y, *_) in zip(keys, pts, strict=True):
            g.add_node(k, xy=(x, y))
        for a, b in zip(keys, keys[1:], strict=False):
            if a != b:
                g.add_edge(a, b, length=float(np.hypot(a[0] - b[0], a[1] - b[1])))
    return g


def graph_length(g: nx.Graph) -> float:
    return float(sum(d["length"] for _, _, d in g.edges(data=True)))


def prune(g: nx.Graph, spur_m: float, min_component_m: float) -> dict:
    """짧은 잔가지(교차점에서 끝점까지 spur_m 미만)와 작은 조각(총 길이 min_component_m 미만)을 지운다."""
    removed_spurs = 0
    for node in [n for n in g if g.degree(n) == 1]:
        if node not in g or g.degree(node) != 1:
            continue
        path, length, prev, cur = [node], 0.0, None, node
        while True:
            nbrs = [n for n in g[cur] if n != prev]
            if len(nbrs) != 1:
                break
            nxt = nbrs[0]
            length += g[cur][nxt]["length"]
            prev, cur = cur, nxt
            if g.degree(cur) != 2:
                break
            path.append(cur)
        if g.degree(cur) >= 3 and length < spur_m:
            g.remove_nodes_from(path)
            removed_spurs += 1
    removed_parts = 0
    for comp in list(nx.connected_components(g)):
        sub = g.subgraph(comp)
        if graph_length(sub) < min_component_m:
            g.remove_nodes_from(list(comp))
            removed_parts += 1
    return {"spurs": removed_spurs, "fragments": removed_parts}


def chains(g: nx.Graph) -> Iterator[list]:
    """핵심 노드(차수 ≠ 2) 사이 노드 열. 핵심 노드가 없는 고리는 임의 노드 하나를 핵심으로 본다."""
    key = {n for n in g if g.degree(n) != 2}
    for comp in nx.connected_components(g):
        if not comp & key:
            key.add(min(comp))
    seen: set[frozenset] = set()
    for k in sorted(key):
        for nb in g[k]:
            e = frozenset((k, nb))
            if e in seen:
                continue
            seen.add(e)
            path, prev, cur = [k, nb], k, nb
            while cur not in key:
                nxt = next((n for n in g[cur] if n != prev), None)
                if nxt is None:
                    break
                seen.add(frozenset((cur, nxt)))
                path.append(nxt)
                prev, cur = cur, nxt
            yield path


def to_lines(g: nx.Graph, simplify: float = 0.0) -> list[LineString]:
    out = []
    for path in chains(g):
        coords = [g.nodes[n]["xy"] for n in path]
        if len(coords) >= 2:
            ln = LineString(coords)
            out.append(ln.simplify(simplify, preserve_topology=True) if simplify else ln)
    return out


def control_nodes(g: nx.Graph, spacing: float) -> list:
    """교차점·끝점 + 그 사이 간선 위 spacing(m)마다 한 점."""
    ctrl = {n for n in g if g.degree(n) != 2}
    for path in chains(g):
        ctrl.add(path[0])
        acc = 0.0
        for a, b in zip(path, path[1:], strict=False):
            acc += g[a][b]["length"]
            if acc >= spacing:
                ctrl.add(b)
                acc = 0.0
    return sorted(ctrl)


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------


def buffer_metrics(ref: list[LineString], pred: list[LineString], buffer_m: float) -> dict[str, float]:
    ref_ml, pred_ml = MultiLineString(ref) if ref else None, MultiLineString(pred) if pred else None
    ref_len = ref_ml.length if ref_ml else 0.0
    pred_len = pred_ml.length if pred_ml else 0.0
    comp = ref_ml.intersection(pred_ml.buffer(buffer_m)).length / ref_len if ref_ml and pred_ml else 0.0
    corr = pred_ml.intersection(ref_ml.buffer(buffer_m)).length / pred_len if ref_ml and pred_ml else 0.0
    denom = comp - comp * corr + corr
    return {
        "completeness": comp,
        "correctness": corr,
        "quality": comp * corr / denom if denom > 0 else 0.0,
        "reference_length_m": ref_len,
        "predicted_length_m": pred_len,
    }


def _snap(src: nx.Graph, nodes: list, dst: nx.Graph, buffer_m: float) -> dict:
    """src 제어 노드 → dst 의 가장 가까운 노드(buffer_m 이내). 없으면 None."""
    if dst.number_of_nodes() == 0:
        return dict.fromkeys(nodes)
    dst_nodes = list(dst.nodes)
    xy = np.array([dst.nodes[n]["xy"] for n in dst_nodes])
    out = {}
    for n in nodes:
        d = np.hypot(*(xy - np.array(src.nodes[n]["xy"])).T)
        i = int(d.argmin())
        out[n] = dst_nodes[i] if d[i] <= buffer_m else None
    return out


def _apls_one_way(g1: nx.Graph, g2: nx.Graph, spacing: float, buffer_m: float, max_nodes: int) -> tuple:
    """g1 기준 APLS 한 방향. (점수, 쌍 수, 경로 연결된 쌍 수)."""
    ctrl = control_nodes(g1, spacing)
    if len(ctrl) > max_nodes:
        pick = np.random.default_rng(0).choice(len(ctrl), max_nodes, replace=False)
        ctrl = [ctrl[i] for i in sorted(pick)]
    snap = _snap(g1, ctrl, g2, buffer_m)
    ctrl_set = set(ctrl)
    total, pairs, connected = 0.0, 0, 0
    for a in ctrl:
        l1 = nx.single_source_dijkstra_path_length(g1, a, weight="length")
        a2 = snap[a]
        l2 = nx.single_source_dijkstra_path_length(g2, a2, weight="length") if a2 is not None else {}
        for b, length in l1.items():
            if b == a or b not in ctrl_set or length <= 0:
                continue
            pairs += 1
            b2 = snap[b]
            if a2 is None or b2 is None or b2 not in l2:
                total += 1.0
                continue
            connected += 1
            total += min(1.0, abs(length - l2[b2]) / length)
    return (1.0 - total / pairs if pairs else None), pairs, connected


def apls(ref: nx.Graph, pred: nx.Graph, spacing: float = 20.0, buffer_m: float = 4.0, max_nodes: int = 400) -> dict:
    if ref.number_of_edges() == 0:
        return {"APLS": None, "APLS_ref_to_pred": None, "APLS_pred_to_ref": None, "connectivity": None, "pairs": 0}
    s1, pairs, connected = _apls_one_way(ref, pred, spacing, buffer_m, max_nodes)
    s2, _, _ = _apls_one_way(pred, ref, spacing, buffer_m, max_nodes) if pred.number_of_edges() else (0.0, 0, 0)
    s1, s2 = s1 or 0.0, s2 or 0.0
    score = 2 * s1 * s2 / (s1 + s2) if s1 + s2 > 0 else 0.0
    return {
        "APLS": score,
        "APLS_ref_to_pred": s1,
        "APLS_pred_to_ref": s2,
        "connectivity": connected / pairs if pairs else None,
        "pairs": pairs,
    }


def node_counts(g: nx.Graph) -> dict[str, int]:
    deg = dict(g.degree())
    # 골격 그래프는 교차점 하나가 이웃한 여러 화소로 나타나므로 이어진 교차 노드 묶음을 하나로 센다.
    hubs = g.subgraph([n for n, d in deg.items() if d >= 3])
    return {
        "junctions": nx.number_connected_components(hubs) if hubs.number_of_nodes() else 0,
        "dead_ends": sum(d == 1 for d in deg.values()),
        "components": nx.number_connected_components(g) if g.number_of_nodes() else 0,
    }
