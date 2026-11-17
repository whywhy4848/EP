"""반포·잠원 보행 네트워크 구축 + 링크별 노출 지표 계산.

입력: data/*.json (OSM Overpass), data/trees_D.json (이전 수목 분석), data/constr.json
출력: out/graph.json (브라우저용 압축 그래프)

링크 노출 지표 (각 0~1, 링크를 10m 간격으로 샘플링해 평균)
  C  공사장   : exp(-d/75),  d = 공사장 폴리곤까지 거리(내부 0)
  T  교통배기 : max_road w_class * exp(-d/50)   (대로변 오염 100~200m 내 급감, Karner et al. 2010)
  P[m] 꽃가루 : 해당 월 비산 수목 커널합 sum(w_lvl * exp(-d/30)), 95퍼센타일로 0~1 정규화
"""
import json, math, glob
import numpy as np
from shapely.geometry import shape, Polygon, LineString, Point
from shapely.ops import unary_union, transform
from shapely.strtree import STRtree
import networkx as nx
from pyproj import Transformer

T2M = Transformer.from_crs(4326, 5186, always_xy=True).transform  # Korea 2000 중부원점 (m)
D = json.load(open('data/trees_D.json'))

# ---------- 대상지 ----------
dongs = {k: shape(g) for k, g in D['dongs'].items()}
parks = {k: shape(g) for k, g in D['parks'].items()}
area = unary_union(list(dongs.values()) + list(parks.values()))
area_m = transform(T2M, area)
clip_m = area_m.buffer(250)

# ---------- OSM 보행망 ----------
nodes, ways = {}, {}
for f in ['osm', 'w', 'e', 'n1', 'n2']:
    for el in json.load(open(f'data/{f}.json'))['elements']:
        if el['type'] == 'node':
            nodes[el['id']] = (el['lon'], el['lat'])
        elif el['type'] == 'way':
            ways[el['id']] = el

NO_WALK = {'busway', 'motorway', 'motorway_link', 'construction', 'proposed', 'raceway', 'bus_guideway', 'abandoned', 'platform'}
MAJOR = {'motorway': 1.0, 'motorway_link': 0.8, 'trunk': 1.0, 'trunk_link': 0.8,
         'primary': 0.7, 'primary_link': 0.6, 'secondary': 0.45, 'secondary_link': 0.4, 'tertiary': 0.25}

xy = {n: T2M(*ll) for n, ll in nodes.items()}

major_lines, major_w = [], []
G = nx.Graph()
for wid, w in ways.items():
    tg = w.get('tags', {})
    hw = tg.get('highway')
    nds = [n for n in w['nodes'] if n in xy]
    if hw in MAJOR and len(nds) > 1:
        major_lines.append(LineString([xy[n] for n in nds])); major_w.append(MAJOR[hw])
    if hw is None or hw in NO_WALK or tg.get('foot') == 'no' or tg.get('access') in ('private', 'no') and tg.get('foot') not in ('yes', 'designated'):
        continue
    if hw in ('trunk', 'trunk_link') and tg.get('foot') not in ('yes', 'designated') and tg.get('sidewalk') in (None, 'no', 'none'):
        continue  # 보도 없는 간선은 제외
    for a, b in zip(nds[:-1], nds[1:]):
        if a == b: continue
        G.add_edge(a, b, hw=hw, name=tg.get('name', ''))

# 대상지 클립 + 최대 연결요소
keep = [n for n in G.nodes if clip_m.contains(Point(xy[n]))]
G = G.subgraph(keep).copy()
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
print('raw nodes/edges', G.number_of_nodes(), G.number_of_edges())

# degree-2 노드 압축 → 교차점 간 링크(폴리라인)
def simplify(G):
    H = nx.Graph()
    ends = {n for n in G.nodes if G.degree(n) != 2}
    seen = set()
    for s in ends:
        for nb in G.neighbors(s):
            if (s, nb) in seen: continue
            path = [s, nb]
            while path[-1] not in ends:
                nxt = [x for x in G.neighbors(path[-1]) if x != path[-2]]
                if not nxt: break
                path.append(nxt[0])
                if path[-1] == s: break
            for a, b in zip(path[:-1], path[1:]):
                seen.add((a, b)); seen.add((b, a))
            u, v = path[0], path[-1]
            if u == v: continue
            if H.has_edge(u, v) and H[u][v]['len'] <= LineString([xy[n] for n in path]).length:
                continue
            H.add_edge(u, v, path=path, hw=G[path[0]][path[1]]['hw'], name=G[path[0]][path[1]]['name'],
                       len=LineString([xy[n] for n in path]).length)
    return H

H = simplify(G)
print('simplified nodes/edges', H.number_of_nodes(), H.number_of_edges())

# ---------- 서울시 도보 네트워크로 교체 (OSM은 간선도로 위치만 사용) ----------
import pandas as pd
from shapely import wkt as _wkt
SEOUL = 'raw/seoul_walk_network_seocho.csv'  # 서울 열린데이터광장 '서울시 자치구별 도보 네트워크 공간정보' (서초구)
df = pd.read_csv(SEOUL, encoding='cp949', dtype=str)
Ldf = df[(df['노드링크 유형'] == 'LINK') & (df['링크 유형 코드'].str[0] == '1')]   # 보행자 통행 가능 링크만
clip_ll = area.buffer(0.0025)
H_osm, nodes_osm, xy_osm = H, nodes, xy
H = nx.Graph(); nodes = {}; xy = {}
for r in Ldf.itertuples(index=False):
    g = _wkt.loads(r[4])
    if not clip_ll.intersects(g): continue
    cs = list(g.coords); a, b = r[7], r[8]
    ids = [('n', a)] + [('v', r[5], i) for i in range(1, len(cs) - 1)] + [('n', b)]
    for k, c in zip(ids, cs):
        if k not in nodes or k[0] == 'v': nodes[k] = c; xy[k] = T2M(*c)
    code = r[6]
    if r[19] == '1': hw = 'crossing'
    elif r[20] == '1': hw = 'park'
    elif r[21] == '1': hw = 'indoor'
    elif code in ('1000', '1011', '1010'): hw = 'footway'   # 보행 전용·보행+자전거
    else: hw = 'shared'                                     # 보행자+차량 혼용
    L = LineString([xy[k] for k in ids]).length
    if a == b: continue
    if H.has_edge(ids[0], ids[-1]) and H[ids[0]][ids[-1]]['len'] <= L: continue
    H.add_edge(ids[0], ids[-1], path=ids, hw=hw, name='', len=L)
# 한강공원 내부는 서울시 망에 없음 → OSM 산책로로 보완, 40m 이내 서울시 노드와 연결
park_m = transform(T2M, unary_union(list(parks.values()))).buffer(15)
seoul_nodes = [k for k in H.nodes]
s_tree = STRtree([Point(xy[k]) for k in seoul_nodes])
added = 0
for u, v, d in H_osm.edges(data=True):
    line = LineString([xy_osm[n] for n in d['path']])
    if line.intersection(park_m).length < 0.6 * line.length: continue
    ids = [('o', n) for n in d['path']]
    for k, n in zip(ids, d['path']): nodes[k] = nodes_osm[n]; xy[k] = xy_osm[n]
    H.add_edge(ids[0], ids[-1], path=ids, hw='park', name='', len=d['len']); added += 1
    for end in (ids[0], ids[-1]):
        i = s_tree.nearest(Point(xy[end])); s = seoul_nodes[i]
        dd = Point(xy[end]).distance(Point(xy[s]))
        if dd < 40 and not H.has_edge(end, s): H.add_edge(end, s, path=[end, s], hw='park', name='', len=max(dd, 1))
print('hangang park osm links added', added)
H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
print('seoul network nodes/edges', H.number_of_nodes(), H.number_of_edges(), 'km', round(sum(d['len'] for *_, d in H.edges(data=True))/1000, 1))

# ---------- 노출 소스 ----------
cj = json.load(open('data/constr.json'))
sites = []
for el in cj['elements']:
    p = Polygon([T2M(n['lon'], n['lat']) for n in el['geometry']])
    if p.distance(area_m) < 300:
        sites.append({'id': el['id'], 'name': el.get('tags', {}).get('name', 'OSM 공사구역'), 'poly': p,
                      'll': [[round(n['lon'], 6), round(n['lat'], 6)] for n in el['geometry']]})
print('construction sites', len(sites))

m_tree = STRtree(major_lines)

LV = {'상': 1.0, '중': 0.5, '하': 0.0}
trees_by_m = {m: [] for m in range(1, 13)}
for lon, lat, sp, lvl, *_ in D['trees']:
    if LV[lvl] == 0: continue
    for m in D['species'][sp]['months']:
        trees_by_m[m].append((T2M(lon, lat), LV[lvl]))
MONTHS = [m for m in range(1, 13) if trees_by_m[m]]
tree_xy = {m: np.array([t[0] for t in trees_by_m[m]]) for m in MONTHS}
tree_w = {m: np.array([t[1] for t in trees_by_m[m]]) for m in MONTHS}
print('pollen months', MONTHS)

def samples(line, step=10):
    n = max(2, int(line.length // step) + 1)
    return [line.interpolate(i / (n - 1), normalized=True) for i in range(n)]

def site_exposure(pt, polys):
    return max((math.exp(-p.distance(pt) / 75) for p in polys), default=0.0)

polys = [s['poly'] for s in sites]
raw = []
for u, v, d in H.edges(data=True):
    line = LineString([xy[n] for n in d['path']])
    pts = samples(line)
    C = np.mean([site_exposure(p, polys) for p in pts])
    Tv = []
    for p in pts:
        idx = m_tree.query(p.buffer(250))
        Tv.append(max((major_w[i] * math.exp(-major_lines[i].distance(p) / 50) for i in idx), default=0.0))
    P = {}
    arr = np.array([[p.x, p.y] for p in pts])
    for m in MONTHS:
        dd = np.sqrt(((arr[:, None, :] - tree_xy[m][None, :, :]) ** 2).sum(-1))
        P[m] = float(np.mean((np.exp(-dd / 30) * tree_w[m]).sum(1)))
    raw.append((u, v, d, C, float(np.mean(Tv)), P))

# 꽃가루 정규화: 전 월 통합 95퍼센타일 = 1
allP = np.array([r[5][m] for r in raw for m in MONTHS])
p95 = np.percentile(allP[allP > 0], 95)
print('pollen p95', p95)

# ---------- 출력 ----------
INV = Transformer.from_crs(5186, 4326, always_xy=True).transform
node_ids = {n: i for i, n in enumerate(H.nodes)}
out_nodes = [[round(nodes[n][0], 6), round(nodes[n][1], 6)] for n in H.nodes]
out_edges = []
for u, v, d, C, Tt, P in raw:
    geom = [[round(nodes[n][0], 6), round(nodes[n][1], 6)] for n in d['path']]
    out_edges.append({
        'u': node_ids[u], 'v': node_ids[v], 'len': round(d['len'], 1), 'hw': d['hw'], 'nm': d['name'],
        'C': round(float(C) * (0.2 if d['hw'] == 'indoor' else 1), 3), 'T': round(Tt * (0.2 if d['hw'] == 'indoor' else 1), 3),
        'P': [round(min(P[m] / p95, 1.5) * (0.2 if d['hw'] == 'indoor' else 1), 3) for m in MONTHS],  # 건물 안(지하상가 등)은 외기 노출 20%
        'g': geom,
    })
import os; os.makedirs('out', exist_ok=True)
json.dump({'nodes': out_nodes, 'edges': out_edges, 'months': MONTHS,
           'sites': [{'id': s['id'], 'name': s['name'], 'll': s['ll']} for s in sites],
           'dongs': D['dongs'], 'parks': D['parks'], 'river': D['river']},
          open('out/graph.json', 'w'), ensure_ascii=False, separators=(',', ':'))
print('written', os.path.getsize('out/graph.json') // 1024, 'KB')
