"""한강공원(반포·잠원) 피크닉 후보 격자점별 노출 계산 → out/picnic.json

각 점(40m 격자)에서
  C  공사장   : max exp(-d/75)
  T  교통배기 : max 도로등급 × exp(-d/50)      (올림픽대로 영향이 핵심)
  P[m] 꽃가루 : 대상지 수목 커널합(링크와 같은 식·같은 정규화) + 수변 버드나무류 추정(3–4월)
                수변 버드나무 = 0.6 × exp(-한강 물가까지 거리/40m)  ← 한강생태지도(2022) 정성 정보 기반 가정
"""
import json, math
import numpy as np
from shapely.geometry import shape, Point, Polygon, LineString
from shapely.ops import unary_union, transform
from shapely.strtree import STRtree
from pyproj import Transformer

T2M = Transformer.from_crs(4326, 5186, always_xy=True).transform
INV = Transformer.from_crs(5186, 4326, always_xy=True).transform
D = json.load(open('data/trees_D.json'))
G = json.load(open('out/graph.json'))
MONTHS = G['months']

parks = {k: transform(T2M, shape(g)) for k, g in D['parks'].items()}
river = transform(T2M, shape(D['river']))
river_edge = river.boundary

# 간선도로
nodes, ways = {}, {}
for f in ['osm', 'w', 'e', 'n1', 'n2']:
    for el in json.load(open(f'data/{f}.json'))['elements']:
        if el['type'] == 'node': nodes[el['id']] = T2M(el['lon'], el['lat'])
        elif el['type'] == 'way': ways[el['id']] = el
MAJOR = {'motorway': 1.0, 'motorway_link': 0.8, 'trunk': 1.0, 'trunk_link': 0.8,
         'primary': 0.7, 'primary_link': 0.6, 'secondary': 0.45, 'secondary_link': 0.4, 'tertiary': 0.25}
mlines, mw = [], []
for w in ways.values():
    hw = w.get('tags', {}).get('highway')
    nd = [nodes[n] for n in w['nodes'] if n in nodes]
    if hw in MAJOR and len(nd) > 1: mlines.append(LineString(nd)); mw.append(MAJOR[hw])
mtree = STRtree(mlines)

sites = [Polygon([T2M(*p) for p in s['ll']]) for s in G['sites']]

LV = {'상': 1.0, '중': 0.5, '하': 0.0}
txy = {m: [] for m in MONTHS}; tw = {m: [] for m in MONTHS}
for lon, lat, sp, lvl, *_ in D['trees']:
    if LV[lvl] == 0: continue
    for m in D['species'][sp]['months']:
        txy[m].append(T2M(lon, lat)); tw[m].append(LV[lvl])
txy = {m: np.array(v) for m, v in txy.items()}; tw = {m: np.array(v) for m, v in tw.items()}
P95 = 8.394518070684692  # build_graph_seoul.py 링크 정규화 값과 동일

# 그래프 노드 (접근 경로용)
gn = np.array([T2M(*n) for n in G['nodes']])

pts = []
for name, poly in parks.items():
    minx, miny, maxx, maxy = poly.bounds
    for x in np.arange(minx, maxx, 40):
        for y in np.arange(miny, maxy, 40):
            p = Point(x, y)
            if not poly.contains(p) or river.contains(p): continue
            pts.append((name, p))
print('grid points', len(pts))

out = []
for name, p in pts:
    C = max((math.exp(-s.distance(p) / 75) for s in sites), default=0.0)
    idx = mtree.query(p.buffer(300))
    T = max((mw[i] * math.exp(-mlines[i].distance(p) / 50) for i in idx), default=0.0)
    dw = river_edge.distance(p)
    willow = 0.6 * math.exp(-dw / 40)
    P = []
    for m in MONTHS:
        d = np.sqrt(((txy[m] - [p.x, p.y]) ** 2).sum(1))
        v = float((np.exp(-d / 30) * tw[m]).sum()) / P95
        if m in (3, 4): v += willow
        P.append(round(min(v, 1.5), 3))
    lon, lat = INV(p.x, p.y)
    ni = int(np.argmin(((gn - [p.x, p.y]) ** 2).sum(1)))
    out.append({'ll': [round(lon, 6), round(lat, 6)], 'park': name, 'C': round(C, 3), 'T': round(T, 3), 'P': P,
                'river': round(dw), 'node': ni})

json.dump({'months': MONTHS, 'pts': out, 'grid': 40,
           'note': '수변 버드나무류(3–4월)는 한강생태지도 2022 정성 정보 기반 추정치'},
          open('out/picnic.json', 'w'), ensure_ascii=False, separators=(',', ':'))
a = np.array([[o['C'], o['T'], max(o['P'])] for o in out])
print('C/T/Pmax percentiles', np.percentile(a, [10, 50, 90], axis=0).round(3))
