"""경로 탐색: 비용 = 길이 × (1 + wC·C·공사활성 + wT·T·첨두 + wP·P[월])
최단 / 균형 / 최저노출 3개 경로. 우회 상한(최단 대비 배수) 안에서 가중치 배율 k를 키워가며 탐색."""
import json, math, sys
import networkx as nx

g = json.load(open('out/graph.json'))
MONTHS = g['months']
G = nx.Graph()
for i, e in enumerate(g['edges']):
    G.add_edge(e['u'], e['v'], i=i, **e)

def nearest(lon, lat):
    best, bd = None, 1e18
    for i, (x, y) in enumerate(g['nodes']):
        d = ((x - lon) * 88.2) ** 2 + ((y - lat) * 111) ** 2
        if d < bd: best, bd = i, d
    return best

def exposure(e, month, hour, weekday):
    c_on = 1.0 if (weekday and 7 <= hour < 18) else 0.15        # 비작업시간엔 잔류 분진만
    peak = 1.3 if hour in (7, 8, 17, 18, 19) else 1.0
    p = e['P'][MONTHS.index(month)] if month in MONTHS else 0.0
    return {'C': e['C'] * c_on, 'T': e['T'] * peak, 'P': p}

def route(o, d, w, k, month, hour, weekday):
    def cost(a, b, e):
        x = exposure(e, month, hour, weekday)
        return e['len'] * (1 + k * (w['C'] * x['C'] + w['T'] * x['T'] + w['P'] * x['P']))
    return nx.shortest_path(G, o, d, weight=cost)

def summarize(path, month, hour, weekday):
    L = dose = 0; parts = {'C': 0, 'T': 0, 'P': 0}
    for a, b in zip(path[:-1], path[1:]):
        e = G[a][b]; x = exposure(e, month, hour, weekday)
        L += e['len']
        for kk in parts: parts[kk] += e['len'] * x[kk]
    return L, parts

def plan(o, d, w, month, hour, weekday, caps=(1.15, 1.4)):
    sp = route(o, d, w, 0, month, hour, weekday)
    L0, _ = summarize(sp, month, hour, weekday)
    cands = []
    for k in (0, 0.25, 0.5, 1, 2, 4, 8, 16):
        p = route(o, d, w, k, month, hour, weekday)
        L, parts = summarize(p, month, hour, weekday)
        cands.append((k, p, L, sum(w[x] * parts[x] for x in parts), parts))
    res = {'shortest': cands[0]}
    for name, cap in zip(('balanced', 'cleanest'), caps):
        ok = [c for c in cands if c[2] <= cap * L0]
        res[name] = min(ok, key=lambda c: (c[3], c[2]))
    return res

if __name__ == '__main__':
    W = {'C': 2.0, 'T': 1.0, 'P': 1.5}
    tests = {
        '구반포역→반포한강공원(세빛섬)': ((126.9873, 37.5013), (126.9946, 37.5113)),
        '고속터미널역→잠원한강공원': ((127.0050, 37.5050), (127.0140, 37.5222)),
        '신반포역→잠원역': ((126.9958, 37.5034), (127.0114, 37.5128)),
    }
    for nm, (a, b) in tests.items():
        o, d = nearest(*a), nearest(*b)
        for month, hour in ((5, 8), (11, 14)):
            r = plan(o, d, W, month, hour, True)
            s = r['shortest']
            print(f'\n{nm} | {month}월 {hour}시(평일)')
            for key, (k, p, L, ex, parts) in r.items():
                print(f'  {key:9s} k={k:<5} {L:6.0f}m (+{(L/s[2]-1)*100:4.1f}%)  노출 {ex:7.0f} ({(ex/max(s[3],1e-9)-1)*100:+5.1f}%)  '
                      + ' '.join(f'{x}={parts[x]:.0f}' for x in parts))
