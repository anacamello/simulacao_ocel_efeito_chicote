"""
experiment.py — orquestração dos experimentos
=============================================

PAPEL NO PROJETO
----------------
Liga as peças: para cada cenário e cada replicação roda a simulação
(``simulation.py``), calcula as métricas (``metrics.py``) e junta os
resultados. Também:

* calcula as MESMAS métricas direto no log real (linha de base / referência);
* monta a tabela de validação (cenário Atual × log), como na avaliação do OCPS;
* calcula médias e intervalos de confiança de 95% entre replicações;
* exporta o log simulado em OCEL 2.0 (mesmo formato do arquivo original).

REPLICAÇÕES E NÚMEROS ALEATÓRIOS COMUNS
---------------------------------------
Uma simulação estocástica precisa de várias replicações (rodadas com
sementes diferentes) para estimar médias e intervalos de confiança.
A replicação r de TODOS os cenários usa a mesma semente (base_seed + r):
assim os cenários "veem" sorteios semelhantes de demanda e lead time, e a
diferença entre eles reflete a política, não o acaso (técnica de "números
aleatórios comuns", que reduz a variância das comparações).
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import metrics as M
from .parameters import (ModelParameters, TIME_ATTRIBUTES, requested_quantity_series,
                         ordered_quantity_series, net_stock_series)
from .policies import ScenarioConfig
from .simulation import SupplyChainSimulation

# Na execução paralela cada processo recebe os parâmetros UMA vez (ao iniciar)
# e os guarda nesta variável global, em vez de recebê-los a cada replicação.
_PARAMS: ModelParameters | None = None


def _init_worker(params):
    """Executada uma vez em cada processo paralelo."""
    global _PARAMS
    _PARAMS = params


def il_maps(params: ModelParameters):
    """Dicionários auxiliares: {IL: Location} e {IL: {fornecedor: proporção}}."""
    il_loc = {il: p.location for il, p in params.item_locations.items()}
    il_sup = {il: p.supplier_probs for il, p in params.item_locations.items()}
    return il_loc, il_sup


# ===========================================================================
# UMA replicação de UM cenário
# ===========================================================================
def run_replication(cfg_dict: dict, seed: int, rep: int, opts: dict, params: ModelParameters | None = None) -> dict:
    """Roda a simulação e devolve um dicionário com métricas e séries resumidas.

    cfg_dict : cenário (dicionário, para poder ser enviado a outro processo)
    seed     : semente desta replicação
    rep      : nº da replicação (0, 1, 2...)
    opts     : opções gerais (horizonte, aquecimento, modo de amostragem...)"""
    params = params or _PARAMS
    cfg = ScenarioConfig(**cfg_dict)

    # 1) simula
    sim = SupplyChainSimulation(
        params, cfg, seed=seed,
        horizon_days=opts.get("horizon_days"),
        include_exceptions=opts.get("include_exceptions", True),
        # o OCEL completo só é guardado na 1ª replicação dos cenários escolhidos (economiza memória)
        record_ocel=(rep == 0 and cfg.label() in (opts.get("record_ocel_scenarios") or [])),
        sampling_mode=opts.get("sampling_mode", "parametric"),
        demand_model=opts.get("demand_model", "tendencia_sazonal"),
        initial_state=opts.get("initial_state", "inicio_log"),
        inherit_backorders=opts.get("inherit_backorders", False),
    ).run()

    # 2) mede o efeito chicote e os KPIs
    il_loc, il_sup = il_maps(params)
    warm = opts.get("warmup_days", 56)
    bdays = opts.get("bucket_days", 7)
    net = sim.on_hand_eod - sim.backorder_eod   # estoque líquido diário
    bw = M.bullwhip(sim.demand[:, : sim.ndays], sim.orders[:, : sim.ndays], sim.ils, il_loc,
                    {k: v[: sim.ndays] for k, v in sim.orders_by_supplier.items()}, il_sup,
                    net_stock=net[:, : sim.ndays], bucket_days=bdays, warmup_days=warm)
    k = M.kpis(sim, params, cfg, warmup_days=warm)

    # 3) empacota o resultado (só dados leves; a simulação em si é descartada)
    res = {
        "scenario": cfg.label(),
        "policy": cfg.policy,
        "rep": rep,
        "seed": seed,
        "kpis": k,
        "bw_summary": bw["summary"],
        "bw_levels": bw["levels"],
        "bw_per_il": bw["per_il"].to_dict("records"),
        "counts": dict(sim.counts),              # eventos por atividade (validação)
        "obj_counts": dict(sim.obj_counts),      # objetos criados por tipo
        "so_variants": dict(sim.so_variants),
        "po_variants": dict(sim.po_variants),
        "weekly_demand": M.bucket(sim.demand[:, : sim.ndays], 7),   # para os gráficos de séries
        "weekly_orders": M.bucket(sim.orders[:, : sim.ndays], 7),
        "daily_net_stock": net[:, : sim.ndays].astype(np.float32),
        "lead_times": np.asarray(sim.lead_times[:2000], dtype=np.float32),
        "ils": sim.ils,
    }
    if sim.record_ocel:
        res["ocel_json"] = to_ocel_json(sim, params)
    return res


# ===========================================================================
# TODAS as replicações de TODOS os cenários
# ===========================================================================
def run_experiment(params: ModelParameters, scenarios: list[ScenarioConfig], n_reps: int, base_seed: int = 42,
                   opts: dict | None = None, progress=None, parallel: bool = True, max_workers: int | None = None):
    """Executa n_reps replicações de cada cenário.

    Se ``parallel`` for True, distribui as replicações entre os núcleos do
    processador (até 8). Se o paralelismo falhar (comum em alguns ambientes
    Windows), roda tudo em sequência automaticamente.
    ``progress(fração, mensagem)`` atualiza a barra de progresso da interface."""
    opts = opts or {}
    # lista de tarefas: (cenário, semente, nº da replicação) — números aleatórios comuns
    tasks = [(sc.to_dict(), base_seed + r, r) for sc in scenarios for r in range(n_reps)]
    out = []
    total = len(tasks)
    if parallel and total > 1:
        workers = max_workers or max(1, min(os.cpu_count() or 1, total, 8))
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(params,)) as ex:
                futs = [ex.submit(run_replication, c, s, r, opts) for c, s, r in tasks]
                for i, f in enumerate(as_completed(futs)):   # à medida que terminam
                    out.append(f.result())
                    if progress:
                        progress((i + 1) / total, f"{i + 1}/{total} replicações concluídas")
            return _sort(out, scenarios)
        except Exception as e:
            if progress:
                progress(0.0, f"Execução paralela indisponível ({type(e).__name__}); rodando em sequência…")
            out = []
    # execução sequencial
    for i, (c, s, r) in enumerate(tasks):
        out.append(run_replication(c, s, r, opts, params=params))
        if progress:
            progress((i + 1) / total, f"{i + 1}/{total} replicações concluídas")
    return _sort(out, scenarios)


def _sort(out, scenarios):
    """Ordena os resultados por cenário (ordem da lista) e por replicação."""
    order = {sc.label(): i for i, sc in enumerate(scenarios)}
    return sorted(out, key=lambda r: (order.get(r["scenario"], 99), r["rep"]))


# ===========================================================================
# Agregação estatística
# ===========================================================================
def ci95(x) -> tuple[float, float, float]:
    """Média e intervalo de confiança de 95% (t de Student):
    média ± t(0,975; n−1) · s / √n.  Retorna (média, limite inferior, limite superior)."""
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan   # com 1 replicação não há como estimar o IC
    from scipy import stats
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return m, m - h, m + h


def results_frame(results: list[dict]) -> pd.DataFrame:
    """Tabela com uma linha por (cenário, replicação) e todas as métricas."""
    rows = []
    for r in results:
        row = {"cenário": r["scenario"], "política": r["policy"], "replicação": r["rep"]}
        row.update(r["bw_summary"])
        row.update({f"BWE {k}": v for k, v in r["bw_levels"].items() if not k.startswith("Rede")})
        row.update(r["kpis"])
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Uma linha por cenário com a média e o IC 95% de cada métrica em ``cols``."""
    rows = []
    for sc, g in df.groupby("cenário", sort=False):
        row = {"cenário": sc, "n": len(g)}
        for c in cols:
            m, lo, hi = ci95(g[c])
            row[c] = m
            row[c + " (IC95%)"] = f"[{lo:.3g}; {hi:.3g}]" if np.isfinite(lo) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Linha de base: métricas calculadas sobre o LOG REAL
# ===========================================================================
def log_baseline(d, params: ModelParameters, warmup_days: int = 56, bucket_days: int = 7) -> dict:
    """Aplica ao OCEL original exatamente as mesmas medidas da simulação,
    para que real e simulado sejam comparáveis."""
    # séries diárias do log: demanda, pedidos e estoque líquido
    dem = requested_quantity_series(d)
    orders = ordered_quantity_series(d)
    ns = net_stock_series(d).reindex(dem.index)
    ils = list(dem.index)
    il_loc, il_sup = il_maps(params)

    # pedidos recebidos por cada fornecedor no log (via relação O2O PO → Supplier)
    o2o = d.o2o
    po_sup = o2o[(o2o["ocel:type"] == "PurchaseOrderItem") & (o2o["ocel:type_2"] == "Supplier")].set_index("ocel:oid")["ocel:oid_2"]
    rel = d.relations[(d.relations["ocel:type"] == "PurchaseOrderItem") & (d.relations["ocel:activity"] == "Purchase Order Item Created")]
    po = rel[["ocel:oid", "ocel:timestamp"]].copy()
    po["sup"] = po["ocel:oid"].map(po_sup)
    po["q"] = po["ocel:oid"].map(d.objects.set_index("ocel:oid")["ordered_quantity"]).astype(float)
    po["day"] = ((po["ocel:timestamp"] - d.start).dt.total_seconds() // 86400).astype(int)
    by_sup = {}
    for sup, g in po.groupby("sup"):
        arr = np.zeros(dem.shape[1])
        for dd, q in g.groupby("day")["q"].sum().items():
            if 0 <= dd < len(arr):
                arr[dd] += q
        by_sup[sup] = arr

    # efeito chicote do log
    bw = M.bullwhip(dem.to_numpy(), orders.to_numpy(), ils, il_loc, by_sup, il_sup,
                    net_stock=ns.to_numpy(), bucket_days=bucket_days, warmup_days=warmup_days)

    # KPIs do log
    ev = d.events
    n_so = (ev["ocel:activity"] == "Sales Order Item Created").sum()
    n_bo = (ev["ocel:activity"] == "Backorder Registered").sum()
    issued = ev.loc[ev["ocel:activity"] == "Goods Issue", "quantity"].sum()
    return {
        "bw": bw,
        "counts": ev["ocel:activity"].value_counts().to_dict(),
        "weekly_demand": M.bucket(dem.to_numpy(), 7),
        "weekly_orders": M.bucket(orders.to_numpy(), 7),
        "daily_net_stock": ns.to_numpy(),
        "ils": ils,
        "kpis": {
            "nível de serviço (fill rate)": float(issued / dem.to_numpy().sum()),
            "% pedidos de venda com backorder": float(n_bo / n_so * 100),
            "nº pedidos de compra": int((ev["ocel:activity"] == "Purchase Order Item Created").sum()),
        },
    }


def validation_table(log: dict, asis_results: list[dict]) -> pd.DataFrame:
    """Validação (como no OCPS): para cada atividade compara o nº de eventos
    no log com a média (e IC 95%) das replicações do cenário Atual."""
    acts = sorted(set(log["counts"]) | {a for r in asis_results for a in r["counts"]})
    rows = []
    for a in acts:
        sim = [r["counts"].get(a, 0) for r in asis_results]
        m, lo, hi = ci95(sim)
        lv = log["counts"].get(a, 0)
        rows.append({"atividade": a, "log": lv, "simulado (média)": m,
                     "IC95%": f"[{lo:.0f}; {hi:.0f}]" if np.isfinite(lo) else "—",
                     "erro relativo %": (m - lv) / lv * 100 if lv else np.nan})
    return pd.DataFrame(rows).sort_values("log", ascending=False)


# ===========================================================================
# Exportação do log simulado em OCEL 2.0 (JSON)
# ===========================================================================
def _iso(base, t: float) -> str:
    """Converte o tempo da simulação (dias desde ``base``) em data ISO 8601."""
    return (base + pd.Timedelta(days=float(t))).strftime("%Y-%m-%dT%H:%M:%SZ")


def _val(name, v, base):
    """Prepara um valor para o JSON (datas em ISO, tipos numpy → Python)."""
    if name in TIME_ATTRIBUTES and isinstance(v, (int, float)) and not isinstance(v, bool):
        return _iso(base, v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def to_ocel_json(sim: SupplyChainSimulation, params: ModelParameters) -> str:
    """Gera o OCEL 2.0 (JSON) da simulação no mesmo esquema do arquivo de entrada:
    eventTypes, objectTypes, events (com relationships E2O) e objects (com
    atributos datados e relationships O2O). Pode ser lido pelo PM4Py/OCPM."""
    # no curto prazo o relógio começa no fim do log
    base = params.start + pd.Timedelta(days=getattr(sim, "t0", 0.0))
    ev_types, ob_types = {}, {}

    # --- eventos
    events = []
    for eid, act, t, objs, attrs in sim.events:
        ev_types.setdefault(act, set()).update(attrs.keys())
        events.append({
            "id": eid, "type": act, "time": _iso(base, t),
            "attributes": [{"name": k, "value": _val(k, v, base)} for k, v in attrs.items()],
            "relationships": [{"objectId": o, "qualifier": q} for o, q in objs.items()],
        })

    # --- objetos (valor inicial + mudanças ao longo do tempo + relações O2O)
    objects = []
    for oid, o in sim.objects.items():
        ob_types.setdefault(o["type"], set()).update(o["attrs"].keys())
        t0 = _iso(base, o.get("time", 0.0))
        attrs = [{"name": k, "time": t0, "value": _val(k, v, base)} for k, v in o["attrs"].items()]
        for name, t, v in o["changes"]:
            ob_types[o["type"]].add(name)
            attrs.append({"name": name, "time": _iso(base, t), "value": _val(name, v, base)})
        obj = {"id": oid, "type": o["type"], "attributes": attrs}
        if o["o2o"]:
            obj["relationships"] = [{"objectId": dst, "qualifier": q} for dst, q in o["o2o"]]
        objects.append(obj)

    def _typ(v):
        """Tipo OCEL de um valor Python."""
        if isinstance(v, bool):
            return "boolean"
        if isinstance(v, int):
            return "integer"
        if isinstance(v, float):
            return "float"
        return "string"

    # --- cabeçalho com os tipos de eventos e de objetos
    doc = {
        "eventTypes": [{"name": k, "attributes": [{"name": a, "type": "string" if a == "cause_code" else "float"}
                                                     for a in sorted(v)]} for k, v in ev_types.items()],
        "objectTypes": [{"name": k, "attributes": [{"name": a, "type": "time" if a in TIME_ATTRIBUTES else "string"}
                                                      for a in sorted(v)]} for k, v in ob_types.items()],
        "events": events,
        "objects": objects,
    }
    # ajusta o tipo declarado dos atributos a partir de um valor de exemplo
    samples = {}
    for o in objects:
        for a in o["attributes"]:
            samples.setdefault((o["type"], a["name"]), a["value"])
    for ot in doc["objectTypes"]:
        for a in ot["attributes"]:
            if a["type"] != "time":
                a["type"] = _typ(samples.get((ot["name"], a["name"]), ""))
    for et in doc["eventTypes"]:
        for a in et["attributes"]:
            if a["name"] in ("critical_understock", "data_complete", "passive_observation"):
                a["type"] = "boolean"
            elif a["name"] in ("cause_code", "policy_version"):
                a["type"] = "string"
    return json.dumps(doc)
