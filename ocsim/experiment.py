"""
experiment.py — orquestração dos experimentos
=============================================

PAPEL NO PROJETO
----------------
Liga as peças: para cada cenário e cada replicação roda a simulação
(``simulation.py``), calcula as métricas (``metrics.py``) e junta os
resultados. Também:

* calcula as MESMAS métricas direto no log original (linha de base / referência);
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
import multiprocessing as mp
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd

from . import metrics as M
from .parameters import (ModelParameters, TIME_ATTRIBUTES, requested_quantity_series,
                         ordered_quantity_series, net_stock_series)
from .policies import ScenarioConfig
from .simulation import SupplyChainSimulation

# EXECUÇÃO PARALELA — ideia geral
# Para ganhar tempo, as replicações podem rodar ao mesmo tempo em vários
# núcleos do processador. Cada núcleo roda uma cópia separada do Python (um
# "processo auxiliar"), que não enxerga as variáveis do programa principal.
# Por isso os parâmetros do modelo (ModelParameters) precisam ser ENVIADOS a
# cada processo auxiliar. Para não reenviá-los a cada replicação, cada
# processo os recebe UMA vez, ao iniciar, e os guarda na variável abaixo.
# ("global" = variável do arquivo inteiro, visível por todas as funções dele.)
_PARAMS: ModelParameters | None = None


def _init_worker(params):
    """Executada uma única vez em cada processo auxiliar, quando ele começa:
    guarda os parâmetros recebidos na variável _PARAMS daquele processo."""
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
def parallel_supported() -> bool:
    """Indica se a execução paralela deve vir LIGADA por padrão neste computador.

    O modo de criar os processos auxiliares depende do sistema operacional
    (ver "Processos paralelos" no glossário de __init__.py):
      • Linux ("fork"): cada processo auxiliar nasce como clone do programa já
        em execução — rápido e confiável;
      • Windows e macOS ("spawn"): cada processo auxiliar é um Python novo que
        carrega tudo do zero. Dentro do Streamlit isso já causou travamentos.
    Por isso a interface deixa a opção ligada só no Linux. Nos outros sistemas
    o usuário pode ligá-la; se travar, o programa volta sozinho ao modo de um
    núcleo (ver run_experiment)."""
    # No Streamlit Community Cloud (servidor Linux na nuvem, com pouca memória e
    # poucos núcleos) cada processo auxiliar copiaria os dados do modelo; por
    # segurança, a opção também vem desligada lá. O Streamlit Cloud executa os
    # apps a partir da pasta /mount/src e com o usuário "adminuser".
    in_streamlit_cloud = os.path.isdir("/mount/src") or os.environ.get("HOME", "") == "/home/adminuser"
    return sys.platform.startswith("linux") and not in_streamlit_cloud


def _kill_pool(ex: ProcessPoolExecutor):
    """Encerra à força todos os processos auxiliares de um grupo ("pool")
    que travou, para que não fiquem rodando esquecidos no computador."""
    for p in list(getattr(ex, "_processes", {}).values()):
        try:
            p.terminate()
        except Exception:
            pass
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


class _HideMainModule:
    """Impede que os processos auxiliares executem o app.py de novo
    (esta é a correção do travamento no Windows/macOS).

    O problema: no modo "spawn" (Windows/macOS), cada processo auxiliar é um
    Python novo. Antes de trabalhar, ele carrega de novo o "arquivo principal"
    do programa — o que o Python chama de módulo ``__main__``. Quando o
    programa roda dentro do Streamlit, o arquivo principal é o próprio app.py.
    Resultado: cada processo auxiliar tentava abrir a interface inteira de novo
    (ler o OCEL, desenhar a tela...) em vez de simular, e a barra de progresso
    ficava parada em "Iniciando…".

    A solução: durante os poucos instantes em que os processos auxiliares são
    criados, trocamos o "arquivo principal" por um módulo vazio. Assim eles
    carregam apenas o que precisam (este arquivo, com a função
    run_replication) e o app.py fica intocado. Logo depois, o original é
    devolvido.

    Uso:  with _HideMainModule():
              ...cria os processos...       ← aqui o app.py fica "escondido"
    """

    def __enter__(self):   # executado ao entrar no bloco "with": esconde o app
        import types
        self._orig = sys.modules.get("__main__")
        sys.modules["__main__"] = types.ModuleType("__main__")
        return self

    def __exit__(self, *exc):   # executado ao sair do bloco "with": devolve o original
        if self._orig is not None:
            sys.modules["__main__"] = self._orig
        return False


def run_experiment(params: ModelParameters, scenarios: list[ScenarioConfig], n_reps: int, base_seed: int = 42,
                   opts: dict | None = None, progress=None, parallel: bool = True, max_workers: int | None = None,
                   stall_timeout: float = 90.0):
    """Executa n_reps replicações de cada cenário.

    Se ``parallel`` for True, distribui as replicações entre os núcleos do
    processador (até 8). Se o paralelismo falhar OU TRAVAR (nenhuma replicação
    concluída em ``stall_timeout`` segundos — situação possível no Windows/macOS
    dentro do Streamlit), os processos são encerrados e tudo roda em sequência,
    automaticamente.
    ``progress(fração, mensagem)`` atualiza a barra de progresso da interface."""
    opts = opts or {}
    # Lista de tarefas, uma por (cenário, replicação). A semente da replicação r
    # é base_seed + r em TODOS os cenários (números aleatórios comuns).
    # Ex.: 5 cenários × 10 replicações = 50 tarefas.
    tasks = [(sc.to_dict(), base_seed + r, r) for sc in scenarios for r in range(n_reps)]
    out = []
    total = len(tasks)
    if parallel and total > 1:
        # nº de processos auxiliares = nº de núcleos do computador (no máximo 8)
        workers = max_workers or max(1, min(os.cpu_count() or 1, total, 8))
        if progress:
            progress(0.0, f"Iniciando {workers} processos paralelos…")
        ex = None
        try:
            # modo de criação dos processos: "fork" no Linux, "spawn" nos demais
            ctx = mp.get_context("fork") if sys.platform.startswith("linux") else mp.get_context("spawn")
            # ProcessPoolExecutor = um "pool" (grupo) de processos auxiliares que
            # pegam tarefas de uma fila; cada um começa executando _init_worker(params)
            ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_init_worker, initargs=(params,))
            # submit = coloca uma tarefa na fila do pool. Os processos auxiliares
            # são criados nesse momento, por isso o app fica "escondido" aqui.
            # Cada submit devolve um "recibo" (future) que depois dá o resultado.
            with _HideMainModule():
                pending = {ex.submit(run_replication, c, s, r, opts) for c, s, r in tasks}
            while pending:
                # espera até QUALQUER replicação terminar (no máximo stall_timeout
                # segundos). done = recibos prontos; pending = os que faltam.
                # Se nenhuma terminar nesse prazo, consideramos que travou.
                done, pending = wait(pending, timeout=stall_timeout, return_when=FIRST_COMPLETED)
                if not done:
                    raise TimeoutError(f"nenhuma replicação concluída em {stall_timeout:.0f} s")
                for f in done:
                    out.append(f.result())
                if progress:
                    progress(len(out) / total, f"{len(out)}/{total} replicações concluídas")
            ex.shutdown(wait=True)
            return _sort(out, scenarios)
        except Exception as e:
            # qualquer problema (erro ou travamento): encerra os processos auxiliares
            # e segue para o modo de um núcleo, logo abaixo
            if ex is not None:
                _kill_pool(ex)
            if progress:
                progress(0.0, f"Execução paralela indisponível ({type(e).__name__}: {e}); rodando em sequência…")
            out = []
    # execução sequencial (um núcleo)
    for i, (c, s, r) in enumerate(tasks):
        if progress and i == 0:
            progress(0.0, f"0/{total} replicações concluídas")
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
    """Média e intervalo de confiança de 95% entre as replicações.

    Fórmula: média ± t · s / √n, onde n = nº de replicações, s = desvio-padrão
    entre elas e t = valor da distribuição t de Student (para n = 10, t ≈ 2,26).
    Interpretação: com 95% de confiança, a média "verdadeira" do cenário está
    dentro do intervalo. Mais replicações → intervalo mais estreito.
    Retorna (média, limite inferior, limite superior)."""
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
# Linha de base: métricas calculadas sobre o LOG ORIGINAL
# ===========================================================================
def log_baseline(d, params: ModelParameters, warmup_days: int = 56, bucket_days: int = 7) -> dict:
    """Aplica ao OCEL original exatamente as mesmas medidas da simulação,
    para que original e simulado sejam comparáveis."""
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
    """Gera o OCEL 2.0 (arquivo de texto no formato JSON) da simulação, com a mesma estrutura do arquivo de entrada:
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
