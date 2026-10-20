"""
parameters.py — Etapa 3: extração dos parâmetros de simulação a partir do OCEL
==============================================================================

PAPEL NO PROJETO
----------------
Este é o módulo que cumpre o requisito "os valores devem ser aleatórios de
acordo com a distribuição encontrada no arquivo". Ele percorre o OCEL e
produz um objeto ``ModelParameters`` com TUDO o que o simulador precisa:

* durações entre atividades por tipo de objeto (via achatamento PM4Py);
* lead time por fornecedor e proporção recebida/pedida;
* tempo de vida dos objetos e nº de ocorrências de cada atividade por objeto;
* distribuição de cada atributo de cada tipo de objeto;
* demanda de cada ItemLocation (probabilidade diária de pedido, quantidade,
  hora do dia, tendência, sazonalidade, dia da semana e dia do mês);
* modelo de objetos (fornecedor de cada IL, planejador por atividade,
  existência de Delivery por PO, cardinalidades);
* probabilidades de roteamento (rejeição, atraso, aprovação) e taxas dos
  eventos de exceção (ajustes, contagens, transferências, lacunas de dados);
* calibração da política atual (mín–máx) e o estado no fim do log.

O simulador (``simulation.py``) só lê este objeto — não consulta o log.

Termos de programação (DataFrame, groupby, merge, unstack, reindex,
dataclass...) estão explicados no glossário do arquivo ocsim/__init__.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import process_mining as pm
from .distributions import Dist, bernoulli, fit_categorical, fit_discrete, fit_numeric

# nomes curtos das colunas do OCEL (definidos em process_mining.py)
ACT, TS, OID, OTYPE, EID = pm.ACT, pm.TS, pm.OID, pm.OTYPE, pm.EID

# ---------------------------------------------------------------------------
# Atributos cujo valor NÃO é sorteado: eles resultam do que acontece na
# simulação. Ex.: o estoque em mão (on_hand) de um IL é a soma das entradas
# menos as saídas simuladas; o status de um pedido de compra muda para
# "received" quando ele chega. Os demais atributos (ex.: carrier, priority)
# não dependem do estado e são sorteados das distribuições do log.
# ---------------------------------------------------------------------------
STATE_ATTRIBUTES = {
    "ItemLocation": {"backorder", "inventory_position", "on_hand", "on_order", "reserved",
                     "lower_threshold", "upper_threshold", "policy_version", "data_complete",
                     "demand_estimate", "lead_time_estimate"},
    "SalesOrderItem": {"requested_quantity"},
    "PurchaseOrderItem": {"ordered_quantity", "confirmed_quantity", "planned_receipt_time",
                          "actual_receipt_time", "status"},
    "ReplenishmentProposal": {"suggested_quantity", "creation_time", "status"},
    "TransferOrder": {"quantity", "source_item_location", "target_item_location", "status"},
    "Delivery": {"shipment_status"},
}

# atributos do tipo data/hora: são modelados como "deslocamento em dias a partir
# do primeiro evento do objeto" (ex.: due_time = criação + ~1 dia)
TIME_ATTRIBUTES = {"due_time", "planned_receipt_time", "actual_receipt_time", "creation_time"}


# ===========================================================================
# Parâmetros de UM ItemLocation (combinação material × local)
# ===========================================================================
@dataclass  # "ficha" com campos (ver glossário)
class ILParams:
    il: str                    # id do ItemLocation (ex.: IL-0043)
    material: str              # material ligado (O2O)
    location: str              # local ligado (O2O)
    material_class: str        # smooth, seasonal, trending, intermittent, promotion_sensitive
    location_class: str        # compact, standard, large
    initial_on_hand: float     # estoque no evento "Initialize Inventory"
    s: float                   # lower_threshold do log (ponto de pedido da política atual)
    S: float                   # upper_threshold do log (nível máximo da política atual)
    lot_size: int              # lote mínimo do material
    unit_cost: float           # custo unitário (para custo de estocagem e EOQ)
    service_level: float       # nível de serviço alvo da Location
    supplier_probs: dict       # P(fornecedor | IL), ex.: {"SUP-001": 0.8, "SUP-003": 0.2}
    p_order: float             # probabilidade de haver pedido de venda num dia
    qty: Dist                  # distribuição da quantidade pedida por pedido
    daily_mean: float          # demanda diária média (incluindo dias sem pedido)
    daily_std: float           # desvio da demanda diária
    weekly_mean: float
    weekly_std: float
    trend_per_day: float       # tendência relativa (fração da média por dia)
    season_index: np.ndarray   # índice multiplicativo por mês (12 valores, média 1)
    lead_time_mean: float      # lead time TOTAL de reposição (proposta → recebimento), em dias
    lead_time_std: float
    n_orders_log: int          # nº de pedidos de venda no log


# ===========================================================================
# Conjunto completo de parâmetros do modelo
# ===========================================================================
@dataclass  # "ficha" com todos os parâmetros que a simulação precisa
class ModelParameters:
    start: pd.Timestamp                        # início do horizonte do log
    horizon_days: float                        # duração do log (dias)
    item_locations: dict[str, ILParams]        # parâmetros de cada IL
    static_objects: dict[str, pd.DataFrame]    # Location, Material, Supplier, Planner (copiados do log)
    delays: dict[str, Dist]                    # durações (dias) entre atividades
    delay_meta: dict[str, dict]                # para cada duração: tipo de objeto, atividade de/para
    probs: dict[str, float]                    # probabilidades de roteamento
    supplier_lead_time: dict[str, Dist]        # lead time por fornecedor ("__all__" = todos)
    attr_dists: dict[str, dict[str, Dist]]     # distribuição de cada atributo de cada tipo
    lifetimes: dict[str, Dist]                 # tempo de vida por tipo de objeto
    activity_counts: dict[str, dict[str, Dist]]  # nº de ocorrências de cada atividade por objeto
    planner_by_activity: dict[str, Dist]       # qual planejador executa cada atividade
    exception_rates: dict[str, float]          # eventos raros por IL·dia
    exception_dists: dict[str, Dist]           # quantidades dos eventos raros
    time_of_day: dict[str, Dist]               # hora do dia (fração de 0 a 1) de pedidos e revisões
    asis_review_prob: float                    # calibração da política atual
    cardinalities: pd.DataFrame                # tabela de cardinalidades O2O
    current_state: dict | None = None          # estado no fim do log (simulação de curto prazo)
    receipt_ratio: Dist | None = None          # quantidade recebida / pedida
    # os dois campos abaixo começam, por padrão, com índice 1 em todos os dias (= sem efeito)
    calendar_dow: np.ndarray = field(default_factory=lambda: np.ones(7))   # índice por dia da semana
    calendar_dom: np.ndarray = field(default_factory=lambda: np.ones(31))  # índice por dia do mês
    notes: list[str] = field(default_factory=list)  # observações sobre os dados

    def delay_table(self) -> pd.DataFrame:
        """Tabela com todas as durações ajustadas (mostrada na etapa 3)."""
        rows = []
        for k, dist in self.delays.items():
            meta = self.delay_meta.get(k, {})
            r = {"parâmetro": k, **meta, **dist.to_row()}
            rows.append(r)
        return pd.DataFrame(rows)


# ===========================================================================
# Funções auxiliares
# ===========================================================================
def _days(td: pd.Series) -> pd.Series:
    """Converte diferenças de tempo (Timedelta) em dias decimais."""
    return td.dt.total_seconds() / 86400


def _frac_day(ts: pd.Series) -> np.ndarray:
    """Hora do dia como fração (08:00 → 0.333)."""
    t = pd.to_datetime(ts, utc=True)
    return ((t - t.dt.normalize()).dt.total_seconds() / 86400).to_numpy()


def _pair(dur: pd.DataFrame, a: str, b: str) -> np.ndarray:
    """Seleciona as durações observadas do arco a → b (descarta negativas,
    que vêm de inconsistências do log)."""
    x = dur[(dur["act"] == a) & (dur["next_act"] == b)]["dias"].to_numpy()
    return x[x >= 0]


# ---------------------------------------------------------------------------
# Séries diárias (IL × dia) — usadas para a demanda e para medir o efeito
# chicote do LOG ORIGINAL com a mesma métrica aplicada à simulação.
# ---------------------------------------------------------------------------
def daily_series(d: pm.OcelData, activity: str, qty_col: str = "quantity") -> pd.DataFrame:
    """Soma diária de uma coluna de quantidade para uma atividade.

    Resultado: uma matriz com uma linha por IL e uma coluna por dia
    (dia 0 = primeiro dia do log). As funções seguintes seguem o mesmo roteiro:
      1) seleciona os eventos de interesse;
      2) calcula o nº do dia de cada evento (segundos desde o início ÷ 86 400);
      3) groupby: soma as quantidades por (IL, dia);
      4) unstack: gira os dias para colunas (matriz IL × dia);
      5) reindex: garante todas as ILs e todos os dias, com 0 onde não houve evento."""
    e = d.events[d.events[ACT] == activity]
    day = ((e[TS] - d.start).dt.total_seconds() // 86400).astype(int)
    s = e.assign(day=day).groupby(["il", "day"])[qty_col].sum().unstack(fill_value=0.0)
    days = np.arange(int(math.ceil(d.horizon_days)))
    ils = sorted(d.events["il"].dropna().unique())
    return s.reindex(index=ils, columns=days, fill_value=0.0)


def requested_quantity_series(d: pm.OcelData) -> pd.DataFrame:
    """DEMANDA DO CLIENTE FINAL: requested_quantity dos SalesOrderItem,
    somada por IL e por dia de criação. Linhas = ILs, colunas = dias.

    A quantidade pedida é um atributo do OBJETO SalesOrderItem e o IL vem do
    EVENTO de criação; por isso juntamos (merge) as duas informações."""
    rel = d.relations[(d.relations[OTYPE] == "SalesOrderItem") & (d.relations[ACT] == "Sales Order Item Created")]
    so = rel[[EID, OID, TS]].merge(d.events[[EID, "il"]], on=EID)
    attrs = d.objects.set_index(OID)["requested_quantity"]
    so["q"] = so[OID].map(attrs).astype(float)
    so["day"] = ((so[TS] - d.start).dt.total_seconds() // 86400).astype(int)
    s = so.groupby(["il", "day"])["q"].sum().unstack(fill_value=0.0)
    days = np.arange(int(math.ceil(d.horizon_days)))
    ils = sorted(d.events["il"].dropna().unique())
    return s.reindex(index=ils, columns=days, fill_value=0.0)


def ordered_quantity_series(d: pm.OcelData) -> pd.DataFrame:
    """PEDIDOS AO FORNECEDOR: ordered_quantity dos PurchaseOrderItem por IL e dia."""
    rel = d.relations[(d.relations[OTYPE] == "PurchaseOrderItem") & (d.relations[ACT] == "Purchase Order Item Created")]
    po = rel[[EID, OID, TS]].merge(d.events[[EID, "il"]], on=EID)
    attrs = d.objects.set_index(OID)["ordered_quantity"]
    po["q"] = po[OID].map(attrs).astype(float)
    po["day"] = ((po[TS] - d.start).dt.total_seconds() // 86400).astype(int)
    s = po.groupby(["il", "day"])["q"].sum().unstack(fill_value=0.0)
    days = np.arange(int(math.ceil(d.horizon_days)))
    ils = sorted(d.events["il"].dropna().unique())
    return s.reindex(index=ils, columns=days, fill_value=0.0)


def net_stock_series(d: pm.OcelData) -> pd.DataFrame:
    """ESTOQUE LÍQUIDO no fim de cada dia = em mão − backorder (último evento do dia)."""
    e = d.events.dropna(subset=["il"]).copy()
    e["day"] = ((e[TS] - d.start).dt.total_seconds() // 86400).astype(int)
    e["ns"] = e["on_hand_after"].astype(float) - e["backorder_after"].astype(float)
    s = e.groupby(["il", "day"])["ns"].last().unstack()
    days = np.arange(int(math.ceil(d.horizon_days)))
    # dias sem nenhum evento não têm valor: ffill repete o último valor conhecido
    # (o estoque não muda se nada acontece); bfill preenche os primeiros dias
    return s.reindex(columns=days).ffill(axis=1).bfill(axis=1)


def _trend_season(series: np.ndarray, start: pd.Timestamp) -> tuple[float, np.ndarray]:
    """Estima TENDÊNCIA e SAZONALIDADE de uma série diária de demanda.

    1) agrega a série por mês (média diária de cada mês);
    2) ajusta uma reta y = a + b·t (regressão linear: a = intercepto, b =
       inclinação, ou seja, quanto a demanda cresce por dia). A tendência só é
       mantida se for estatisticamente significativa: a inclinação precisa ser
       mais de 2 vezes maior que o seu erro-padrão (equivale a um teste t com
       ~95% de confiança);
    3) divide a série pela reta e calcula a razão média de cada mês do ano
       → índice sazonal (12 valores com média 1; ex.: 1,10 em dezembro =
       demanda 10% acima do normal);
    4) a sazonalidade só é mantida se a diferença entre o maior e o menor
       índice for > 10%; abaixo disso é considerada ruído.
    Retorna (tendência relativa por dia, índice sazonal).
    Ex.: tendência 0,001 = demanda cresce 0,1% da média por dia (≈ 37% ao ano)."""
    n = len(series)
    if series.sum() <= 0 or n < 60:
        return 0.0, np.ones(12)
    dates = pd.date_range(start, periods=n, freq="D")
    s = pd.Series(series, index=dates)
    monthly = s.resample("MS").mean()
    t = (monthly.index - monthly.index[0]).days.to_numpy().astype(float)
    y = monthly.to_numpy()
    mean = y.mean()
    if mean <= 0:
        return 0.0, np.ones(12)
    b, a = np.polyfit(t, y, 1)  # ajusta a reta: devolve inclinação b e intercepto a
    resid = y - (a + b * t)     # resíduos = diferença entre os dados e a reta
    # erro-padrão da inclinação (fórmula clássica da regressão linear simples)
    se = math.sqrt(max(np.sum(resid ** 2) / max(len(y) - 2, 1), 1e-12) / max(np.sum((t - t.mean()) ** 2), 1e-12))
    trend = b / mean if abs(b / se) > 2 else 0.0  # mantém só se |b / erro| > 2
    fitted = mean * (1 + trend * (t - t.mean()))
    ratio = pd.Series(np.where(fitted > 0, y / fitted, 1.0), index=monthly.index)
    idx = ratio.groupby(ratio.index.month).mean().reindex(range(1, 13)).fillna(1.0).to_numpy()
    idx = idx / idx.mean()
    if np.ptp(idx) < 0.10:
        idx = np.ones(12)
    return float(trend), idx


# ===========================================================================
# FUNÇÃO PRINCIPAL
# ===========================================================================
def extract_parameters(d: pm.OcelData, progress=None) -> ModelParameters:
    """Percorre o OCEL e devolve todos os parâmetros da simulação.
    ``progress`` (opcional) é uma função que recebe mensagens de andamento;
    a interface e o run_experiment.py a usam para mostrar em que passo estamos."""

    def step(msg):  # envia a mensagem de andamento, se alguém pediu para recebê-la
        if progress:
            progress(msg)

    notes: list[str] = []
    H = d.horizon_days
    ev = d.events

    # -----------------------------------------------------------------------
    # 1) ACHATAMENTO por tipo de objeto e amostras de duração (PM4Py)
    # -----------------------------------------------------------------------
    step("Achatando o OCEL por tipo de objeto (PM4Py)…")
    flats = {ot: pm.flatten(d, ot) for ot in pm.OT_ORDER if ot in set(d.objects[OTYPE])}
    durs = {ot: pm.dfg_durations(fl) for ot, fl in flats.items()}

    delays: dict[str, Dist] = {}
    meta: dict[str, dict] = {}

    def add_delay(key, otype, a, b, x=None, seed=0):
        """Ajusta a distribuição do tempo entre as atividades a → b no log
        achatado de ``otype`` e guarda com o nome ``key``.
        Ex.: add_delay("rp_created_to_approved", "ReplenishmentProposal",
        "Replenishment Proposal Created", "Replenishment Proposal Approved")
        guarda em delays["rp_created_to_approved"] a distribuição do tempo de
        aprovação das propostas. O simulador depois sorteia desse nome."""
        if x is None:
            x = _pair(durs[otype], a, b)
        delays[key] = fit_numeric(x, seed=seed)
        meta[key] = {"tipo de objeto": otype, "de": a, "para": b}

    # -----------------------------------------------------------------------
    # 2) DURAÇÕES entre atividades consecutivas (uma por arco do processo)
    # -----------------------------------------------------------------------
    step("Ajustando distribuições das durações entre atividades…")
    # ciclo de vida do pedido de venda (SalesOrderItem)
    S = "SalesOrderItem"
    add_delay("so_created_to_reservation", S, "Sales Order Item Created", "Reservation Created")
    add_delay("so_reservation_to_issue", S, "Reservation Created", "Goods Issue")
    add_delay("so_created_to_backorder", S, "Sales Order Item Created", "Backorder Registered")
    add_delay("so_issue_to_backorder", S, "Goods Issue", "Backorder Registered")
    # aprovação da proposta de reposição (ReplenishmentProposal)
    R = "ReplenishmentProposal"
    add_delay("rp_created_to_approved", R, "Replenishment Proposal Created", "Replenishment Proposal Approved")
    # proposta aprovada → pedido de compra: as duas atividades não compartilham
    # objeto transacional, então usamos o log achatado por ItemLocation
    IL = "ItemLocation"
    add_delay("rp_approved_to_po", IL, "Replenishment Proposal Approved", "Purchase Order Item Created")
    # ciclo de vida do pedido de compra (PurchaseOrderItem)
    P = "PurchaseOrderItem"
    add_delay("po_created_to_confirmation", P, "Purchase Order Item Created", "Supplier Confirmation Received")
    add_delay("po_confirmation_to_receipt", P, "Supplier Confirmation Received", "Goods Receipt")
    add_delay("po_confirmation_to_delayed", P, "Supplier Confirmation Received", "Delivery Delayed")
    add_delay("po_delayed_to_expedite", P, "Delivery Delayed", "Expedite Requested")
    add_delay("po_expedite_to_receipt", P, "Expedite Requested", "Goods Receipt")
    add_delay("po_confirmation_to_rejected", P, "Supplier Confirmation Received", "Receipt Rejected")
    # transferências entre ILs (TransferOrder) e duração das lacunas de dados
    T = "TransferOrder"
    add_delay("to_requested_to_ship", T, "Transfer Requested", "Transfer Ship")
    add_delay("to_ship_to_receive", T, "Transfer Ship", "Transfer Receive")
    add_delay("data_gap_duration", IL, "Data Gap Started", "Data Gap Ended")

    # -----------------------------------------------------------------------
    # 3) LEAD TIME por fornecedor: 'PO Created' → 'Goods Receipt'
    # -----------------------------------------------------------------------
    step("Calculando lead time por fornecedor…")
    rel = d.relations
    fl = flats[P]
    # tabela com uma linha por pedido de compra e uma coluna por atividade,
    # contendo a data/hora da 1ª vez que a atividade ocorreu naquele pedido
    first = fl.groupby(["case", "act"])["ts"].min().unstack()
    lt = _days(first.get("Goods Receipt") - first.get("Purchase Order Item Created")).dropna()
    lt = lt[lt >= 0]
    o2o = d.o2o
    # relações O2O: fornecedor de cada PO e IL reposto por cada PO
    po_sup = o2o[(o2o[OTYPE] == P) & (o2o[OTYPE + "_2"] == "Supplier")].set_index(OID)[OID + "_2"]
    po_il = o2o[(o2o[OTYPE] == P) & (o2o[OTYPE + "_2"] == IL)].set_index(OID)[OID + "_2"]
    lt_df = lt.rename("lt").to_frame()
    lt_df["sup"] = lt_df.index.map(po_sup)
    supplier_lt: dict[str, Dist] = {"__all__": fit_numeric(lt_df["lt"])}
    for sup, g in lt_df.groupby("sup"):
        if len(g) >= 15:   # ajusta por fornecedor só se houver dados suficientes
            supplier_lt[sup] = fit_numeric(g["lt"])
        else:
            notes.append(f"Fornecedor {sup}: apenas {len(g)} recebimentos — usa a distribuição global de lead time.")

    # -----------------------------------------------------------------------
    # 4) RECEBIMENTO PARCIAL: quantidade recebida / quantidade pedida
    #    (no log, em média chegam ~84% do pedido)
    # -----------------------------------------------------------------------
    gr = rel[(rel[OTYPE] == P) & (rel[ACT] == "Goods Receipt")][[EID, OID]].drop_duplicates(OID)
    gr = gr.merge(ev[[EID, "quantity"]], on=EID)
    gr["ordered"] = gr[OID].map(d.objects.set_index(OID)["ordered_quantity"]).astype(float)
    ratio = (gr["quantity"].astype(float) / gr["ordered"]).replace([np.inf, -np.inf], np.nan).dropna()
    receipt_ratio = fit_numeric(ratio.clip(0, 2).to_numpy())
    receipt_ratio.mode = "empirical"

    # -----------------------------------------------------------------------
    # 5) PROBABILIDADES DE ROTEAMENTO = % de objetos que passam por cada atividade
    # -----------------------------------------------------------------------
    step("Estimando probabilidades de roteamento…")
    cnt = pm.activity_counts_per_object(d, P)  # matriz PO × atividade
    n_po = len(cnt)
    probs = {
        "po_confirmation": float((cnt.get("Supplier Confirmation Received", 0) > 0).mean()),
        "po_delayed": float((cnt.get("Delivery Delayed", 0) > 0).mean()),
        "po_rejected": float((cnt.get("Receipt Rejected", 0) > 0).mean()),
    }
    rpc = pm.activity_counts_per_object(d, R)
    probs["rp_approved"] = float((rpc.get("Replenishment Proposal Approved", 0) > 0).mean())
    n_del = (d.objects[OTYPE] == "Delivery").sum()
    probs["po_has_delivery"] = float(min(1.0, n_del / max(n_po, 1)))  # cardinalidade 0..1 PO → Delivery
    soc = pm.activity_counts_per_object(d, S)
    probs["so_partial_backorder"] = float(
        ((soc.get("Goods Issue", 0) > 0) & (soc.get("Backorder Registered", 0) > 0)).mean()
    )

    # -----------------------------------------------------------------------
    # 6) CALIBRAÇÃO da política atual (ver _asis_review_probability)
    # -----------------------------------------------------------------------
    step("Calibrando a política atual (mín–máx) observada no log…")
    asis_p = _asis_review_probability(d)

    # -----------------------------------------------------------------------
    # 7) ATRIBUTOS de todos os tipos de objeto
    #    - data/hora → deslocamento (dias) a partir do 1º evento do objeto
    #    - booleano/texto → categórica
    #    - número → melhor distribuição (inteiro se todos os valores forem inteiros)
    # -----------------------------------------------------------------------
    step("Ajustando distribuições dos atributos dos objetos…")
    attr_dists: dict[str, dict[str, Dist]] = {}
    first_ts = d.relations.groupby(OID)[TS].min()
    for ot, g in d.objects.groupby(OTYPE):
        attr_dists[ot] = {}
        for col in g.columns:
            if col in (OID, OTYPE) or g[col].isna().all():
                continue
            vals = g[col].dropna()
            if col in TIME_ATTRIBUTES:
                t = pd.to_datetime(vals, utc=True, errors="coerce")
                base = g.loc[vals.index, OID].map(first_ts)
                off = _days(t - pd.to_datetime(base, utc=True)).dropna()
                attr_dists[ot][col] = fit_numeric(off.to_numpy(), nonneg=False)
                attr_dists[ot][col].mode = "empirical"
            elif pd.api.types.is_bool_dtype(vals) or vals.map(type).eq(bool).all():
                attr_dists[ot][col] = fit_categorical(vals.astype(str))
            elif pd.api.types.is_numeric_dtype(vals):
                integer = bool(np.all(np.mod(vals.astype(float), 1) == 0))
                attr_dists[ot][col] = fit_numeric(vals.astype(float).to_numpy(), integer=integer)
            else:
                attr_dists[ot][col] = fit_categorical(vals)

    # -----------------------------------------------------------------------
    # 8) TEMPO DE VIDA dos objetos e Nº DE OCORRÊNCIAS de cada atividade por objeto
    # -----------------------------------------------------------------------
    step("Tempos de vida e nº de ocorrências de atividades por objeto…")
    lifetimes = {ot: fit_numeric(pm.object_lifetimes(fl).to_numpy()) for ot, fl in flats.items()}
    activity_counts = {}
    for ot in flats:
        m = pm.activity_counts_per_object(d, ot)
        activity_counts[ot] = {a: fit_discrete(m[a].to_numpy()) for a in m.columns}

    # -----------------------------------------------------------------------
    # 9) PLANEJADOR que participa de cada atividade (relações E2O)
    # -----------------------------------------------------------------------
    planner_by_activity = {}
    for a, g in rel[rel[OTYPE] == "Planner"].groupby(ACT):
        planner_by_activity[a] = fit_categorical(g[OID])

    # -----------------------------------------------------------------------
    # 10) HORA DO DIA dos pedidos de venda (~08h) e das revisões (~17–18h)
    # -----------------------------------------------------------------------
    tod = {}
    for a in ["Sales Order Item Created", "Replenishment Proposal Created"]:
        x = _frac_day(ev.loc[ev[ACT] == a, TS])
        tod[a] = fit_numeric(x)
        tod[a].mode = "empirical"

    # -----------------------------------------------------------------------
    # 11) EVENTOS DE EXCEÇÃO (raros): taxa por IL·dia e quantidades
    #     Os 35 'FORCED_EPISODE_RESET' são artefatos do gerador e ficam de fora.
    # -----------------------------------------------------------------------
    step("Taxas dos eventos de exceção…")
    n_il = ev["il"].nunique()
    exc_acts = ["Inventory Adjustment", "Cycle Count Performed", "Transfer Requested",
                "Policy Threshold Updated", "Data Gap Started"]
    real = ev[ev["cause_code"].fillna("") != "FORCED_EPISODE_RESET"]
    exception_rates = {a: float((real[ACT] == a).sum() / (n_il * H)) for a in exc_acts}
    exc_d = {}
    ia = ev[(ev[ACT] == "Inventory Adjustment") & (ev["cause_code"] != "FORCED_EPISODE_RESET")]
    exc_d["adjustment_qty"] = fit_numeric(ia["quantity"].dropna().to_numpy(), integer=True, nonneg=False)
    exc_d["transfer_qty"] = fit_numeric(
        d.objects.loc[d.objects[OTYPE] == T, "quantity"].dropna().astype(float).to_numpy(), integer=True)
    exc_d["threshold_upper"] = fit_numeric(
        ev.loc[ev[ACT] == "Policy Threshold Updated", "upper_threshold"].dropna().to_numpy(), integer=True)
    cc = ev[ev[ACT] == "Cycle Count Performed"]
    exc_d["count_discrepancy"] = fit_numeric(cc["quantity"].dropna().to_numpy(), integer=True)
    notes.append("No log, o 'Goods Receipt' reduz o backorder sem consumir o estoque recebido (contabilidade "
                 "do gerador). O cenário Atual pode reproduzir essa contabilidade (opção 'log') para validação; "
                 "os demais cenários usam a contabilidade física correta (backorders atendidos com o estoque recebido).")
    notes.append("Eventos 'FORCED_EPISODE_RESET' (35 ajustes de inventário que reiniciam o estoque) foram tratados "
                 "como artefatos do gerador do log e não são reproduzidos.")

    # -----------------------------------------------------------------------
    # 12) ITEMLOCATIONS: demanda, fornecedores, lead time e dados cadastrais
    # -----------------------------------------------------------------------
    step("Ajustando a demanda de cada ItemLocation…")
    dem = requested_quantity_series(d)            # demanda diária (IL × dia)
    il_attr = pm.static_attributes(d, IL)
    mat = pm.static_attributes(d, "Material")
    loc = pm.static_attributes(d, "Location")
    il_mat = o2o[(o2o[OTYPE] == IL) & (o2o[OTYPE + "_2"] == "Material")].set_index(OID)[OID + "_2"]
    il_loc = o2o[(o2o[OTYPE] == IL) & (o2o[OTYPE + "_2"] == "Location")].set_index(OID)[OID + "_2"]
    init = ev[ev[ACT] == "Initialize Inventory"].groupby("il")["on_hand_after"].first()
    so_rel = rel[(rel[OTYPE] == S) & (rel[ACT] == "Sales Order Item Created")][[EID, OID]].merge(ev[[EID, "il"]], on=EID)
    so_q = so_rel.assign(q=so_rel[OID].map(d.objects.set_index(OID)["requested_quantity"]).astype(float))
    # tabela cruzada (como uma tabela dinâmica): nº de POs por IL (linha) e fornecedor (coluna)
    sup_counts = pd.crosstab(po_il, po_sup)
    rp_mean = delays["rp_created_to_approved"].mean
    ap_mean = delays["rp_approved_to_po"].mean

    # tendência e sazonalidade estimadas por CLASSE DE MATERIAL (séries somadas
    # das ILs da mesma classe — mais robusto do que estimar IL por IL).
    # As 2 primeiras semanas têm picos forçados pelo gerador e são suavizadas.
    class_profile = {}
    for mc, g in il_attr.groupby("material_class"):
        agg = dem.loc[dem.index.intersection(g.index)].sum(axis=0).to_numpy().astype(float)
        agg[:14] = agg[14:42].mean()
        class_profile[str(mc)] = _trend_season(agg, d.start)

    ils = {}
    for i, il in enumerate(dem.index):
        series = dem.loc[il].to_numpy()
        q = so_q.loc[so_q["il"] == il, "q"].to_numpy()   # quantidades dos pedidos deste IL
        p_order = len(q) / H                              # pedidos por dia (no máx. 1 por dia no log)
        mat_id = il_mat.get(il)
        loc_id = il_loc.get(il)
        # P(fornecedor | IL): proporção de POs do IL em cada fornecedor
        if il in sup_counts.index and sup_counts.loc[il].sum() > 0:
            sp = (sup_counts.loc[il] / sup_counts.loc[il].sum())
            sp = {k: float(v) for k, v in sp.items() if v > 0}
        else:
            tot = po_sup.value_counts(normalize=True)
            sp = {k: float(v) for k, v in tot.items()}
        # lead time esperado ponderado pelos fornecedores do IL
        lt_mean = sum(p * supplier_lt.get(s_, supplier_lt["__all__"]).mean for s_, p in sp.items())
        lt_var = sum(p * supplier_lt.get(s_, supplier_lt["__all__"]).std ** 2 for s_, p in sp.items())
        weekly = pd.Series(series).groupby(np.arange(len(series)) // 7).sum()
        trend, season = class_profile[str(il_attr.loc[il, "material_class"])]
        ils[il] = ILParams(
            il=il,
            material=mat_id,
            location=loc_id,
            material_class=str(il_attr.loc[il, "material_class"]),
            location_class=str(il_attr.loc[il, "location_class"]),
            initial_on_hand=float(init.get(il, il_attr.loc[il, "on_hand"])),
            s=float(il_attr.loc[il, "lower_threshold"]),
            S=float(il_attr.loc[il, "upper_threshold"]),
            lot_size=int(mat.loc[mat_id, "lot_size"]) if mat_id in mat.index else 1,
            unit_cost=float(mat.loc[mat_id, "unit_cost"]) if mat_id in mat.index else 1.0,
            service_level=float(loc.loc[loc_id, "service_level_target"]) if loc_id in loc.index else 0.95,
            supplier_probs=sp,
            p_order=float(min(p_order, 1.0)),
            # quantidade: só distribuições de suporte positivo (quantidade ≥ 1)
            qty=fit_numeric(q, integer=True, candidates=["gama", "lognormal", "weibull", "exponencial", "normal"],
                            seed=i),
            daily_mean=float(series.mean()),
            daily_std=float(series.std(ddof=1)),
            weekly_mean=float(weekly.mean()),
            weekly_std=float(weekly.std(ddof=1)),
            trend_per_day=trend,
            season_index=season,
            # lead time total = aprovação + emissão do PO + fornecedor
            lead_time_mean=float(lt_mean + rp_mean + ap_mean),
            lead_time_std=float(math.sqrt(lt_var)),
            n_orders_log=int(len(q)),
        )

    # -----------------------------------------------------------------------
    # 13) EFEITOS DE CALENDÁRIO na demanda total
    #     razão entre a demanda do dia e a média móvel de 28 dias, agrupada
    #     por dia da semana e por dia do mês (ex.: segundas +15%, sextas −19%)
    # -----------------------------------------------------------------------
    tot = pd.Series(dem.sum(axis=0).to_numpy(), index=pd.date_range(d.start, periods=dem.shape[1], freq="D"))
    # média móvel centrada de 28 dias = "nível normal" da demanda em torno de cada dia
    base = tot.rolling(28, center=True, min_periods=7).mean()
    ratio = (tot / base).iloc[14:]
    dow = ratio.groupby(ratio.index.dayofweek).mean().reindex(range(7)).fillna(1).to_numpy()
    dom = ratio.groupby(ratio.index.day).mean().reindex(range(1, 32)).fillna(1).to_numpy()
    dow, dom = dow / dow.mean(), dom / dom.mean()

    # -----------------------------------------------------------------------
    # 14) ESTADO NO FIM DO LOG (para a simulação de curto prazo)
    # -----------------------------------------------------------------------
    step("Reconstruindo o estado atual (fim do log) para simulação de curto prazo…")
    current = current_state(d, po_sup, po_il)

    # objetos estruturais são copiados do log (a rede não muda entre cenários)
    static = {ot: pm.static_attributes(d, ot) for ot in ["Location", "Material", "Supplier", "Planner"]}
    cards = pm.o2o_cardinalities(d)

    return ModelParameters(
        start=d.start,
        horizon_days=H,
        item_locations=ils,
        static_objects=static,
        delays=delays,
        delay_meta=meta,
        probs=probs,
        supplier_lead_time=supplier_lt,
        attr_dists=attr_dists,
        lifetimes=lifetimes,
        activity_counts=activity_counts,
        planner_by_activity=planner_by_activity,
        exception_rates=exception_rates,
        exception_dists=exc_d,
        time_of_day=tod,
        asis_review_prob=asis_p,
        cardinalities=cards,
        receipt_ratio=receipt_ratio,
        current_state=current,
        calendar_dow=dow,
        calendar_dom=dom,
        notes=notes,
    )


# ===========================================================================
# Calibração da política atual
# ===========================================================================
def _asis_review_probability(d: pm.OcelData) -> float:
    """P(gerar proposta de reposição | posição de estoque ≤ s na revisão diária).

    No log a política é mín–máx (s=20, S=80/90), mas nem todo dia em que a
    posição de estoque está abaixo de s gera uma proposta. Para reproduzir
    isso, para cada IL e dia olhamos a posição de estoque ANTES da revisão
    (eventos antes das 17h) e verificamos se houve 'Replenishment Proposal
    Created'. A fração de dias "abaixo de s" com proposta é a probabilidade
    (≈ 0,18 neste log)."""
    e = d.events.dropna(subset=["il"]).copy()
    e["day"] = e[TS].dt.normalize()          # data sem a hora
    before = e[e[TS].dt.hour < 17]           # eventos antes da revisão das 17h
    # último valor da posição de estoque e do limite s antes da revisão, por IL e dia
    ip = before.groupby(["il", "day"])[["inventory_position_after", "lower_threshold"]].last()
    ip = ip.groupby(level=0).ffill()         # dias sem evento herdam o último valor do mesmo IL
    rp = set(map(tuple, e.loc[e[ACT] == "Replenishment Proposal Created", ["il", "day"]].to_numpy()))
    low = ip[ip["inventory_position_after"] <= ip["lower_threshold"]]
    if len(low) == 0:
        return 1.0
    hit = np.array([(il, day) in rp for il, day in low.index])
    return float(hit.mean())


# ===========================================================================
# Estado atual (fim do log) — simulação de curto prazo (van der Aalst)
# ===========================================================================
def current_state(d: pm.OcelData, po_sup: pd.Series, po_il: pd.Series) -> dict:
    """Reconstrói o estado no fim do log:
    * para cada IL: último estoque em mão, backorder, em pedido e limites;
    * POs em aberto: criados e ainda sem 'Goods Receipt' nem 'Receipt Rejected'
      (com idade ≤ 150 dias; mais antigos são considerados perdidos)."""
    e = d.events.dropna(subset=["il"]).sort_values(TS)
    cols = ["on_hand_after", "backorder_after", "on_order_after", "upper_threshold", "lower_threshold"]
    # último valor registrado (não vazio) de cada coluna, para cada IL
    last = e.groupby("il")[cols].agg(lambda x: x.dropna().iloc[-1] if x.notna().any() else np.nan)
    rel = d.relations[d.relations[OTYPE] == "PurchaseOrderItem"]
    acts = rel.groupby(OID)[ACT].agg(set)              # atividades já ocorridas em cada PO
    created = rel[rel[ACT] == "Purchase Order Item Created"].groupby(OID)[TS].min()
    qty = d.objects.set_index(OID)["ordered_quantity"]
    open_pos = []
    for po, a in acts.items():
        if "Purchase Order Item Created" in a and not ({"Goods Receipt", "Receipt Rejected"} & a):
            age = (d.end - created[po]).total_seconds() / 86400
            if age <= 150:
                open_pos.append({"po": po, "il": po_il.get(po), "supplier": po_sup.get(po),
                                 "qty": float(qty.get(po, 0.0)), "age": float(age)})
    ils = {il: {k: float(r[k]) for k in cols if pd.notna(r[k])} for il, r in last.iterrows()}
    return {"date": d.end, "offset_days": d.horizon_days, "ils": ils, "open_pos": open_pos}
