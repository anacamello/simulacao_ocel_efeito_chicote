"""
impact.py — impacto de um tipo de objeto sobre outro (análise centrada em objetos)
==================================================================================

PAPEL NO PROJETO
----------------
Mostra, com números, POR QUE a análise precisa ser centrada em objetos: o
efeito chicote e a falta de estoque nascem da interação entre tipos de objeto
diferentes. Cada análise aqui só é possível porque o OCEL liga os objetos
(relações E2O e O2O); com um log de um único tipo de caso ela não poderia ser
calculada.

As funções recebem um ``OcelData`` (process_mining.py) e, por isso, funcionam
igualmente para o LOG ORIGINAL e para os LOGS SIMULADOS exportados pelo
simulador (mesmo formato OCEL 2.0).

ANÁLISES
--------
1. FORNECEDOR → CLIENTE (PurchaseOrderItem → SalesOrderItem, via ItemLocation)
   No instante em que cada pedido de venda é criado, verifica a situação dos
   pedidos de compra em aberto do MESMO ItemLocation (nenhum / no prazo /
   atrasado) e mede a % de pedidos de venda que viram backorder em cada
   situação, comparando sempre dentro do mesmo IL.

2. CADEIA DE REPOSIÇÃO → FALTA DE ESTOQUE (por ItemLocation)
   Para cada IL relaciona o lead time médio dos seus pedidos de compra, o tempo
   de aprovação das suas propostas e a demanda com a % de pedidos de venda em
   backorder (correlação de Spearman).

3. DEMANDA → PEDIDOS (SalesOrderItem → PurchaseOrderItem, por política)
   Correlação cruzada entre a demanda da semana t e os pedidos da semana t+k:
   mostra com quantas semanas de atraso e com que intensidade cada política
   "responde" à demanda — o mecanismo do efeito chicote.

4. SENSIBILIDADE (Supplier/demanda → efeito chicote) — feita no app com o
   simulador: varia o lead time do fornecedor e mede o BWE de cada política.

MEDIDAS ESTATÍSTICAS USADAS
---------------------------
* Correlação de Spearman (ρ, "rô"): mede se duas grandezas crescem juntas
  (ρ próximo de +1), uma cresce enquanto a outra cai (próximo de −1) ou não
  têm relação (próximo de 0). Usa a ORDEM dos valores (1º, 2º, 3º...), e não
  os valores em si, por isso funciona mesmo quando a relação não é uma reta e
  é pouco afetada por valores extremos.
* p-valor: probabilidade de observar um resultado tão forte quanto esse só por
  acaso, se na verdade não houvesse relação. Usualmente p < 0,05 é tratado
  como "estatisticamente significativo".
* Teste de Mann–Whitney: compara dois grupos (ex.: pedidos de venda com
  fornecedor atrasado × no prazo) e verifica se os valores de um tendem a ser
  maiores que os do outro, sem supor distribuição normal.
* Correlação de Pearson (na análise 3): mede o quanto duas séries variam
  juntas em linha reta, de −1 a +1.

Termos de programação (DataFrame, groupby, merge...) estão no glossário de
ocsim/__init__.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import process_mining as pm

ACT, TS, OID, OTYPE, EID = pm.ACT, pm.TS, pm.OID, pm.OTYPE, pm.EID


# ===========================================================================
# Tabelas auxiliares: um registro por objeto, com o IL associado
# ===========================================================================
def _object_events(d: pm.OcelData, otype: str) -> pd.DataFrame:
    """Eventos (atividade, timestamp, IL) de todos os objetos de um tipo."""
    r = d.relations[d.relations[OTYPE] == otype][[EID, OID, ACT, TS]]
    return r.merge(d.events[[EID, "il"]], on=EID, how="left")


def sales_order_table(d: pm.OcelData) -> pd.DataFrame:
    """Um registro por SalesOrderItem: IL, data de criação e se teve backorder."""
    e = _object_events(d, "SalesOrderItem")
    g = e.groupby(OID)
    out = pd.DataFrame({
        "il": g["il"].first(),
        "criado": e[e[ACT] == "Sales Order Item Created"].groupby(OID)[TS].min(),
        "backorder": g[ACT].apply(lambda s: "Backorder Registered" in set(s)),
    }).dropna(subset=["criado", "il"])
    return out


def purchase_order_table(d: pm.OcelData) -> pd.DataFrame:
    """Um registro por PurchaseOrderItem: IL, criação, fim (recebimento ou
    rejeição), lead time e se teve atraso/rejeição."""
    e = _object_events(d, "PurchaseOrderItem")
    first = e.groupby([OID, ACT])[TS].min().unstack()
    il = e.groupby(OID)["il"].first()
    created = first.get("Purchase Order Item Created")
    received = first.get("Goods Receipt")
    rejected = first.get("Receipt Rejected")
    delayed = first.get("Delivery Delayed")
    out = pd.DataFrame({"il": il, "criado": created})
    out["recebido"] = received if received is not None else pd.NaT
    out["rejeitado"] = rejected if rejected is not None else pd.NaT
    out["atrasado"] = delayed.notna() if delayed is not None else False
    out["fim"] = out["recebido"].fillna(out["rejeitado"])
    out = out.dropna(subset=["criado", "fim", "il"])
    out = out[out["fim"] >= out["criado"]]
    out["lead time (dias)"] = (out["fim"] - out["criado"]).dt.total_seconds() / 86400
    out["situação"] = np.where(out["rejeitado"].notna(), "rejeitado",
                               np.where(out["atrasado"], "atrasado", "normal"))
    return out


# ===========================================================================
# 1) FORNECEDOR → CLIENTE
# ===========================================================================
SUPPLY_STATES = ["sem pedido de compra em aberto", "pedido em aberto, no prazo", "pedido em aberto, atrasado"]


def supplier_to_customer(d: pm.OcelData, overdue_days: float | None = None) -> dict:
    """Situação dos PEDIDOS DE COMPRA de um IL no momento em que cada PEDIDO DE
    VENDA do mesmo IL é criado, e a consequência para o cliente (backorder).

    Para cada SalesOrderItem classifica o IL, naquele instante, em:
      * sem pedido de compra em aberto;
      * pedido em aberto, no prazo   (todos com idade ≤ ``overdue_days``);
      * pedido em aberto, atrasado   (algum com idade > ``overdue_days``).
    ``overdue_days`` padrão = mediana do lead time observado no próprio log.

    A comparação é feita DENTRO de cada IL (diferença em relação à taxa de
    backorder média do próprio IL), para não confundir com o fato de ILs de
    alta demanda terem mais backorders. O resultado agregado é a média dessas
    diferenças, ponderada pelo nº de pedidos de venda."""
    so = sales_order_table(d)
    po = purchase_order_table(d)
    if overdue_days is None:
        overdue_days = float(po["lead time (dias)"].median()) if len(po) else 20.0
    po_by_il = {il: g[["criado", "fim"]].to_numpy() for il, g in po.groupby("il")}

    state, oldest = [], []
    for il, t in zip(so["il"], so["criado"]):
        arr = po_by_il.get(il)
        if arr is None:
            state.append(SUPPLY_STATES[0])
            oldest.append(np.nan)
            continue
        # pedidos de compra "em aberto" no instante t: criados antes de t e
        # ainda não recebidos/rejeitados em t (True/False para cada pedido)
        open_mask = (arr[:, 0] <= t) & (arr[:, 1] > t)
        if not open_mask.any():
            state.append(SUPPLY_STATES[0])
            oldest.append(np.nan)
            continue
        # idade (em dias) de cada pedido de compra em aberto naquele instante
        ages = np.array([(t - c).total_seconds() / 86400 for c in arr[open_mask, 0]])
        state.append(SUPPLY_STATES[2] if (ages > overdue_days).any() else SUPPLY_STATES[1])
        oldest.append(float(ages.max()))   # há quantos dias o IL espera o fornecedor
    so = so.copy()
    so["situação do abastecimento"] = state
    so["espera pelo fornecedor (dias)"] = oldest
    so["backorder"] = so["backorder"].astype(float)

    # COMPARAR DENTRO DO PRÓPRIO IL: alguns ILs têm muito mais backorders que
    # outros (demanda alta). Se comparássemos ILs diferentes entre si,
    # misturaríamos esse efeito com o do fornecedor. Por isso calculamos a taxa
    # média de backorder de cada IL (groupby + transform repete essa média em
    # todas as linhas do IL) e medimos o "excesso" de cada pedido de venda em
    # relação a ela. Em estatística isso se chama controlar pelo "efeito fixo" do IL.
    # Ex.: IL com 40% de backorder em geral; pedido que virou backorder → excesso = 1 − 0,40 = +0,60.
    so["taxa do IL"] = so.groupby("il")["backorder"].transform("mean")
    so["excesso"] = so["backorder"] - so["taxa do IL"]

    summary = so.groupby("situação do abastecimento").agg(
        pedidos_de_venda=("il", "size"),
        pct_backorder_bruto=("backorder", "mean"),
        excesso_vs_media_do_il=("excesso", "mean"),
    ).reindex(SUPPLY_STATES).reset_index()
    summary["pct_backorder_bruto"] *= 100
    summary["excesso_vs_media_do_il"] *= 100   # em pontos percentuais

    # % de backorder conforme há quantos dias o IL espera o pedido de compra mais antigo
    bins = [0, 7, 14, 28, 56, np.inf]
    labels = ["0–7 dias", "7–14 dias", "14–28 dias", "28–56 dias", "> 56 dias"]
    w = so.dropna(subset=["espera pelo fornecedor (dias)"]).copy()
    # pd.cut classifica cada valor numa faixa (0–7 dias, 7–14 dias...)
    w["faixa de espera"] = pd.cut(w["espera pelo fornecedor (dias)"], bins=bins, labels=labels, right=False)
    by_wait = w.groupby("faixa de espera", observed=False).agg(
        pedidos_de_venda=("il", "size"),
        pct_backorder_bruto=("backorder", "mean"),
        excesso_vs_media_do_il=("excesso", "mean"),
    ).reset_index()
    by_wait[["pct_backorder_bruto", "excesso_vs_media_do_il"]] *= 100
    rho_w, p_w = (stats.spearmanr(w["espera pelo fornecedor (dias)"], w["excesso"])
                  if len(w) > 5 else (np.nan, np.nan))

    # teste de Mann–Whitney: o excesso de backorder é MAIOR quando o pedido de
    # compra está atrasado do que quando está no prazo? (p-valor baixo = sim)
    a = so.loc[so["situação do abastecimento"] == SUPPLY_STATES[2], "excesso"]
    b = so.loc[so["situação do abastecimento"] == SUPPLY_STATES[1], "excesso"]
    test = stats.mannwhitneyu(a, b, alternative="greater") if len(a) > 5 and len(b) > 5 else None

    # tabela IL × situação com a % de backorder (pivot_table = tabela dinâmica),
    # para ver se o padrão se repete em cada IL
    per_il = so.pivot_table(index="il", columns="situação do abastecimento", values="backorder",
                            aggfunc="mean").reindex(columns=SUPPLY_STATES) * 100
    return {"summary": summary, "by_wait": by_wait, "spearman_wait": (float(rho_w), float(p_w)),
            "per_il": per_il, "overdue_days": overdue_days,
            "test_p": float(test.pvalue) if test is not None else np.nan, "per_so": so}


# ===========================================================================
# 2) CADEIA DE REPOSIÇÃO → FALTA DE ESTOQUE (por IL)
# ===========================================================================
def replenishment_to_stockout(d: pm.OcelData) -> dict:
    """Por ItemLocation: lead time médio, tempo de aprovação, demanda e % backorder."""
    so = sales_order_table(d)
    po = purchase_order_table(d)
    rp = _object_events(d, "ReplenishmentProposal")
    f = rp.groupby([OID, ACT])[TS].min().unstack()
    rp_il = rp.groupby(OID)["il"].first()
    appr = pd.Series(dtype=float)
    if "Replenishment Proposal Approved" in f and "Replenishment Proposal Created" in f:
        appr = (f["Replenishment Proposal Approved"] - f["Replenishment Proposal Created"]).dt.total_seconds() / 86400
    horizon = max(d.horizon_days, 1)
    t = pd.DataFrame({
        "pedidos de venda/dia": so.groupby("il").size() / horizon,
        "% backorder": so.groupby("il")["backorder"].mean() * 100,
        "lead time médio (dias)": po.groupby("il")["lead time (dias)"].mean(),
        "pedidos de compra": po.groupby("il").size(),
        "aprovação média (dias)": appr.groupby(rp_il).mean() if len(appr) else np.nan,
    }).dropna(subset=["% backorder"])
    corr = []
    for c in ["pedidos de venda/dia", "lead time médio (dias)", "aprovação média (dias)", "pedidos de compra"]:
        x = t[[c, "% backorder"]].dropna()
        if len(x) > 3 and x[c].nunique() > 1:
            rho, p = stats.spearmanr(x[c], x["% backorder"])
            corr.append({"fator (por IL)": c, "ρ de Spearman com % backorder": rho, "p-valor": p, "ILs": len(x)})
    return {"per_il": t.reset_index(names="il"), "correlations": pd.DataFrame(corr)}


# ===========================================================================
# 3) DEMANDA → PEDIDOS (correlação cruzada com defasagem)
# ===========================================================================
def lagged_response(weekly_demand: np.ndarray, weekly_orders: np.ndarray, max_lag: int = 8,
                    warmup_weeks: int = 8) -> pd.DataFrame:
    """Correlação entre a demanda da semana t e os pedidos da semana t+k (k = 0..max_lag).

    Calculada (a) para a rede (soma das ILs) e (b) por IL, reportando a mediana.
    Um pico em k mostra que a política reage à demanda k semanas depois; a altura
    do pico mostra o quanto os pedidos "seguem" a demanda."""
    D = np.asarray(weekly_demand, dtype=float)[:, warmup_weeks:]
    O = np.asarray(weekly_orders, dtype=float)[:, warmup_weeks:]

    def ccf(x, y, k):
        """Correlação entre a demanda x na semana t e os pedidos y na semana t + k.
        Ex.: k = 2 compara a demanda das semanas 1, 2, 3... com os pedidos das
        semanas 3, 4, 5... ("correlação cruzada com defasagem")."""
        if k >= len(x) - 3:
            return np.nan
        a, b = x[: len(x) - k], y[k:]
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    rows = []
    for k in range(max_lag + 1):
        net = ccf(D.sum(0), O.sum(0), k)
        per = [ccf(D[i], O[i], k) for i in range(D.shape[0])]
        per = [v for v in per if np.isfinite(v)]
        rows.append({"defasagem (semanas)": k, "correlação rede": net,
                     "correlação mediana por IL": float(np.median(per)) if per else np.nan})
    return pd.DataFrame(rows)
