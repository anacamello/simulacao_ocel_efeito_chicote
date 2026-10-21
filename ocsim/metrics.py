"""
metrics.py — medidas do efeito chicote e indicadores de desempenho
==================================================================

PAPEL NO PROJETO
----------------
Transforma as séries diárias geradas pela simulação (ou extraídas do log original)
nos números que respondem à pergunta do trabalho: qual regra de reposição
amplifica menos a variabilidade dos pedidos?

EFEITO CHICOTE (Bullwhip Effect, BWE) — razão de variâncias
-----------------------------------------------------------
    BWE = Var(pedidos ao fornecedor) / Var(demanda do cliente)

* BWE = 1 → os pedidos variam tanto quanto a demanda (sem amplificação);
* BWE > 1 → amplificação: a variabilidade cresce rumo ao fornecedor (montante);
* BWE < 1 → suavização.

As séries diárias são somadas em BALDES (padrão: semanas), descartando um
período inicial de AQUECIMENTO (padrão: 56 dias), quando o sistema ainda
carrega o efeito do estado inicial. O BWE é calculado em 4 níveis:
  - por ItemLocation (cada elo individual);
  - por Location (soma das ILs de cada armazém);
  - por Fornecedor (pedidos que o fornecedor recebe ÷ demanda final que ele atende);
  - para a rede inteira (soma de tudo).

EFEITO CHICOTE pelo COEFICIENTE DE VARIAÇÃO (BWE_CV)
---------------------------------------------------
    BWE_CV = CV(pedidos) / CV(demanda),   com CV = desvio-padrão / média

(Fransoo & Wouters, 2000). Compara a variabilidade RELATIVA à média de cada
série, por isso não é distorcida quando a média dos pedidos difere da média da
demanda — ex.: recebimento parcial, perdas, ajustes de inventário ou o efeito
de transição no curto prazo. Interpretação igual à da razão de variâncias
(> 1 = amplificação). Quando as médias são iguais, BWE_CV = √BWE; por isso
seus valores tendem a ser menores, e as duas métricas não devem ser
comparadas diretamente entre si, só cada uma entre cenários.
É calculado nos mesmos 4 níveis e com os mesmos baldes e aquecimento.

Também se calcula a amplificação do estoque líquido
(NSAmp, Disney & Towill, 2003) = Var(estoque líquido) / Var(demanda).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bucket(series: np.ndarray, size: int) -> np.ndarray:
    """Soma as séries diárias em "baldes" de ``size`` dias (ex.: semanas).

    Ex.: 731 dias com size=7 → 104 semanas (os 3 dias que sobram no fim são
    descartados, para todas as semanas terem 7 dias).
    Como funciona: a matriz ILs × 728 dias é remodelada (reshape) para
    ILs × 104 semanas × 7 dias, e então somamos os 7 dias de cada semana."""
    n = series.shape[-1] // size
    s = series[..., : n * size]
    return s.reshape(*series.shape[:-1], n, size).sum(-1)


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    """Razão de variâncias Var(num) / Var(den).
    ``ddof=1`` = variância amostral (divide por n − 1).
    Devolve NaN ("não é um número", isto é, valor indefinido) quando o
    denominador não varia — ex.: um IL sem nenhuma demanda —, pois não há
    como dividir por zero."""
    vd = np.var(den, ddof=1)
    if vd <= 1e-12:
        return np.nan
    return float(np.var(num, ddof=1) / vd)


def cv(x: np.ndarray) -> float:
    """Coeficiente de variação amostral: desvio-padrão (ddof=1) / média.
    NaN quando a média é ~0 (série sem movimento): o CV não está definido."""
    x = np.asarray(x, dtype=float)
    m = x.mean() if x.size else 0.0
    if x.size < 2 or abs(m) <= 1e-12:
        return np.nan
    return float(np.std(x, ddof=1) / m)


def _cv_ratio(num: np.ndarray, den: np.ndarray) -> float:
    """Razão de coeficientes de variação CV(num) / CV(den).
    NaN se algum CV for indefinido ou se a demanda não variar (CV = 0)."""
    cn, cd = cv(num), cv(den)
    if not (np.isfinite(cn) and np.isfinite(cd)) or cd <= 1e-12:
        return np.nan
    return float(cn / cd)


def bullwhip(
    demand: np.ndarray,
    orders: np.ndarray,
    ils: list[str],
    il_location: dict,
    il_supplier_orders: dict | None = None,
    il_supplier_probs: dict | None = None,
    net_stock: np.ndarray | None = None,
    bucket_days: int = 7,
    warmup_days: int = 56,
    min_weekly_demand: float = 1.0,
) -> dict:
    """Calcula o efeito chicote em vários níveis.

    demand, orders     : matrizes IL × dia (demanda do cliente e pedidos ao fornecedor)
    ils                : ids dos ILs, na ordem das linhas
    il_location        : {IL: Location}
    il_supplier_orders : {fornecedor: série diária de pedidos recebidos}
    il_supplier_probs  : {IL: {fornecedor: proporção}} — para montar a demanda que cada fornecedor atende
    net_stock          : matriz IL × dia do estoque líquido (para o NSAmp)
    min_weekly_demand  : ILs com demanda média menor que isso ficam fora (razão instável)

    Retorna {"per_il": tabela por IL, "levels": BWE por nível,
             "levels_cv": BWE_CV por nível, "summary": resumo (as duas métricas)}."""
    # 1) descarta o aquecimento e agrega em baldes
    w0 = warmup_days
    D = bucket(demand[:, w0:], bucket_days)
    O = bucket(orders[:, w0:], bucket_days)

    # 2) BWE por ItemLocation
    rows = []
    for i, il in enumerate(ils):
        md = D[i].mean()
        ok = md >= min_weekly_demand
        r = {
            "il": il,
            "location": il_location.get(il),
            "demanda média/sem": md,
            "pedidos média/sem": O[i].mean(),
            "var demanda": np.var(D[i], ddof=1),
            "var pedidos": np.var(O[i], ddof=1),
            "CV demanda": cv(D[i]),
            "CV pedidos": cv(O[i]),
            "BWE": _ratio(O[i], D[i]) if ok else np.nan,
            "BWE_CV": _cv_ratio(O[i], D[i]) if ok else np.nan,
        }
        if net_stock is not None:
            ns = net_stock[i, w0:]
            r["NSAmp"] = _ratio(ns, demand[i, w0:]) if md >= min_weekly_demand else np.nan
        rows.append(r)
    per_il = pd.DataFrame(rows)

    # 3) BWE agregado: rede inteira e por Location
    levels = {"Rede (total)": _ratio(O.sum(0), D.sum(0))}
    levels_cv = {"Rede (total)": _cv_ratio(O.sum(0), D.sum(0))}
    for loc in sorted(set(il_location.values())):
        idx = [i for i, il in enumerate(ils) if il_location.get(il) == loc]
        levels[f"Location {loc}"] = _ratio(O[idx].sum(0), D[idx].sum(0))
        levels_cv[f"Location {loc}"] = _cv_ratio(O[idx].sum(0), D[idx].sum(0))

    # 4) BWE por fornecedor: pedidos recebidos ÷ demanda final ponderada pela
    #    proporção de cada IL atendida por esse fornecedor
    if il_supplier_orders:
        for sup, arr in il_supplier_orders.items():
            so = bucket(np.asarray(arr)[w0: demand.shape[1]], bucket_days)
            if so.sum() > 0 and len(so) == D.shape[1]:
                w = np.array([(il_supplier_probs or {}).get(il, {}).get(sup, 0.0) for il in ils])
                if w.sum() >= 1.0:  # ignora fornecedores marginais (atendem menos de 1 IL inteiro)
                    dsup = (D * w[:, None]).sum(0)
                    levels[f"Fornecedor {sup}"] = _ratio(so, dsup)
                    levels_cv[f"Fornecedor {sup}"] = _cv_ratio(so, dsup)

    # 5) resumo — as duas métricas, com os mesmos agregados
    summary = {}
    for col, lv in (("BWE", levels), ("BWE_CV", levels_cv)):
        valid = per_il[col].dropna()
        summary[f"{col} rede"] = lv["Rede (total)"]
        summary[f"{col} médio IL"] = float(valid.mean()) if len(valid) else np.nan
        summary[f"{col} mediano IL"] = float(valid.median()) if len(valid) else np.nan
        summary[f"% IL com {col}>1"] = float((valid > 1).mean() * 100) if len(valid) else np.nan
    if net_stock is not None:
        summary["NSAmp mediano IL"] = float(per_il["NSAmp"].dropna().median())
    return {"per_il": per_il, "levels": levels, "levels_cv": levels_cv, "summary": summary}


def kpis(sim, params, cfg, warmup_days: int = 56) -> dict:
    """Indicadores de serviço e custo de UMA replicação (após o aquecimento).

    * fill rate      = unidades atendidas na hora ÷ unidades demandadas;
    * custo de estocagem = Σ_IL estoque médio × custo unitário × h × anos;
    * custo de pedidos   = nº de POs × K (proporcional ao período medido);
    * custo de falta     = Σ (unidades em backorder por dia) × custo de falta."""
    w0 = warmup_days
    H_years = max(sim.ndays - w0, 1) / 365
    unit_cost = np.array([params.item_locations[il].unit_cost for il in sim.ils])
    oh = sim.on_hand_eod[:, w0: sim.ndays]   # estoque no fim de cada dia
    bo = sim.backorder_eod[:, w0: sim.ndays]  # backorder no fim de cada dia
    holding = float((oh.mean(1) * unit_cost * cfg.holding_rate).sum() * H_years)
    ordering = float(sim.n_po * cfg.order_cost * (sim.ndays - w0) / max(sim.ndays, 1))
    shortage = float(bo.sum() * cfg.backorder_cost)
    return {
        "nível de serviço (fill rate)": sim.served_immediately / max(sim.total_demand, 1e-9),
        "% pedidos de venda com backorder": sim.so_with_backorder / max(sim.n_so, 1) * 100,
        "estoque médio (un/IL)": float(oh.mean()),
        "backorder médio (un/IL)": float(bo.mean()),
        "nº pedidos de compra": sim.n_po,
        "lead time médio (dias)": float(np.mean(sim.lead_times)) if sim.lead_times else np.nan,
        "custo de estocagem": holding,
        "custo de pedidos": ordering,
        "custo de falta": shortage,
        "custo total": holding + ordering + shortage,
    }
