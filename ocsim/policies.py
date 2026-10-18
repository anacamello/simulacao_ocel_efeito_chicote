"""
policies.py — Etapa 4: regras de reposição (os cenários da simulação)
=====================================================================

PAPEL NO PROJETO
----------------
Define COMO cada ItemLocation decide quando e quanto pedir ao fornecedor.
O simulador chama estas regras; a forma como elas reagem à demanda é o
que amplifica (ou não) a variabilidade dos pedidos — o efeito chicote.

NOTAÇÃO
-------
  d̄  = demanda diária média (estimada do log)
  σ  = desvio-padrão da demanda diária
  L  = lead time total de reposição (proposta → recebimento, minerado do log)
  σ_L= desvio do lead time
  R  = intervalo de revisão (dias)
  z  = Φ⁻¹(nível de serviço)  — ex.: 95% → z = 1,645
  Q  = lote econômico (EOQ) arredondado ao lot_size do material
  F  = previsão da demanda diária (estática ou atualizada todo dia)
  IP = posição de estoque = em mão + em pedido + propostas pendentes − backorders

CENÁRIOS
--------
* ATUAL – política observada no log (mín–máx): se IP ≤ s, pede até S.
  Revisão diária com a probabilidade calibrada no log. É a linha de base e
  serve para VALIDAR o simulador.
* SQ  – Lote Fixo e Ponto de Pedido (s, Q), revisão contínua:
        s = F·L + z·σ_L      ;  quando IP ≤ s, pede Q (ou múltiplos de Q).
* RS  – Revisão Periódica com Intervalo Fixo (R, S) — "order-up-to":
        a cada R dias pede S − IP, com S = F·(L+R) + z·σ_(L+R).
        Com previsão dinâmica, S muda a cada revisão — mecanismo que
        Dejonckheere et al. (2003) mostram amplificar a variância dos pedidos.
* RSS – Revisão Periódica com Intervalo Fixo e Ponto de Pedido (R, s, S):
        a cada R dias, SE IP ≤ s, pede S − IP;  s = F·(L+R)+z·σ_(L+R), S = s + Q.
* L4L – Lote a Lote: revisão diária, pede exatamente a necessidade líquida do
        dia (base-stock fixo ⇒ em regime repõe o que foi consumido, sem lote).

O desvio de proteção combina a incerteza da demanda e a do lead time:
        σ_T = √( T·σ² + F²·σ_L² )
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.stats import norm

from .parameters import ILParams


# nomes exibidos na interface
POLICY_LABELS = {
    "ATUAL": "Atual (mín–máx do log)",
    "SQ": "Lote Fixo e Ponto de Pedido (s, Q)",
    "RS": "Revisão Periódica com Intervalo Fixo (R, S)",
    "RSS": "Revisão Periódica com Intervalo Fixo e Ponto de Pedido (R, s, S)",
    "L4L": "Lote a Lote (L4L)",
}

# nomes curtos (eixos dos gráficos)
POLICY_SHORT = {"ATUAL": "Atual", "SQ": "(s,Q)", "RS": "(R,S)", "RSS": "(R,s,S)", "L4L": "Lote a lote"}


# ===========================================================================
# Configuração de um cenário (o que o usuário escolhe na etapa 4)
# ===========================================================================
@dataclass
class ScenarioConfig:
    policy: str = "RS"                  # código da política (ver POLICY_LABELS)
    name: str = ""                      # nome personalizado (opcional)
    review_period: int = 7              # R em dias (para RS e RSS)
    forecast: str = "media_movel"       # 'estatico' | 'media_movel' | 'suav_exponencial'
    ma_window: int = 14                 # janela da média móvel (dias)
    es_alpha: float = 0.2               # α da suavização exponencial
    service_level: float | None = None  # None = usa o alvo de cada Location (log)
    order_cost: float = 100.0           # K: custo fixo por pedido
    holding_rate: float = 0.25          # h: custo de estocagem (fração do custo unitário por ano)
    backorder_cost: float = 5.0         # custo por unidade·dia em falta
    lead_time_factor: float = 1.0       # sensibilidade: multiplica o lead time
    demand_factor: float = 1.0          # sensibilidade: multiplica a demanda
    include_lt_variance: bool = True    # incluir a variabilidade do lead time no estoque de segurança
    round_to_lot: bool = True           # arredondar pedidos ao lot_size
    staggered_reviews: bool = True      # ILs revisam em dias diferentes (evita picos sincronizados)
    backorder_accounting: str = "correta"  # 'correta' | 'log' (reproduz a contabilidade do log)
    partial_receipts: bool = True       # recebimento parcial conforme a razão recebido/pedido do log

    def label(self) -> str:
        return self.name or POLICY_LABELS.get(self.policy, self.policy)

    def to_dict(self) -> dict:
        """Converte em dicionário (necessário para enviar a processos paralelos)."""
        return asdict(self)


# ===========================================================================
# Estado da política de UM IL durante a simulação
# ===========================================================================
@dataclass
class PolicyState:
    """Guarda a previsão de demanda e os parâmetros correntes (s, S, Q) de um IL."""

    il: ILParams
    cfg: ScenarioConfig
    z: float          # fator de segurança
    L: float          # lead time médio usado no planejamento
    sigma_L: float    # desvio do lead time
    Q: float          # lote (EOQ)
    forecast: float   # previsão atual da demanda diária (F)
    sigma: float      # desvio atual da demanda diária
    history: list = field(default_factory=list)  # demanda diária observada na simulação
    s: float = 0.0    # ponto de pedido corrente
    S: float = 0.0    # nível máximo corrente

    # ---------------------------------------------------------------- previsão
    def observe_day(self, demand: float):
        """Chamado uma vez por dia com a demanda do dia: atualiza a previsão
        e recalcula s e S. É aqui que nasce o efeito chicote das previsões dinâmicas."""
        cfg = self.cfg
        self.history.append(demand)
        if cfg.forecast == "media_movel":
            # média e desvio dos últimos N dias
            w = self.history[-cfg.ma_window:]
            self.forecast = float(np.mean(w))
            if len(w) >= 3:
                self.sigma = max(float(np.std(w, ddof=1)), 1e-6)
        elif cfg.forecast == "suav_exponencial":
            # F_novo = α·demanda + (1−α)·F_antigo
            a = cfg.es_alpha
            err = demand - self.forecast
            self.forecast = a * demand + (1 - a) * self.forecast
            # desvio estimado pelo MAD (erro absoluto médio) suavizado: σ ≈ 1,25·MAD
            mad = getattr(self, "_mad", self.sigma / 1.25)
            mad = a * abs(err) + (1 - a) * mad
            self._mad = mad
            self.sigma = max(1.25 * mad, 1e-6)
        # 'estatico': mantém a média e o desvio estimados do log
        self.recompute()

    def protection_sigma(self, horizon: float) -> float:
        """Desvio da demanda durante o período de proteção T:
        σ_T = √(T·σ² + F²·σ_L²)."""
        var = horizon * self.sigma ** 2
        if self.cfg.include_lt_variance:
            var += (self.forecast ** 2) * (self.sigma_L ** 2)
        return math.sqrt(max(var, 0.0))

    def recompute(self):
        """Recalcula s e S com a previsão atual."""
        cfg = self.cfg
        R = cfg.review_period
        F, L = self.forecast, self.L
        if cfg.policy == "SQ":
            # período de proteção = L (revisão contínua)
            self.s = F * L + self.z * self.protection_sigma(L)
            self.S = self.s + self.Q
        elif cfg.policy in ("RS", "RSS"):
            # período de proteção = L + R (o próximo pedido só sai daqui a R dias)
            self.s = F * (L + R) + self.z * self.protection_sigma(L + R)
            self.S = self.s if cfg.policy == "RS" else self.s + self.Q
        elif cfg.policy == "L4L":
            pass  # base-stock fixo, definido uma única vez em make_policy_state
        elif cfg.policy == "ATUAL":
            pass  # s e S vêm do log (podem mudar via 'Policy Threshold Updated')

    def round_qty(self, q: float) -> float:
        """Arredonda a quantidade pedida: ao lot_size (políticas com lote) ou à unidade."""
        if q <= 0:
            return 0.0
        if self.cfg.round_to_lot and self.cfg.policy in ("SQ", "RS", "RSS"):
            lot = max(self.il.lot_size, 1)
            return float(math.ceil(q / lot) * lot)
        return float(math.ceil(q))


# ===========================================================================
# Lote econômico de compra (EOQ)
# ===========================================================================
def eoq(il: ILParams, cfg: ScenarioConfig, daily_demand: float) -> float:
    """Q* = √(2·D·K / (h·c)), arredondado para cima ao lot_size.
    D = demanda anual, K = custo por pedido, h = taxa de estocagem, c = custo unitário."""
    D = max(daily_demand, 1e-6) * 365
    h = max(cfg.holding_rate * il.unit_cost, 1e-6)
    q = math.sqrt(2 * D * cfg.order_cost / h)
    lot = max(il.lot_size, 1)
    return float(max(lot, math.ceil(q / lot) * lot))


# ===========================================================================
# Cria o estado inicial da política de um IL
# ===========================================================================
def make_policy_state(il: ILParams, cfg: ScenarioConfig, lt_mean: float, lt_std: float) -> PolicyState:
    """Calcula z, Q, a previsão inicial e os níveis s/S iniciais de um IL."""
    # nível de serviço: o escolhido pelo usuário ou o alvo da Location
    sl = cfg.service_level if cfg.service_level else il.service_level
    sl = min(max(sl, 0.5), 0.9999)
    z = float(norm.ppf(sl))
    # previsão inicial = estatísticas do log (multiplicadas pelo fator de sensibilidade)
    d = il.daily_mean * cfg.demand_factor
    sd = max(il.daily_std * cfg.demand_factor, 1e-6)
    st = PolicyState(il=il, cfg=cfg, z=z, L=lt_mean, sigma_L=lt_std, Q=eoq(il, cfg, d), forecast=d, sigma=sd)
    if cfg.policy == "L4L":
        # base-stock = demanda durante (L+1) dias + estoque de segurança
        st.S = d * (lt_mean + 1) + z * st.protection_sigma(lt_mean + 1)
        st.s = st.S
    elif cfg.policy == "ATUAL":
        st.s, st.S = il.s, il.S  # limites do log (20 / 80 ou 90)
    else:
        st.recompute()
    return st


# ===========================================================================
# Cenários padrão (os cinco pedidos no trabalho)
# ===========================================================================
def default_scenarios() -> list[ScenarioConfig]:
    return [
        # Atual: reproduz o log, inclusive a contabilidade de backorder (validação)
        ScenarioConfig(policy="ATUAL", forecast="estatico", review_period=1, backorder_accounting="log"),
        ScenarioConfig(policy="SQ", forecast="estatico"),
        ScenarioConfig(policy="RS", review_period=7, forecast="media_movel"),
        ScenarioConfig(policy="RSS", review_period=7, forecast="estatico"),
        ScenarioConfig(policy="L4L", review_period=1, forecast="estatico"),
    ]
