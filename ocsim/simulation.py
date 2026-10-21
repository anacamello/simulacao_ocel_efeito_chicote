"""
simulation.py — Etapa 5: simulação de eventos discretos centrada em objetos (SimPy)
===================================================================================

PAPEL NO PROJETO
----------------
É o "motor" do trabalho. Recebe os parâmetros extraídos do log
(``ModelParameters``) e uma regra de reposição (``ScenarioConfig``) e simula,
dia a dia, toda a cadeia: pedidos de venda chegando, estoque sendo consumido,
propostas de reposição, pedidos de compra, entregas e recebimentos.

COMO O SIMPY FUNCIONA (resumo)
-------------------------------
Uma simulação de eventos discretos não avança o tempo de segundo em segundo:
ela pula direto de um acontecimento para o próximo.

* ``simpy.Environment`` é o relógio da simulação. Aqui, 1 unidade = 1 dia;
  0,5 = meio-dia; 0,333 = 8h da manhã.
* Um PROCESSO é uma sequência de passos com esperas no meio, escrita como uma
  função que pode "pausar" (função geradora — ver glossário em __init__.py).
  ``yield env.timeout(2.5)`` quer dizer: "pause este processo por 2,5 dias de
  simulação e depois continue da linha seguinte".
* Muitos processos existem ao mesmo tempo (um por pedido de venda, um por
  reposição...). O SimPy mantém uma agenda e sempre executa o próximo
  acontecimento no tempo.
* ``env.process(f())`` coloca um processo na agenda; ``env.run(until=H)`` roda
  a agenda até o instante H.

Exemplo de linha do tempo de UM pedido de compra:
   dia 10,7  proposta criada        → yield timeout(1,5)  (aprovação)
   dia 12,2  proposta aprovada      → yield timeout(0,01) (emissão do PO)
   dia 12,2  PO criado              → yield timeout(22)   (lead time)
   dia 34,2  mercadoria recebida → estoque do IL aumenta
Enquanto esse processo está "esperando", centenas de outros (pedidos de
venda, outras reposições) acontecem normalmente.

ESTRUTURA CENTRADA EM OBJETOS (inspirada no OCPS, Knopp & van der Aalst)
------------------------------------------------------------------------
1. MODELO DE OBJETOS – objetos estruturais (Location, Material, Supplier,
   Planner, ItemLocation) e suas ligações O2O vêm do log. Objetos
   transacionais (SalesOrderItem, ReplenishmentProposal, PurchaseOrderItem,
   Delivery, TransferOrder) são CRIADOS durante a simulação, respeitando as
   cardinalidades mineradas (1 SOI → 1 IL; 1 PO → 1 IL e 1 Supplier;
   0..1 Delivery por PO; TransferOrder com IL de origem e de destino).
2. COMPORTAMENTO POR TIPO DE OBJETO – cada tipo transacional tem um processo
   SimPy que percorre as atividades do seu ciclo de vida; as durações entre
   atividades são sorteadas das distribuições ajustadas ao log.
3. SINCRONIZAÇÃO – um evento compartilhado (ex.: Goods Receipt envolve PO,
   Delivery, IL, Material, Location e Supplier) é registrado UMA vez com
   todas as relações E2O, como no OCEL original.
4. SAÍDA – séries diárias por IL (para o efeito chicote), contadores e,
   opcionalmente, o OCEL simulado completo.

MAPA DOS PROCESSOS
------------------
  demand_generator (1 por IL) ──cria──▶ sales_order (1 por pedido de venda)
  review_process   (1 por IL) ──decide──▶ place_order ──▶ replenishment
                                             (proposta → aprovação → PO → recebimento)
  daily_snapshot   (1 global) ── grava estoque/backorder no fim de cada dia
  exceptions_process (1 global) ── ajustes, contagens, transferências, lacunas de dados
  open_po (curto prazo) ── POs herdados do log que ainda vão chegar
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import simpy

from .parameters import ModelParameters, ILParams
from .policies import ScenarioConfig, PolicyState, make_policy_state, spread_lead_time

REVIEW_HOUR = 17.5 / 24  # hora padrão da revisão (17h30) se não houver distribuição no log


# ===========================================================================
# Estado de estoque de UM ItemLocation durante a simulação
# (os valores mudam a cada evento: venda, recebimento, ajuste...)
# ===========================================================================
@dataclass  # "ficha" com campos (ver glossário)
class ILState:
    p: ILParams                    # parâmetros fixos do IL (vindos do log)
    pol: PolicyState               # estado da política de reposição (previsão, s, S)
    on_hand: float                 # estoque físico em mão
    reserved: float = 0.0          # reservado para pedidos de venda ainda não expedidos
    on_order: float = 0.0          # pedidos de compra emitidos e não recebidos
    pending: float = 0.0           # propostas criadas que ainda não viraram PO
    # fila de pendências na ordem de chegada (o primeiro a entrar é o primeiro
    # a ser atendido). Cada item é [id do pedido de venda, quantidade que falta].
    # "deque" é um tipo de lista eficiente para tirar itens do começo.
    backlog: deque = field(default_factory=deque)
    backorder: float = 0.0         # total pendente
    data_gap: bool = False         # True durante uma lacuna de dados (revisões suspensas)
    asis_S: float = 0.0            # nível máximo da política atual (pode mudar no tempo)
    consumed_today: float = 0.0    # demanda acumulada no dia (alimenta a previsão)

    # @property: permite escrever st.ip_log (sem parênteses), como se fosse um campo,
    # mas o valor é calculado na hora a partir dos outros campos
    @property
    def ip_log(self) -> float:
        """Posição de estoque como o log a registra (sem propostas pendentes).
        Usada pela política ATUAL, para reproduzir o comportamento do log."""
        return self.on_hand + self.on_order - self.backorder

    @property
    def ip(self) -> float:
        """Posição de estoque completa = em mão + em pedido + pendente − backorder.
        Inclui as propostas ainda não aprovadas para não pedir em dobro."""
        return self.on_hand + self.on_order + self.pending - self.backorder

    @property
    def available(self) -> float:
        """Estoque livre para novos pedidos de venda."""
        return self.on_hand - self.reserved


# ===========================================================================
# A SIMULAÇÃO
# ===========================================================================
class SupplyChainSimulation:
    def __init__(
        self,
        params: ModelParameters,
        cfg: ScenarioConfig,
        seed: int = 0,
        horizon_days: float | None = None,
        include_exceptions: bool = True,
        record_ocel: bool = False,
        sampling_mode: str = "parametric",
        demand_model: str = "tendencia_sazonal",
        initial_state: str = "inicio_log",
        inherit_backorders: bool = False,
    ):
        """
        params             : parâmetros extraídos do log
        cfg                : cenário (regra de reposição e custos)
        seed               : semente aleatória (mesma semente → mesmo resultado)
        horizon_days       : duração da simulação (padrão = duração do log)
        include_exceptions : simular eventos raros (ajustes, transferências...)
        record_ocel        : guardar todos os eventos/objetos para exportar o OCEL
        sampling_mode      : 'parametric' (distribuições ajustadas) ou 'empirical' (valores do log)
        demand_model       : 'tendencia_sazonal' ou 'estacionario'
        initial_state      : 'inicio_log' ou 'estado_atual' (fim do log, curto prazo)
        inherit_backorders : no curto prazo, herdar os backorders acumulados do log
        """
        self.P = params
        self.cfg = cfg
        # gerador de números aleatórios desta replicação: TODOS os sorteios da
        # simulação passam por ele, por isso a mesma semente repete o resultado
        self.rng = np.random.default_rng(seed)
        self.H = float(horizon_days or params.horizon_days)
        self.ndays = int(math.ceil(self.H))
        self.include_exceptions = include_exceptions
        self.record_ocel = record_ocel
        self.mode = sampling_mode
        self.demand_model = demand_model

        # --- estado inicial (início do log ou fim do log)
        cs = params.current_state if initial_state == "estado_atual" else None
        self.cs = cs
        self.inherit_backorders = inherit_backorders
        # deslocamento (dias) do início da simulação em relação ao início do log;
        # usado para continuar a tendência e o calendário no curto prazo
        self.t0 = float(cs["offset_days"]) if cs else 0.0

        self.env = simpy.Environment()   # o relógio e a agenda da simulação (começa em 0)
        self.ils = list(params.item_locations)
        self.il_index = {il: i for i, il in enumerate(self.ils)}
        n = len(self.ils)

        # --- séries diárias (linhas = ILs, colunas = dias) para o efeito chicote
        self.demand = np.zeros((n, self.ndays + 1))         # demanda do cliente
        self.orders = np.zeros((n, self.ndays + 1))         # pedidos ao fornecedor
        self.receipts = np.zeros((n, self.ndays + 1))       # recebimentos
        self.on_hand_eod = np.zeros((n, self.ndays + 1))    # estoque no fim do dia
        self.backorder_eod = np.zeros((n, self.ndays + 1))  # backorder no fim do dia
        self.ip_eod = np.zeros((n, self.ndays + 1))         # posição de estoque no fim do dia
        self.orders_by_supplier: dict[str, np.ndarray] = {}  # pedidos recebidos por fornecedor

        # --- contadores para KPIs e validação
        self.counts = Counter()        # nº de eventos por atividade
        self.obj_counts = Counter()    # nº de objetos criados por tipo
        self.served_immediately = 0.0  # unidades atendidas do estoque na hora
        self.total_demand = 0.0
        self.so_with_backorder = 0
        self.n_so = 0
        self.n_po = 0
        self.lead_times = []           # lead times realizados (validação)
        self.so_variants = Counter()   # variantes do SalesOrderItem
        self.po_variants = Counter()   # variantes do PurchaseOrderItem

        # --- armazenamento do OCEL simulado (só se record_ocel=True; guardar
        #     cada evento ocupa memória, então isso é feito só quando pedido)
        self.events: list = []
        self.objects: dict = {}
        self._eid = 0
        self._oid = Counter()
        self.state: dict[str, ILState] = {}

        # --- calendário de cada dia simulado (mês, dia da semana, dia do mês)
        _dates = pd.date_range(params.start + pd.Timedelta(days=self.t0), periods=self.ndays + 1, freq="D")
        self.month_of_day = _dates.month.to_numpy()
        self.dow_of_day = _dates.dayofweek.to_numpy()
        self.dom_of_day = _dates.day.to_numpy()

        self._setup_objects()

    # =======================================================================
    # Funções auxiliares
    # =======================================================================
    def d(self, key: str) -> float:
        """Sorteia uma duração (dias) da distribuição ``key`` (nunca negativa)."""
        return max(0.0, float(self.P.delays[key].sample(self.rng, self.mode)))

    def bern(self, p: float) -> bool:
        """Sorteio sim/não com probabilidade p."""
        return self.rng.random() < p

    def new_oid(self, otype: str, prefix: str) -> str:
        """Cria um id novo para um objeto simulado (ex.: POI-S000123).
        O "S" no id diferencia objetos simulados dos objetos do log original."""
        self._oid[otype] += 1
        self.obj_counts[otype] += 1
        return f"{prefix}-S{self._oid[otype]:06d}"

    def add_object(self, oid: str, otype: str, attrs: dict | None = None):
        """Registra um objeto no OCEL simulado (com instante de criação)."""
        if self.record_ocel:
            self.objects[oid] = {"type": otype, "attrs": dict(attrs or {}), "changes": [], "o2o": [],
                                 "time": self.env.now}

    def set_attr(self, oid: str, name: str, value):
        """Registra a mudança de um atributo ao longo do tempo (ex.: status → received)."""
        if self.record_ocel and oid in self.objects:
            self.objects[oid]["changes"].append((name, self.env.now, value))

    def link(self, src: str, dst: str, q: str):
        """Cria uma relação O2O (ex.: PO —supplier→ SUP-001)."""
        if self.record_ocel and src in self.objects:
            self.objects[src]["o2o"].append((dst, q))

    def emit(self, act: str, objs: dict, attrs: dict | None = None):
        """Registra UM evento com todos os objetos envolvidos (relações E2O).
        ``objs`` = dicionário {id do objeto: qualificador}, ex.:
        {"POI-S000012": "purchase order item", "SUP-001": "supplier", "IL-0043": ...}.
        O contador de eventos por atividade é sempre atualizado; a lista completa
        de eventos só é guardada quando o OCEL simulado foi pedido."""
        self.counts[act] += 1
        if self.record_ocel:
            self._eid += 1
            self.events.append((f"SIM-E-{self._eid:07d}", act, self.env.now, objs, attrs or {}))

    def il_objs(self, st: ILState) -> dict:
        """Objetos estruturais presentes em todo evento de um IL (IL, material, local)."""
        return {st.p.il: "inventory perspective", st.p.material: "material", st.p.location: "location"}

    def il_attrs(self, st: ILState, qty: float = 0.0, extra: dict | None = None) -> dict:
        """Atributos de estado gravados em cada evento (mesmos nomes do OCEL original)."""
        if not self.record_ocel:
            return {}
        a = {
            "quantity": float(qty),
            "on_hand_after": st.on_hand,
            "reserved_after": st.reserved,
            "backorder_after": st.backorder,
            "on_order_after": st.on_order,
            "inventory_position_after": st.ip_log,
            "lower_threshold": st.pol.s,
            "upper_threshold": st.pol.S,
        }
        if extra:
            a.update(extra)
        return a

    def day(self) -> int:
        """Dia corrente da simulação (índice inteiro)."""
        return min(int(self.env.now), self.ndays)

    def planner_for(self, act: str):
        """Sorteia o planejador que executa a atividade (distribuição E2O do log)."""
        dist = self.P.planner_by_activity.get(act)
        return dist.sample(self.rng) if dist is not None else None

    def sample_attr(self, otype: str, name: str, default=None):
        """Sorteia o valor de um atributo de objeto a partir da distribuição do log."""
        dist = self.P.attr_dists.get(otype, {}).get(name)
        return dist.sample(self.rng, "empirical" if dist.kind == "empirical" else self.mode) if dist else default

    # =======================================================================
    # Montagem inicial: objetos estruturais e estado de cada IL
    # =======================================================================
    def _setup_objects(self):
        P = self.P
        # Location, Material, Supplier e Planner são copiados do log
        for otype, df in P.static_objects.items():
            for oid, row in df.iterrows():
                self.add_object(oid, otype, {k: _py(v) for k, v in row.items() if _notna(v)})
        # ItemLocations: cria o estado de estoque e a política de cada um
        for il, ip in P.item_locations.items():
            lt_mean = ip.lead_time_mean * self.cfg.lead_time_factor
            pol = make_policy_state(ip, self.cfg, lt_mean, ip.lead_time_std * self.cfg.lead_time_factor)
            st = ILState(p=ip, pol=pol, on_hand=ip.initial_on_hand, asis_S=ip.S)
            # simulação de curto prazo: parte do estado no fim do log
            if self.cs and il in self.cs["ils"]:
                c = self.cs["ils"][il]
                st.on_hand = max(0.0, c.get("on_hand_after", st.on_hand))
                if self.cfg.policy == "ATUAL":
                    st.asis_S = c.get("upper_threshold", st.asis_S)
                    pol.S = st.asis_S
                if self.inherit_backorders and c.get("backorder_after", 0) > 0:
                    st.backorder = c["backorder_after"]
                    st.backlog.append(["BACKLOG-LOG", st.backorder])
            self.state[il] = st
            self.add_object(il, "ItemLocation", {
                "material_class": ip.material_class, "location_class": ip.location_class,
                "on_hand": st.on_hand, "reserved": 0.0, "backorder": 0.0, "on_order": 0.0,
                "inventory_position": st.on_hand, "lower_threshold": pol.s, "upper_threshold": pol.S,
                "demand_estimate": pol.forecast, "lead_time_estimate": pol.L,
                "policy_version": self.cfg.policy, "data_complete": True,
            })
            # relações O2O do IL (iguais às do log)
            self.link(il, ip.location, "location")
            self.link(il, ip.material, "material")

    # =======================================================================
    # Execução
    # =======================================================================
    def run(self):
        """Inicia todos os processos e roda a simulação até o horizonte H."""
        env = self.env
        for il in self.ils:
            st = self.state[il]
            self.emit("Initialize Inventory", self.il_objs(st), self.il_attrs(st, st.on_hand, {"cause_code": "INITIALIZATION"}))
            env.process(self.demand_generator(st))   # chegada de pedidos de venda
            env.process(self.review_process(st))     # revisão de estoque (política)
        if self.cs:
            # curto prazo: POs que já estavam em trânsito no fim do log
            for po in self.cs["open_pos"]:
                if po["il"] in self.state:
                    env.process(self.open_po(self.state[po["il"]], po))
        env.process(self.daily_snapshot())
        if self.include_exceptions:
            env.process(self.exceptions_process())
        env.run(until=self.H)
        for il in self.ils:
            st = self.state[il]
            self.emit("Simulation End Snapshot", self.il_objs(st), self.il_attrs(st, 0, {"cause_code": "HORIZON_END"}))
        return self

    # =======================================================================
    # DEMANDA (SalesOrderItem)
    # =======================================================================
    def demand_factor(self, t: float, st: ILState) -> float:
        """Multiplicador da quantidade pedida no instante t:
        fator de sensibilidade × tendência × sazonalidade × dia da semana × dia do mês."""
        f = self.cfg.demand_factor
        if self.demand_model == "tendencia_sazonal":
            p = st.p
            # tendência linear centrada no meio do horizonte do log (nunca abaixo de 10%)
            f *= max(0.1, 1.0 + p.trend_per_day * (t + self.t0 - self.P.horizon_days / 2))
            dd = min(int(t), self.ndays)
            f *= p.season_index[self.month_of_day[dd] - 1]
            f *= self.P.calendar_dow[self.dow_of_day[dd]] * self.P.calendar_dom[self.dom_of_day[dd] - 1]
        return f

    def demand_generator(self, st: ILState):
        """PROCESSO: chegada de pedidos de venda de um IL.

        Como no log, há no máximo 1 pedido por IL por dia. A cada dia:
        sorteia se haverá pedido (prob. p_order); se sim, sorteia a hora
        (~08h) e a quantidade, e inicia o ciclo de vida do pedido."""
        env = self.env
        tod = self.P.time_of_day["Sales Order Item Created"]
        day = 0
        while day < self.ndays:
            if self.bern(st.p.p_order):
                t = day + float(np.clip(tod.sample(self.rng, "empirical"), 0, 0.999))
                if t > env.now:
                    yield env.timeout(t - env.now)     # pausa este processo até a hora do pedido
                if env.now >= self.H:
                    break
                q = st.p.qty.sample(self.rng, self.mode)
                q = max(1, int(round(q * self.demand_factor(env.now, st))))
                # cria um NOVO processo só para este pedido de venda e segue em frente:
                # o gerador de demanda não espera o pedido terminar
                env.process(self.sales_order(st, float(q)))
            day += 1
            if day > env.now:
                yield env.timeout(day - env.now)       # pausa até a meia-noite do próximo dia

    def sales_order(self, st: ILState, qty: float):
        """PROCESSO: ciclo de vida de um SalesOrderItem.

        Created → Reservation Created → Goods Issue          (havia estoque)
        Created → Backorder Registered                        (sem estoque)
        Created → Reservation → Goods Issue → Backorder       (atendimento parcial)"""
        env = self.env
        # --- cria o objeto e registra a demanda
        soi = self.new_oid("SalesOrderItem", "SOI")
        self.n_so += 1
        self.total_demand += qty
        self.demand[self.il_index[st.p.il], self.day()] += qty
        st.consumed_today += qty
        due = self.sample_attr("SalesOrderItem", "due_time", 1.0)
        self.add_object(soi, "SalesOrderItem", {
            "requested_quantity": qty,
            "customer_class": self.sample_attr("SalesOrderItem", "customer_class", "stochastic"),
            "priority": self.sample_attr("SalesOrderItem", "priority", "standard"),
            "due_time": env.now + float(due),
        })
        self.link(soi, st.p.il, "consumes-from")       # O2O: SOI consome do IL
        objs = {soi: "sales order item", **self.il_objs(st)}
        self.emit("Sales Order Item Created", objs, self.il_attrs(st, qty, {"cause_code": "STOCHASTIC_DEMAND"}))
        variant = ["Created"]

        # --- quanto dá para atender agora?
        avail = max(0.0, st.available)
        take = min(avail, qty)    # parte atendida
        short = qty - take        # parte que vira backorder

        if take > 0:
            st.reserved += take   # reserva imediatamente (evita que outro pedido use o mesmo estoque)
            yield env.timeout(self.d("so_created_to_reservation"))
            self.emit("Reservation Created", objs, self.il_attrs(st, take))
            yield env.timeout(self.d("so_reservation_to_issue"))
            st.reserved -= take
            st.on_hand -= take    # saída física do estoque
            self.served_immediately += take
            self.emit("Goods Issue", objs, self.il_attrs(st, take))
            variant += ["Reservation", "Goods Issue"]
        if short > 0:
            yield env.timeout(self.d("so_issue_to_backorder" if take > 0 else "so_created_to_backorder"))
            st.backlog.append([soi, short])   # entra na fila de pendências
            st.backorder += short
            self.so_with_backorder += 1
            self.emit("Backorder Registered", objs, self.il_attrs(st, short, {"cause_code": "STOCHASTIC_SHORTAGE"}))
            variant.append("Backorder")
        self.so_variants[" → ".join(variant)] += 1

        # na política (s,Q) a revisão é contínua: verifica após cada pedido
        if self.cfg.policy == "SQ":
            self.check_continuous(st)

    # =======================================================================
    # POLÍTICAS DE REPOSIÇÃO (revisão de estoque)
    # =======================================================================
    def review_process(self, st: ILState):
        """PROCESSO: revisão diária de estoque de um IL (por volta das 17–18h).

        Todo dia atualiza a previsão com a demanda do dia. Depois, conforme a
        política, decide se pede e quanto. Durante uma lacuna de dados
        (data_gap) a revisão não é feita."""
        env = self.env
        cfg = self.cfg
        tod = self.P.time_of_day.get("Replenishment Proposal Created")
        # intervalo de revisão: diário para Atual, L4L e (s,Q); R dias para RS e RSS
        R = 1 if cfg.policy in ("ATUAL", "L4L", "SQ") else max(1, int(cfg.review_period))  # RS, RSS e POUT: R dias
        # revisões escalonadas: cada IL revisa num dia diferente do ciclo de R dias
        offset = (self.il_index[st.p.il] % R) if (cfg.staggered_reviews and R > 1) else 0
        day = 0
        while day < self.ndays:
            hour = float(np.clip(tod.sample(self.rng, "empirical"), 0, 0.999)) if tod else REVIEW_HOUR
            t = day + hour
            if t > env.now:
                yield env.timeout(t - env.now)
            # atualiza a previsão com a demanda observada no dia
            st.pol.observe_day(st.consumed_today)
            st.consumed_today = 0.0
            if not st.data_gap:
                if cfg.policy == "ATUAL":
                    # mín–máx do log: se IP ≤ s, pede até S (com a prob. calibrada)
                    if st.ip_log <= st.pol.s and self.bern(self.P.asis_review_prob):
                        q = math.ceil(st.asis_S - st.ip_log)
                        if q > 0:
                            self.place_order(st, float(q), "periodic review")
                elif cfg.policy == "SQ":
                    self.check_continuous(st)
                elif cfg.policy == "L4L":
                    # repõe até o base-stock fixo = repõe o que foi consumido
                    q = st.pol.round_qty(st.pol.S - st.ip)
                    if q > 0:
                        self.place_order(st, q, "lot for lot")
                elif (day - offset) % R == 0:   # é dia de revisão deste IL?
                    if cfg.policy == "RS":
                        # sempre pede até S
                        q = st.pol.round_qty(st.pol.S - st.ip)
                        if q > 0:
                            self.place_order(st, q, "periodic review")
                    elif cfg.policy == "POUT":
                        # pede F·R + α·(S − IP − F·R): fecha só a fração α da diferença
                        q = st.pol.round_qty(st.pol.pout_qty(st.ip))
                        if q > 0:
                            self.place_order(st, q, "periodic review")
                    elif cfg.policy == "RSS":
                        # só pede se estiver abaixo do ponto de pedido s
                        if st.ip <= st.pol.s:
                            q = st.pol.round_qty(st.pol.S - st.ip)
                            if q > 0:
                                self.place_order(st, q, "periodic review")
            day += 1
            if day > env.now:
                yield env.timeout(day - env.now)

    def check_continuous(self, st: ILState):
        """Política (s,Q): se IP ≤ s, pede n·Q, com n suficiente para IP voltar acima de s."""
        if st.data_gap:
            return
        pol = st.pol
        if st.ip <= pol.s:
            n = math.floor((pol.s - st.ip) / pol.Q) + 1
            self.place_order(st, float(n * pol.Q), "reorder point")

    def place_order(self, st: ILState, qty: float, reason: str):
        """Registra a quantidade como pendente e inicia o processo de reposição."""
        st.pending += qty
        self.env.process(self.replenishment(st, qty, reason))

    # =======================================================================
    # REPOSIÇÃO: ReplenishmentProposal → PurchaseOrderItem (+ Delivery) → recebimento
    # =======================================================================
    def replenishment(self, st: ILState, qty: float, reason: str):
        """PROCESSO: ciclo de vida de uma reposição, envolvendo vários objetos."""
        env = self.env
        P = self.P

        # --- 1) proposta de reposição (ReplenishmentProposal)
        rp = self.new_oid("ReplenishmentProposal", "RP")
        self.add_object(rp, "ReplenishmentProposal", {
            "suggested_quantity": qty, "reason": reason, "creation_time": env.now, "status": "created"})
        self.link(rp, st.p.il, "replenishes")
        planner = self.planner_for("Replenishment Proposal Created")
        objs = {rp: "proposal", **self.il_objs(st)}
        if planner:
            objs[planner] = "planner"
        self.emit("Replenishment Proposal Created", objs, self.il_attrs(st, qty, {"cause_code": "PERIODIC_REVIEW"}))

        # --- 2) aprovação (tempo sorteado; ~2% das propostas não são aprovadas)
        yield env.timeout(self.d("rp_created_to_approved"))
        if not self.bern(P.probs.get("rp_approved", 1.0)):
            st.pending -= qty  # proposta não aprovada (fica 'created', como no log)
            if self.cfg.policy == "SQ":
                self.check_continuous(st)
            return
        self.set_attr(rp, "status", "approved")
        self.emit("Replenishment Proposal Approved", objs, self.il_attrs(st, qty))
        yield env.timeout(self.d("rp_approved_to_po"))

        # --- 3) pedido de compra (PurchaseOrderItem) e fornecedor
        sup = _choice(self.rng, st.p.supplier_probs)   # fornecedor ~ P(fornecedor | IL)
        po = self.new_oid("PurchaseOrderItem", "POI")
        self.n_po += 1
        st.pending -= qty
        st.on_order += qty
        di = self.il_index[st.p.il]
        dday = self.day()
        self.orders[di, dday] += qty                     # ← série usada no efeito chicote
        arr = self.orders_by_supplier.setdefault(sup, np.zeros(self.ndays + 1))
        arr[dday] += qty
        # lead time do fornecedor (PO criado → recebimento)
        lt_dist = P.supplier_lead_time.get(sup, P.supplier_lead_time["__all__"])
        # dispersão (k) aplicada em torno da média da distribuição; depois o fator de média
        lt = spread_lead_time(float(lt_dist.sample(self.rng, self.mode)), lt_dist,
                              self.cfg.lead_time_spread) * self.cfg.lead_time_factor
        created = env.now
        self.add_object(po, "PurchaseOrderItem", {
            "ordered_quantity": qty, "confirmed_quantity": qty,
            "planned_receipt_time": created + st.pol.L, "status": "open"})
        self.link(po, st.p.il, "replenishes")
        self.link(po, sup, "supplier")

        # --- 4) entrega (Delivery): existe para ~95% dos POs (cardinalidade 0..1)
        delivery = None
        if self.bern(P.probs.get("po_has_delivery", 1.0)):
            delivery = self.new_oid("Delivery", "DEL")
            self.add_object(delivery, "Delivery", {
                "carrier": self.sample_attr("Delivery", "carrier", "SIM"),
                "delay_code": "none", "shipment_status": "in transit"})
        po_objs = {po: "purchase order item", sup: "supplier", **self.il_objs(st)}
        pl = self.planner_for("Purchase Order Item Created")
        if pl and self.bern(0.02):  # no log, o planejador aparece em ~1% dos eventos de PO
            po_objs[pl] = "planner"
        self.emit("Purchase Order Item Created", po_objs, self.il_attrs(st, qty, {"cause_code": "STOCHASTIC_ORDER"}))
        variant = ["Created"]
        rx_objs = dict(po_objs)          # objetos do evento de recebimento (inclui a Delivery)
        if delivery:
            rx_objs[delivery] = "delivery"

        # --- 5) confirmação do fornecedor
        if self.bern(P.probs.get("po_confirmation", 1.0)):
            yield env.timeout(self.d("po_created_to_confirmation"))
            self.emit("Supplier Confirmation Received", po_objs, self.il_attrs(st, qty))
            variant.append("Confirmation")

        # --- 6) caminhos alternativos sorteados com as frequências do log
        rejected = self.bern(P.probs.get("po_rejected", 0.0))      # ~3% rejeitados
        if self.bern(P.probs.get("po_delayed", 0.0)):               # ~1% atrasados
            t1 = self.d("po_confirmation_to_delayed")
            yield env.timeout(t1)
            self.emit("Delivery Delayed", rx_objs, self.il_attrs(st, qty))
            if delivery:
                self.set_attr(delivery, "shipment_status", "delayed")
                self.set_attr(delivery, "delay_code", self.sample_attr("Delivery", "delay_code", "capacity"))
            t2 = self.d("po_delayed_to_expedite")
            yield env.timeout(t2)
            self.emit("Expedite Requested", rx_objs, self.il_attrs(st, qty))
            variant += ["Delayed", "Expedite"]

        # --- 7a) recebimento rejeitado: a quantidade se perde
        if rejected:
            rest = self.d("po_confirmation_to_rejected") - (env.now - created)
            yield env.timeout(max(0.0, rest))
            st.on_order -= qty
            self.set_attr(po, "status", "rejected")
            self.emit("Receipt Rejected", rx_objs, self.il_attrs(st, qty, {"cause_code": "QUALITY_REJECTION"}))
            variant.append("Rejected")
            self.po_variants[" → ".join(variant)] += 1
            if self.cfg.policy == "SQ":
                self.check_continuous(st)
            return

        # --- 7b) recebimento normal após o lead time
        rest = lt - (env.now - created)       # tempo restante do lead time
        yield env.timeout(max(0.0, rest))
        st.on_order -= qty
        # recebimento parcial: chega uma fração do pedido (média ≈ 84% no log)
        rr = P.receipt_ratio.sample(self.rng, "empirical") if (P.receipt_ratio is not None and self.cfg.partial_receipts) else 1.0
        rq = float(max(0.0, qty * rr))
        st.on_hand += rq
        self.receipts[di, self.day()] += rq
        self.lead_times.append(env.now - created)
        self.on_stock_increase(st, rq)        # atende backorders
        self.set_attr(po, "status", "received")
        self.set_attr(po, "actual_receipt_time", env.now)
        if delivery:
            self.set_attr(delivery, "shipment_status", "received")
        self.set_attr(po, "confirmed_quantity", rq)
        # evento compartilhado: PO + Delivery + Supplier + IL + Material + Location
        self.emit("Goods Receipt", rx_objs, self.il_attrs(st, rq, {"cause_code": "STOCHASTIC_RECEIPT"}))
        variant.append("Goods Receipt")
        self.po_variants[" → ".join(variant)] += 1

    # =======================================================================
    # Tratamento dos backorders quando entra estoque
    # =======================================================================
    def on_stock_increase(self, st: ILState, qty: float):
        """Escolhe a contabilidade de backorder do cenário."""
        if self.cfg.backorder_accounting == "log":
            self.clear_backlog_log_style(st, qty)
        else:
            self.serve_backlog(st)

    def clear_backlog_log_style(self, st: ILState, qty: float):
        """Contabilidade observada NO LOG: o recebimento abate o backorder sem
        consumir o estoque recebido. Usada só para validar o cenário Atual."""
        rem = min(qty, st.backorder)
        st.backorder -= rem
        while st.backlog and rem > 1e-9:
            item = st.backlog[0]
            take = min(item[1], rem)
            item[1] -= take
            rem -= take
            if item[1] <= 1e-9:
                st.backlog.popleft()
        if st.backorder < 1e-9:
            st.backorder = 0.0

    def serve_backlog(self, st: ILState):
        """Contabilidade FÍSICA correta: as pendências mais antigas (FIFO) são
        atendidas com o estoque disponível, que é consumido."""
        while st.backlog and st.available > 0:
            item = st.backlog[0]
            take = min(item[1], st.available)
            st.on_hand -= take
            st.backorder -= take
            item[1] -= take
            if item[1] <= 1e-9:
                st.backlog.popleft()
        if st.backorder < 1e-9:
            st.backorder = 0.0

    # =======================================================================
    # Curto prazo: POs em aberto herdados do log
    # =======================================================================
    def open_po(self, st: ILState, po: dict):
        """PROCESSO: PO já emitido no log e ainda não recebido.

        O lead time restante é sorteado da distribuição do fornecedor
        CONDICIONADA a LT > idade do pedido (se o pedido já tem 10 dias,
        só valem lead times maiores que 10)."""
        P = self.P
        qty = po["qty"]
        st.on_order += qty
        dist = P.supplier_lead_time.get(po.get("supplier"), P.supplier_lead_time["__all__"])
        age = po["age"]
        lt = None
        # "amostragem por rejeição": sorteia lead times até sair um maior que a
        # idade do pedido (no máximo 200 tentativas)
        for _ in range(200):
            x = spread_lead_time(float(dist.sample(self.rng, self.mode)), dist, self.cfg.lead_time_spread)
            if x > age:
                lt = x
                break
        rest = (lt - age) if lt is not None else float(self.rng.exponential(2.0))
        yield self.env.timeout(rest * self.cfg.lead_time_factor)
        st.on_order -= qty
        if self.bern(P.probs.get("po_rejected", 0.0)):
            self.emit("Receipt Rejected", {po["po"]: "purchase order item", **self.il_objs(st)}, self.il_attrs(st, qty))
            return
        rr = P.receipt_ratio.sample(self.rng, "empirical") if (P.receipt_ratio is not None and self.cfg.partial_receipts) else 1.0
        rq = float(max(0.0, qty * rr))
        st.on_hand += rq
        self.receipts[self.il_index[st.p.il], self.day()] += rq
        self.on_stock_increase(st, rq)
        self.emit("Goods Receipt", {po["po"]: "purchase order item", **self.il_objs(st)},
                  self.il_attrs(st, rq, {"cause_code": "OPEN_ORDER_FROM_LOG"}))

    # =======================================================================
    # Fotografia diária do estoque
    # =======================================================================
    def daily_snapshot(self):
        """PROCESSO: no fim de cada dia (23:59) grava estoque, backorder e IP de cada IL."""
        env = self.env
        for dd in range(self.ndays):
            t = dd + 0.99999   # 23h59min59s do dia dd (o dia termina em dd + 1)
            if t > env.now:
                yield env.timeout(t - env.now)
            for i, il in enumerate(self.ils):
                st = self.state[il]
                self.on_hand_eod[i, dd] = st.on_hand
                self.backorder_eod[i, dd] = st.backorder
                self.ip_eod[i, dd] = st.ip

    # =======================================================================
    # Eventos de exceção (raros)
    # =======================================================================
    def exceptions_process(self):
        """PROCESSO: eventos raros em toda a rede.

        Os eventos raros acontecem "ao acaso no tempo": não há horário fixo, e
        um não influencia o outro. Esse padrão é o processo de Poisson; nele, o
        tempo entre um evento e o próximo segue a distribuição exponencial.
        A taxa total é a soma das taxas de cada exceção × nº de ILs (ex.: ~0,05
        evento por dia na rede inteira). A cada chegada, sorteia qual exceção
        ocorre (com probabilidade proporcional à taxa) e em qual IL."""
        env = self.env
        rates = self.P.exception_rates
        n = len(self.ils)
        total = sum(rates.values()) * n
        if total <= 0:
            return
        acts = list(rates)
        p = np.array([rates[a] for a in acts]) * n / total
        while True:
            yield env.timeout(self.rng.exponential(1 / total))
            if env.now >= self.H:
                return
            act = acts[int(self.rng.choice(len(acts), p=p))]
            st = self.state[self.ils[int(self.rng.integers(n))]]
            if act == "Inventory Adjustment":
                # ajuste de inventário (positivo ou negativo, sem deixar o estoque abaixo do reservado)
                adj = float(self.P.exception_dists["adjustment_qty"].sample(self.rng, "empirical"))
                adj = max(adj, -st.on_hand + st.reserved)
                st.on_hand += adj
                pl = self.planner_for(act)
                self.emit(act, {**self.il_objs(st), **({pl: "auditor"} if pl else {})},
                          self.il_attrs(st, adj, {"cause_code": "COUNT_CORRECTION"}))
                if adj > 0:
                    self.on_stock_increase(st, adj)
            elif act == "Cycle Count Performed":
                self.emit(act, self.il_objs(st), self.il_attrs(st, 0, {"cause_code": "PATTERN_NOISE"}))
            elif act == "Transfer Requested":
                env.process(self.transfer(st))
            elif act == "Policy Threshold Updated":
                # alteração do nível máximo (afeta só a política Atual, que usa os limites do log)
                new_S = float(self.P.exception_dists["threshold_upper"].sample(self.rng, "empirical"))
                if self.cfg.policy == "ATUAL":
                    st.asis_S = new_S
                    st.pol.S = new_S
                self.emit(act, self.il_objs(st), self.il_attrs(st, 0, {"cause_code": "CAPACITY_ALERT"}))
            elif act == "Data Gap Started":
                env.process(self.data_gap(st))

    def transfer(self, src: ILState):
        """PROCESSO: TransferOrder — Requested → Ship (sai da origem) → Receive (entra no destino)."""
        env = self.env
        # destino: IL com a menor posição de estoque (exceto a própria origem)
        cand = [s for s in self.state.values() if s is not src]
        tgt = min(cand, key=lambda s: s.ip)
        qty = float(self.P.exception_dists["transfer_qty"].sample(self.rng, "empirical"))
        to = self.new_oid("TransferOrder", "TO")
        self.add_object(to, "TransferOrder", {"quantity": qty, "source_item_location": src.p.il,
                                              "target_item_location": tgt.p.il, "status": "requested"})
        self.link(to, src.p.il, "source")
        self.link(to, tgt.p.il, "target")
        self.emit("Transfer Requested", {to: "transfer order", **self.il_objs(src)}, self.il_attrs(src, 0))
        yield env.timeout(self.d("to_requested_to_ship"))
        qty = min(qty, max(src.available, 0.0))   # não envia mais do que há disponível
        src.on_hand -= qty
        self.emit("Transfer Ship", {to: "transfer order", **self.il_objs(src)}, self.il_attrs(src, qty))
        yield env.timeout(self.d("to_ship_to_receive"))
        tgt.on_hand += qty
        self.on_stock_increase(tgt, qty)
        self.set_attr(to, "status", "received")
        self.emit("Transfer Receive", {to: "transfer order", **self.il_objs(tgt)}, self.il_attrs(tgt, qty))

    def data_gap(self, st: ILState):
        """PROCESSO: lacuna de dados — durante a lacuna o IL não faz revisões."""
        st.data_gap = True
        self.emit("Data Gap Started", self.il_objs(st), self.il_attrs(st, 0, {"data_complete": False}))
        yield self.env.timeout(self.d("data_gap_duration"))
        st.data_gap = False
        self.emit("Data Gap Ended", self.il_objs(st), self.il_attrs(st, 0, {"data_complete": True}))


# ===========================================================================
# Utilitários
# ===========================================================================
def _choice(rng, probs: dict):
    """Sorteia uma chave de um dicionário {opção: probabilidade}."""
    keys = list(probs)
    if len(keys) == 1:
        return keys[0]
    p = np.array([probs[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _td(t: float):
    """Converte dias em pd.Timedelta."""
    return pd.Timedelta(days=float(t))


def _notna(v) -> bool:
    """True se o valor não é ausente (None/NaN)."""
    try:
        return not (v is None or (isinstance(v, float) and math.isnan(v)))
    except Exception:
        return True


def _py(v):
    """Converte tipos do numpy em tipos nativos do Python (para gravar o JSON)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v
