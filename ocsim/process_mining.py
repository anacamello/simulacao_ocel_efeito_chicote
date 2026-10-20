"""
process_mining.py — Etapas 1 e 2: leitura do OCEL e mineração de processos (PM4Py)
==================================================================================

PAPEL NO PROJETO
----------------
É a "porta de entrada" dos dados. Lê o arquivo OCEL 2.0 (JSON) com o PM4Py e
o transforma em tabelas pandas fáceis de consultar. Depois oferece as funções
de mineração de processos centrada em objetos usadas pela interface e pelo
módulo ``parameters.py``.

Seguindo o OCPS (Knopp & van der Aalst), do OCEL extraímos três coisas:
  (i)   o MODELO DE OBJETOS – tipos, atributos e cardinalidades entre tipos
        (relações objeto-a-objeto, O2O, e evento-a-objeto, E2O);
  (ii)  o MODELO DE CONTROLE DE FLUXO – OC-DFG e Petri net centrada em
        objetos (OCPN), além das variantes por tipo de objeto;
  (iii) as ESTATÍSTICAS TEMPORAIS – tempo entre atividades consecutivas de
        cada objeto, tempo de vida dos objetos e nº de ocorrências de cada
        atividade por objeto.

GLOSSÁRIO RÁPIDO
----------------
* E2O (event-to-object): "este evento envolve estes objetos".
* O2O (object-to-object): "este PurchaseOrderItem é fornecido por este Supplier".
* Achatamento (flattening): escolher UM tipo de objeto como "caso" e gerar um
  log tradicional (caso, atividade, timestamp) – necessário para usar as
  técnicas clássicas (DFG, variantes, desempenho).

Termos de programação (DataFrame, groupby, merge...) estão explicados no
glossário do arquivo ocsim/__init__.py.
"""
from __future__ import annotations

import io
import os
import contextlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _quiet():
    """Silencia as mensagens que o PM4Py escreve no terminal.

    Toda vez que é carregado, o PM4Py escreve no terminal um aviso de várias
    linhas sobre a licença de uso dele (AGPL). Esse aviso não tem utilidade para
    o projeto e polui a tela. Esta função cria um "desvio" temporário: tudo o
    que for escrito no terminal (com print) enquanto o desvio estiver ativo vai
    para uma área de texto descartável, em vez de aparecer na tela.

    Uso:   with _quiet():
               ...código do PM4Py...      ← mensagens escritas aqui são descartadas
    Ao sair do bloco ``with``, o terminal volta ao normal."""
    return contextlib.redirect_stdout(io.StringIO())


class _LazyPM4Py:
    """Carrega o PM4Py somente na primeira vez em que ele é realmente usado.

    Por que isso existe: carregar (importar) o PM4Py leva vários segundos. A
    simulação em si não usa o PM4Py — só a leitura do OCEL e a mineração usam.
    Quando a simulação roda em vários núcleos ao mesmo tempo no Windows ou no
    macOS, cada núcleo começa um Python do zero e carregaria o PM4Py à toa,
    atrasando (ou até travando) o início.

    Como funciona: ``pm4py`` abaixo não é a biblioteca, e sim um
    "intermediário". Na primeira vez que o código pede algo dele (por exemplo
    ``pm4py.read_ocel2_json``), o intermediário carrega a biblioteca de verdade
    e repassa o pedido; nas vezes seguintes, só repassa."""
    _mod = None

    def __getattr__(self, name):
        # chamado automaticamente quando o código pede pm4py.<algo>
        if _LazyPM4Py._mod is None:          # primeira vez: carrega a biblioteca
            with _quiet():
                import pm4py as _m  # noqa: E402
            _LazyPM4Py._mod = _m
        return getattr(_LazyPM4Py._mod, name)


pm4py = _LazyPM4Py()  # use "pm4py.funcao(...)" normalmente no resto do arquivo


# Ordem de exibição dos tipos de objeto (transacionais primeiro, estruturais depois)
OT_ORDER = [
    "SalesOrderItem",
    "PurchaseOrderItem",
    "ReplenishmentProposal",
    "Delivery",
    "ItemLocation",
    "Material",
    "TransferOrder",
    "Location",
    "Planner",
    "Supplier",
]

# O PM4Py entrega as tabelas do OCEL com colunas de nomes longos ("ocel:eid"...).
# Guardamos esses nomes em variáveis curtas para o código ficar mais legível.
EID, ACT, TS, OID, OTYPE, QUAL = (
    "ocel:eid",        # id do evento
    "ocel:activity",   # nome da atividade
    "ocel:timestamp",  # data/hora do evento
    "ocel:oid",        # id do objeto
    "ocel:type",       # tipo do objeto
    "ocel:qualifier",  # qualificador da relação (ex.: "supplier", "replenishes")
)


# ===========================================================================
# Estrutura que guarda o OCEL já convertido em tabelas
# ===========================================================================
@dataclass
class OcelData:
    """Ficha com o conteúdo do OCEL já convertido em tabelas (DataFrames do pandas).

    É o que este módulo devolve depois de ler o arquivo, e o que os outros
    módulos recebem para trabalhar. Cada campo abaixo é uma das tabelas."""

    ocel: object                  # objeto OCEL original do PM4Py (usado nas funções de descoberta)
    events: pd.DataFrame          # 1 linha por evento: atividade, timestamp, atributos e o IL envolvido
    relations: pd.DataFrame       # relações E2O: 1 linha por par (evento, objeto)
    objects: pd.DataFrame         # 1 linha por objeto com o valor inicial de cada atributo
    object_changes: pd.DataFrame  # mudanças de atributos ao longo do tempo
    o2o: pd.DataFrame             # relações O2O com o tipo de origem e de destino
    start: pd.Timestamp           # início do horizonte (1º dia)
    end: pd.Timestamp             # fim do horizonte

    @property  # permite escrever d.horizon_days (sem parênteses), como se fosse um campo
    def horizon_days(self) -> float:
        """Duração do log em dias (731 no arquivo fornecido)."""
        return (self.end - self.start).total_seconds() / 86400


def load_ocel(path: str) -> OcelData:
    """Lê o OCEL 2.0 (JSON) e monta as tabelas de trabalho."""
    # 1) leitura com o PM4Py
    with _quiet():
        ocel = pm4py.read_ocel2_json(path)
    ev = ocel.events.copy()
    rel = ocel.relations.copy()
    ev[TS] = pd.to_datetime(ev[TS], utc=True)
    rel[TS] = pd.to_datetime(rel[TS], utc=True)

    # 2) quase todo evento envolve exatamente um ItemLocation (IL = item × local).
    #    Guardamos o IL em uma coluna 'il' da tabela de eventos para facilitar
    #    as análises por IL (demanda, estoque, pedidos).
    #    Passo a passo: pega as relações evento→objeto cujo objeto é um ItemLocation,
    #    fica com uma por evento e junta (merge, como um PROCV) à tabela de eventos.
    il = rel[rel[OTYPE] == "ItemLocation"][[EID, OID]].drop_duplicates(EID).rename(columns={OID: "il"})
    ev = ev.merge(il, on=EID, how="left").sort_values(TS).reset_index(drop=True)

    # 3) objetos, mudanças de atributos e relações O2O
    objs = ocel.objects.copy()
    oc = ocel.object_changes.copy() if ocel.object_changes is not None else pd.DataFrame()
    o2o = ocel.o2o.copy()
    # a tabela O2O só traz os ids dos dois objetos; aqui buscamos (merge) o tipo de
    # cada um, para saber, por exemplo, que a ligação é PurchaseOrderItem → Supplier
    o2o = o2o.merge(objs[[OID, OTYPE]], on=OID).merge(
        objs[[OID, OTYPE]].rename(columns={OID: OID + "_2", OTYPE: OTYPE + "_2"}), on=OID + "_2"
    )

    # 4) horizonte: a inicialização do estoque acontece às 23:58 de 31/12/2022;
    #    nesse caso o horizonte começa no dia seguinte (01/01/2023)
    start = ev[TS].min().normalize() + pd.Timedelta(days=1) if ev[TS].min().hour >= 23 else ev[TS].min().normalize()
    end = ev[TS].max().normalize()
    return OcelData(ocel, ev, rel, objs, oc, o2o, start, end)


# ===========================================================================
# Etapa 1 — resumos do log
# ===========================================================================
def object_type_summary(d: OcelData) -> pd.DataFrame:
    """Para cada tipo de objeto: nº de objetos, nº de eventos e atividades em que aparece."""
    total = d.objects.groupby(OTYPE)[OID].nunique()
    linked = d.relations.groupby(OTYPE)[OID].nunique()
    nev = d.relations.groupby(OTYPE)[EID].nunique()
    with _quiet():
        # dicionário {tipo de objeto: conjunto de atividades em que ele aparece}
        ot_acts = pm4py.ocel_object_type_activities(d.ocel)
    df = pd.DataFrame({"objetos (total)": total, "objetos em eventos": linked, "eventos relacionados": nev})
    df["atividades"] = [", ".join(sorted(ot_acts.get(t, []))) for t in df.index]
    df["nº atividades"] = [len(ot_acts.get(t, [])) for t in df.index]
    return df.reindex([t for t in OT_ORDER if t in df.index]).fillna(0)


def activity_summary(d: OcelData) -> pd.DataFrame:
    """Matriz atividade × tipo de objeto com o nº de relações E2O."""
    cnt = d.events[ACT].value_counts().rename("eventos")   # nº de eventos de cada atividade
    # groupby + count: conta as relações por (atividade, tipo); unstack gira os tipos para colunas
    m = d.relations.groupby([ACT, OTYPE])[OID].count().unstack(fill_value=0)
    return pd.concat([cnt, m], axis=1).fillna(0).astype(int).sort_values("eventos", ascending=False)


def object_type_attributes(d: OcelData) -> dict[str, list[str]]:
    """Lista dos atributos (colunas não vazias) de cada tipo de objeto."""
    out = {}
    for t, g in d.objects.groupby(OTYPE):
        cols = [c for c in g.columns if c not in (OID, OTYPE) and g[c].notna().any()]
        out[t] = cols
    return out


def static_attributes(d: OcelData, otype: str) -> pd.DataFrame:
    """Tabela de atributos dos objetos de um tipo (índice = id do objeto)."""
    g = d.objects[d.objects[OTYPE] == otype].set_index(OID)
    cols = [c for c in g.columns if c != OTYPE and g[c].notna().any()]
    return g[cols]


# ===========================================================================
# Etapa 2 — descoberta de modelos (OC-DFG e Petri net centrada em objetos)
# ===========================================================================
def filter_object_types(d: OcelData, otypes: list[str]) -> OcelData:
    """Mantém apenas alguns tipos de objeto (deixa os diagramas legíveis)."""
    with _quiet():
        o = pm4py.filter_ocel_object_types(d.ocel, otypes)
    return OcelData(o, d.events, d.relations, d.objects, d.object_changes, d.o2o, d.start, d.end)


def discover_ocdfg(d: OcelData):
    """OC-DFG: um grafo de atividades diretamente consecutivas para cada tipo de objeto."""
    with _quiet():
        return pm4py.discover_ocdfg(d.ocel)


def save_ocdfg_svg(d: OcelData, path: str, annotation: str = "frequency", act_threshold: int = 0,
                   edge_threshold: int = 0, performance_aggregation: str = "mean") -> str:
    """Descobre o OC-DFG e salva a imagem SVG (via Graphviz).
    ``annotation``: 'frequency' (nº de ocorrências) ou 'performance' (tempo médio)."""
    with _quiet():
        ocdfg = pm4py.discover_ocdfg(d.ocel)
        pm4py.save_vis_ocdfg(
            ocdfg,
            path,
            annotation=annotation,
            act_threshold=act_threshold,
            edge_threshold=edge_threshold,
            performance_aggregation=performance_aggregation,
            rankdir="LR",  # desenho da esquerda para a direita
        )
    return path


def save_ocpn_svg(d: OcelData, path: str) -> str:
    """Descobre a Petri net centrada em objetos (OCPN) e salva o SVG."""
    with _quiet():
        ocpn = pm4py.discover_oc_petri_net(d.ocel)
        pm4py.save_vis_ocpn(ocpn, path, rankdir="LR")
    return path


# ===========================================================================
# Achatamento por tipo de objeto e estatísticas derivadas
# ===========================================================================
def flatten(d: OcelData, otype: str) -> pd.DataFrame:
    """Gera o log tradicional usando ``otype`` como caso.

    Ex.: com 'PurchaseOrderItem', cada pedido de compra vira um caso e seus
    eventos (Created, Confirmation, Goods Receipt...) formam o "traço"."""
    with _quiet():
        fl = pm4py.ocel_flattening(d.ocel, otype)
    # renomeia para nomes curtos: case, act, ts
    fl = fl.rename(columns={"case:concept:name": "case", "concept:name": "act", "time:timestamp": "ts"})
    fl["ts"] = pd.to_datetime(fl["ts"], utc=True)
    return fl.sort_values(["case", "ts"]).reset_index(drop=True)


def variants(fl: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Variantes = sequências distintas de atividades, com frequência e %."""
    v = fl.groupby("case")["act"].apply(lambda s: " → ".join(s)).value_counts()
    df = v.rename("objetos").to_frame()
    df["%"] = df["objetos"] / df["objetos"].sum() * 100
    return df.head(top)


def dfg_durations(fl: pd.DataFrame) -> pd.DataFrame:
    """Amostras de tempo (em dias) entre atividades DIRETAMENTE consecutivas de cada objeto.

    É daqui que saem as observações usadas para ajustar as distribuições de
    duração (ex.: todos os tempos 'Supplier Confirmation Received → Goods Receipt')."""
    f = fl[["case", "act", "ts"]].copy()
    # shift(-1) dentro de cada caso: coloca, na mesma linha, o evento atual e o
    # PRÓXIMO evento do mesmo objeto. Ex.: na linha de "PO Created" aparecem
    # também "Supplier Confirmation" e o horário dela — daí sai a duração.
    f["next_act"] = f.groupby("case")["act"].shift(-1)
    f["next_ts"] = f.groupby("case")["ts"].shift(-1)
    f = f.dropna(subset=["next_act"])  # o último evento não tem sucessor
    f["dias"] = (f["next_ts"] - f["ts"]).dt.total_seconds() / 86400
    return f[["case", "act", "next_act", "dias"]]


def performance_dfg_table(fl: pd.DataFrame) -> pd.DataFrame:
    """DFG de desempenho do PM4Py como tabela: frequência e tempos (horas) de cada arco."""
    log = fl.rename(columns={"case": "case:concept:name", "act": "concept:name", "ts": "time:timestamp"})
    with _quiet():
        # DFG = grafo "diretamente seguido por": quais atividades vêm logo após quais.
        perf, sa, ea = pm4py.discover_performance_dfg(log)  # tempos de cada arco, em segundos
        freq, _, _ = pm4py.discover_dfg(log)                # quantas vezes cada arco ocorre
    rows = []
    for (a, b), st in perf.items():
        rows.append(
            {
                "de": a,
                "para": b,
                "frequência": freq.get((a, b), 0),
                "média (h)": st.get("mean", np.nan) / 3600,
                "mediana (h)": st.get("median", np.nan) / 3600,
                "mín (h)": st.get("min", np.nan) / 3600,
                "máx (h)": st.get("max", np.nan) / 3600,
            }
        )
    return pd.DataFrame(rows).sort_values("frequência", ascending=False)


def object_lifetimes(fl: pd.DataFrame) -> pd.Series:
    """Tempo de vida (dias) de cada objeto = último evento − primeiro evento."""
    g = fl.groupby("case")["ts"]
    return (g.max() - g.min()).dt.total_seconds() / 86400


def activity_counts_per_object(d: OcelData, otype: str) -> pd.DataFrame:
    """Matriz objeto × atividade com o nº de ocorrências.

    Inclui objetos com zero ocorrências (importante para estimar
    probabilidades: ex.: % de POs que têm 'Receipt Rejected')."""
    r = d.relations[d.relations[OTYPE] == otype]
    # conta eventos por (objeto, atividade) e gira as atividades para colunas
    m = r.groupby([OID, ACT])[EID].nunique().unstack(fill_value=0)
    # objetos que nunca tiveram uma atividade não aparecem na contagem;
    # reindex inclui todos os objetos do tipo, com 0 onde faltar
    all_objs = d.objects.loc[d.objects[OTYPE] == otype, OID]
    return m.reindex(all_objs, fill_value=0)


# ===========================================================================
# Modelo de objetos: cardinalidades
# ===========================================================================
def o2o_cardinalities(d: OcelData) -> pd.DataFrame:
    """Cardinalidades O2O no mesmo formato do diagrama.

    Para cada relação (origem —qualificador→ destino) conta quantos objetos
    destino cada origem tem, e vice-versa. Ex.: SalesOrderItem —consumes-from→
    ItemLocation: cada SOI tem exatamente 1 IL; cada IL tem de 4 a 711 SOIs."""
    rows = []
    o2o = d.o2o
    for (ta, tb, q), g in o2o.groupby([OTYPE, OTYPE + "_2", QUAL]):
        n_a = (d.objects[OTYPE] == ta).sum()
        n_b = (d.objects[OTYPE] == tb).sum()
        per_a = g.groupby(OID)[OID + "_2"].nunique()      # nº de destinos por origem
        per_b = g.groupby(OID + "_2")[OID].nunique()      # nº de origens por destino
        # objetos sem nenhuma ligação não aparecem na contagem; reindex os inclui com 0
        a_full = per_a.reindex(d.objects.loc[d.objects[OTYPE] == ta, OID], fill_value=0)
        b_full = per_b.reindex(d.objects.loc[d.objects[OTYPE] == tb, OID], fill_value=0)
        rows.append(
            {
                "origem": ta,
                "qualificador": q,
                "destino": tb,
                "ligações": len(g),
                f"min..max {tb} por {ta}": f"{a_full.min()}..{a_full.max()}",
                f"min..max {ta} por {tb}": f"{b_full.min()}..{b_full.max()}",
                "cardinalidade": f"{_card(b_full)} : {_card(a_full)}",
                "média destino/origem": a_full.mean(),
                "média origem/destino": b_full.mean(),
                "_per_a": a_full,
                "_per_b": b_full,
                "n_origem": n_a,
                "n_destino": n_b,
            }
        )
    return pd.DataFrame(rows)


def _card(s: pd.Series) -> str:
    """Converte (mín, máx) em notação de cardinalidade: 0:1, 1:1, 1:n, 0:n..."""
    lo, hi = int(s.min()), int(s.max())
    return f"{lo}:{'n' if hi > 1 else hi}"


def e2o_cardinalities(d: OcelData) -> pd.DataFrame:
    """Nº médio de objetos de cada tipo envolvidos em um evento de cada atividade."""
    r = d.relations.groupby([EID, ACT, OTYPE])[OID].nunique().unstack(fill_value=0)
    return r.groupby(level=1).mean()


def object_interactions(d: OcelData) -> pd.DataFrame:
    """Co-ocorrência de objetos em eventos (grafo de tipos de objetos do OCPS).

    Dois objetos 'interagem' quando aparecem juntos em algum evento. Para cada
    par de tipos mostra quantos objetos do outro tipo cada objeto encontra."""
    r = d.relations[[EID, OID, OTYPE]].drop_duplicates()
    # junta a tabela com ela mesma pelo id do evento: cada linha passa a ser um
    # PAR de objetos que participaram do mesmo evento
    m = r.merge(r, on=EID, suffixes=("", "_b"))
    m = m[m[OTYPE] != m[OTYPE + "_b"]]                 # só pares de tipos diferentes
    pairs = m[[OID, OTYPE, OID + "_b", OTYPE + "_b"]].drop_duplicates()
    per = pairs.groupby([OTYPE, OTYPE + "_b", OID])[OID + "_b"].nunique()
    return per.groupby(level=[0, 1]).agg(["mean", "min", "max", "count"]).reset_index()


# ===========================================================================
# Qualidade do achatamento (recomendação de van der Aalst)
# ===========================================================================
def flattening_diagnostics(d: OcelData) -> pd.DataFrame:
    """Mede as três distorções do achatamento para cada tipo usado como caso:

    * DEFICIÊNCIA  – eventos que não envolvem nenhum objeto do tipo somem do log achatado;
    * CONVERGÊNCIA – evento ligado a 2+ objetos do tipo é replicado (infla contagens/tempos);
    * DIVERGÊNCIA  – a mesma atividade se repete no mesmo caso (por envolver objetos
      de outro tipo) e os eventos ficam indistinguíveis.
    """
    rel = d.relations
    n_ev = d.events[EID].nunique()
    rows = []
    for ot in [t for t in OT_ORDER if t in set(rel[OTYPE])]:
        r = rel[rel[OTYPE] == ot]
        per_ev = r.groupby(EID)[OID].nunique()      # objetos do tipo por evento
        covered = len(per_ev)                        # eventos que permanecem
        replicated = int((per_ev > 1).sum())         # eventos que seriam replicados
        dup = r.groupby([OID, ACT])[EID].nunique()   # repetições de (objeto, atividade)
        rows.append({
            "tipo de objeto (caso)": ot,
            "eventos no log achatado": int(per_ev.sum()),
            "deficiência: eventos perdidos": n_ev - covered,
            "deficiência %": (n_ev - covered) / n_ev * 100,
            "convergência: eventos replicados": replicated,
            "convergência %": replicated / max(covered, 1) * 100,
            "divergência: pares (objeto, atividade) repetidos %": float((dup > 1).mean() * 100),
        })
    return pd.DataFrame(rows)
