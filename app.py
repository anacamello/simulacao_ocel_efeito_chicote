"""
app.py — interface Streamlit (passo a passo)
============================================

Executar:  streamlit run app.py

PAPEL NO PROJETO
----------------
É a "vitrine" do trabalho. NÃO faz cálculos próprios: chama os módulos do
pacote ``ocsim`` e mostra os resultados em 11 páginas (etapas 0 a 10), na
ordem do método:

  0 Visão geral → 1 Dados OCEL → 2 Mineração → 3 Distribuições → 4 Cenários →
  5 Simulação → 6 Validação → 7 Visão unificada → 8 Comparação →
  9 Impacto entre objetos → 10 Exportar

COMO O STREAMLIT FUNCIONA (resumo)
-----------------------------------
O Streamlit transforma um arquivo Python numa página web: cada comando
``st.algumacoisa`` desenha um elemento na tela (``st.title`` = título,
``st.dataframe`` = tabela, ``st.button`` = botão, ``st.plotly_chart`` = gráfico).

* O arquivo inteiro é executado de cima para baixo A CADA interação do usuário
  (clique, troca de opção...). É como se a página fosse redesenhada do zero.
* Por isso, cálculos pesados (ler o OCEL de 60 MB, ajustar distribuições)
  usam CACHE (ver glossário em ocsim/__init__.py). As linhas
  ``@st.cache_resource`` e ``@st.cache_data`` colocadas logo acima de uma
  função dizem ao Streamlit: "guarde o resultado; se a função for chamada de
  novo com os mesmos valores, devolva o resultado guardado em vez de
  recalcular".
* Como tudo é reexecutado a cada clique, as variáveis comuns "esquecem" seus
  valores. Para lembrar algo entre cliques (os cenários escolhidos, os
  resultados da simulação) usamos ``st.session_state``, um dicionário que o
  Streamlit mantém enquanto a aba do navegador estiver aberta.
* ``st.stop()`` interrompe a execução naquele ponto (o resto da página não é
  desenhado). É usado, por exemplo, enquanto nenhum arquivo foi enviado.

Termos de programação estão no glossário de ocsim/__init__.py.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ocsim import experiment as X
from ocsim import impact as IMP
from ocsim import process_mining as pm
from ocsim.parameters import STATE_ATTRIBUTES, extract_parameters
from ocsim.policies import POLICY_LABELS, POLICY_SHORT, ScenarioConfig, make_policy_state
import dataclasses
from ocsim.ui_charts import (LOG_COLOR, MUTED, SERIES, bar_ci, color_for, hist_with_fit, style)

# ---------------------------------------------------------------------------
# Caminhos e configuração da página
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent                 # pasta do projeto
CACHE = ROOT / "cache"                       # SVGs e OCELs simulados gerados pelo app
CACHE.mkdir(exist_ok=True)

st.set_page_config(page_title="Simulação OCEL · Efeito Chicote", page_icon="🔗", layout="wide")

# pequenos ajustes visuais (CSS)
st.markdown(
    """<style>
    .block-container {padding-top: 1.6rem; max-width: 1400px;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .step {color:#52514e; font-size:0.85rem; text-transform:uppercase; letter-spacing:.06em; margin-bottom:-.6rem}
    .note {background:#f1f0ec; border-left:3px solid #2a78d6; padding:.7rem 1rem; border-radius:4px; font-size:.92rem}
    </style>""",
    unsafe_allow_html=True,
)


# =============================================================== cache
# Funções "caras" (ler o OCEL, ajustar distribuições, desenhar diagramas) são
# executadas uma única vez por arquivo e reaproveitadas nas próximas interações.
# Cada função recebe, além do caminho do arquivo, uma "chave" (key) que
# identifica o arquivo; se a chave muda, o Streamlit sabe que precisa recalcular.
def _file_hash(path: str) -> str:
    """Cria a "chave" de identificação de um arquivo.

    Junta o caminho, o tamanho e a data de modificação do arquivo e passa por
    uma função de hash (MD5). Um hash é como uma "impressão digital": transforma
    qualquer texto num código curto (ex.: '3f9a1c0b7e21'); textos diferentes
    geram códigos diferentes. Se o arquivo mudar, a chave muda e o cache é refeito."""
    h = hashlib.md5()
    st_ = os.stat(path)
    h.update(f"{path}|{st_.st_size}|{st_.st_mtime}".encode())
    return h.hexdigest()[:12]


@st.cache_resource(show_spinner="Lendo o OCEL com PM4Py…")
def load_data(path: str, key: str):
    """Etapa 1: lê o OCEL (process_mining.load_ocel)."""
    return pm.load_ocel(path)


@st.cache_resource(show_spinner="Extraindo parâmetros e ajustando distribuições…")
def load_params(path: str, key: str):
    """Etapa 3: extrai todos os parâmetros e distribuições (parameters.extract_parameters)."""
    return extract_parameters(load_data(path, key))


@st.cache_resource(show_spinner="Calculando medidas do log original…")
def load_log_baseline(path: str, key: str, warmup: int, bucket: int):
    """Mesmas métricas da simulação calculadas sobre o log original (referência)."""
    return X.log_baseline(load_data(path, key), load_params(path, key), warmup_days=warmup, bucket_days=bucket)


@st.cache_data(show_spinner="Descobrindo o modelo com PM4Py (pode levar alguns segundos)…")
def svg_file(path: str, key: str, kind: str, annotation: str = "frequency", otypes: tuple = ()) -> str:
    """Etapa 2: gera (uma vez) e devolve o SVG do OC-DFG ou da Petri net,
    opcionalmente filtrando alguns tipos de objeto."""
    tag = hashlib.md5("|".join(otypes).encode()).hexdigest()[:6]
    out = CACHE / f"{key}_{kind}_{annotation}_{tag}.svg"
    if not out.exists():
        d = load_data(path, key)
        if otypes:
            d = pm.filter_object_types(d, list(otypes))
        if kind == "ocdfg":
            pm.save_ocdfg_svg(d, str(out), annotation=annotation)
        else:
            pm.save_ocpn_svg(d, str(out))
    return out.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def flat_cached(path: str, key: str, otype: str):
    """Log achatado por tipo de objeto (com cache)."""
    return pm.flatten(load_data(path, key), otype)


@st.cache_data(show_spinner="Calculando o impacto entre objetos…")
def impact_cached(path: str, key: str):
    """Etapa 9: análises de impacto entre objetos (impact.py) para um OCEL."""
    d = load_data(path, key)
    a = IMP.supplier_to_customer(d)
    a.pop("per_so", None)  # tabela grande, não é exibida
    return {"fornecedor_cliente": a, "reposicao_falta": IMP.replenishment_to_stockout(d)}


def simulated_ocel_path(rr: dict) -> tuple[str, str]:
    """Grava (uma vez) o OCEL simulado de uma replicação em cache/ e devolve (caminho, chave)."""
    sim_key = hashlib.md5(rr["ocel_json"][:2000].encode() + str(len(rr["ocel_json"])).encode()).hexdigest()[:12]
    sim_path = CACHE / f"sim_{sim_key}.json"
    if not sim_path.exists():
        sim_path.write_text(rr["ocel_json"])
    return str(sim_path), sim_key


def show_svg(svg: str, height: int = 600):
    """Exibe um diagrama SVG (formato de imagem vetorial gerado pelo Graphviz)
    numa caixa com barra de rolagem."""
    # O Graphviz grava no SVG uma largura e altura fixas (ex.: width="2400pt"),
    # o que faria o diagrama vazar da tela. Esta linha procura esse trecho no
    # texto do SVG (re.sub = "procurar e substituir") e troca por 100%, para o
    # desenho se ajustar à largura disponível sem distorcer.
    svg = re.sub(r'<svg width="[^"]+" height="[^"]+"', '<svg width="100%" height="100%"', svg, count=1)
    st.components.v1.html(
        f'<div style="overflow:auto;height:{height - 10}px;border:1px solid #e6e5e0;border-radius:6px;'
        f'background:#fff">{svg}</div>', height=height)


# =============================================================== barra lateral
# O arquivo OCEL é SEMPRE enviado pelo usuário (não há arquivo padrão).
# Enquanto nenhum arquivo for enviado, o app mostra apenas as instruções.
st.sidebar.markdown("### 🔗 Simulação OCEL\n**Efeito chicote na cadeia de suprimentos**")
up = st.sidebar.file_uploader("Arquivo OCEL 2.0 (JSON)", type=["json", "jsonocel"],
                              help="Envie o log centrado em objetos no formato OCEL 2.0 JSON.")

if up is None:
    # nenhum arquivo enviado: limpa resultados antigos e interrompe o app aqui
    for k in ("results", "opts", "run_scenarios", "ocel_key"):
        st.session_state.pop(k, None)
    st.title("Simulação centrada em objetos — efeito chicote")
    st.info("⬅️ **Envie o arquivo OCEL** (formato OCEL 2.0 JSON) na barra lateral para começar.")
    st.markdown(
        "Depois do envio, o app lê o log com o PM4Py, ajusta as distribuições e libera as etapas:\n\n"
        "1. Dados OCEL · 2. Mineração de processos · 3. Distribuições · 4. Cenários de reposição · "
        "5. Simulação · 6. Validação · 7. Visão unificada · 8. Comparação de cenários · 9. Impacto entre objetos · "
        "10. Exportar\n\n"
        "A primeira leitura de um arquivo grande (≈60 MB) leva de 10 a 20 segundos; depois fica em cache.")
    st.stop()

# O arquivo enviado chega à memória do Streamlit; o PM4Py, porém, só lê arquivos
# gravados no disco. Por isso ele é salvo na pasta cache/.
# O nome do arquivo salvo vem de um hash ("impressão digital") do conteúdo —
# calculado só sobre o 1º e o último megabyte, para ser rápido. Assim, enviar
# de novo o MESMO arquivo reaproveita o que já foi calculado, e um arquivo
# DIFERENTE gera um novo processamento.
_buf = up.getbuffer()  # conteúdo do arquivo enviado, em bytes
_digest = hashlib.md5(bytes(_buf[:1_000_000]) + bytes(_buf[-1_000_000:]) + str(up.size).encode()).hexdigest()[:12]
tmp = CACHE / f"upload_{_digest}.json"
if not tmp.exists() or tmp.stat().st_size != up.size:
    tmp.write_bytes(_buf)
path = str(tmp)
KEY = _file_hash(path)

# se o usuário trocou de arquivo, descarta os resultados de simulação do arquivo anterior
if st.session_state.get("ocel_key") != KEY:
    for k in ("results", "opts", "run_scenarios", "scenarios"):
        st.session_state.pop(k, None)
    st.session_state["ocel_key"] = KEY

# lista de páginas (etapas); a escolhida fica em `page`

PAGES = [
    "0 · Visão geral",
    "1 · Dados OCEL",
    "2 · Mineração de processos",
    "3 · Distribuições",
    "4 · Cenários de reposição",
    "5 · Simulação",
    "6 · Validação (Atual × log)",
    "7 · Visão unificada (log × simulado)",
    "8 · Comparação de cenários",
    "9 · Impacto entre objetos",
    "10 · Exportar",
]
page = st.sidebar.radio("Etapas", PAGES, index=0)
st.sidebar.caption(f"Arquivo carregado: `{up.name}`")
if "results" in st.session_state:
    st.sidebar.success(f"{len(st.session_state['results'])} replicações em memória")

# carrega (ou recupera do cache) o OCEL e os parâmetros — usados por todas as páginas
D = load_data(path, KEY)   # tabelas do OCEL (process_mining.OcelData)
P = load_params(path, KEY)  # parâmetros da simulação (parameters.ModelParameters)


def header(step: str, title: str, text: str = ""):
    """Cabeçalho padrão de cada página: nº da etapa, título e explicação."""
    st.markdown(f'<div class="step">{step}</div>', unsafe_allow_html=True)
    st.title(title)
    if text:
        st.markdown(text)


# ===================================================================== 0
# ETAPA 0 — Visão geral: números do log, fluxo do método, recomendações do
# artigo de van der Aalst e o diagrama do modelo de objetos.
if page == PAGES[0]:
    header("Etapa 0", "Visão geral do método",
           "Simulação de eventos discretos **centrada em objetos** de uma cadeia de suprimentos, construída a "
           "partir de um log OCEL, para analisar o **efeito chicote** sob diferentes regras de reposição.")
    c = st.columns(5)
    c[0].metric("Eventos", f"{len(D.events):,}".replace(",", "."))
    c[1].metric("Objetos", f"{D.objects.shape[0]:,}".replace(",", "."))
    c[2].metric("Tipos de objeto", D.objects[pm.OTYPE].nunique())
    c[3].metric("Atividades", D.events[pm.ACT].nunique())
    c[4].metric("Horizonte (dias)", f"{D.horizon_days:.0f}")

    st.subheader("Fluxo de trabalho")
    steps = [
        ("1 · Dados OCEL", "Leitura do OCEL 2.0 (PM4Py): tipos de objetos, atributos, atividades, relações E2O e O2O."),
        ("2 · Mineração", "OC-DFG e Petri net centrada em objetos; achatamento por tipo de objeto, variantes e desempenho."),
        ("3 · Distribuições", "Ajuste (MLE + AIC, teste KS) das durações, quantidades, atributos, nº de ocorrências "
                              "de atividades por objeto, cardinalidades e demanda por ItemLocation."),
        ("4 · Cenários", "Regras de reposição: Atual (mín–máx), (s,Q), (R,S), (R,s,S) e Lote a Lote."),
        ("5 · Simulação", "Modelo SimPy centrado em objetos (OCPS): processos por tipo de objeto, eventos "
                          "compartilhados, N replicações por cenário com números aleatórios comuns."),
        ("6 · Validação", "O cenário Atual é comparado ao log (frequências, variantes, lead time, efeito chicote)."),
        ("8 · Comparação", "Efeito chicote (razão de variâncias) por IL, Location, fornecedor e rede, "
                           "nível de serviço e custos, com intervalos de confiança."),
        ("9 · Impacto", "Como um tipo de objeto afeta outro: fornecedor → cliente, reposição → falta, "
                        "demanda → pedidos, lead time → efeito chicote."),
    ]
    cols = st.columns(len(steps))
    for col, (t, txt) in zip(cols, steps):
        col.markdown(f"**{t}**")
        col.caption(txt)

    # diagrama da arquitetura do projeto (assets/fluxo_projeto.png, gerado com Graphviz a partir de assets/fluxo.dot)
    with st.expander("Arquitetura do projeto: módulos e fluxo dos dados", expanded=False):
        st.image(str(ROOT / "assets" / "fluxo_projeto.png"), use_container_width=True)

    st.subheader("Recomendações de van der Aalst (2023) e onde estão no app")
    st.markdown(
        "| Recomendação do artigo | Implementação |\n|---|---|\n"
        "| Descobrir o modelo de simulação a partir dos dados (controle de fluxo, dados, recursos, tempo) | Etapas 2–3: OC-DFG/OCPN, durações, atributos, planejadores e demanda ajustados do OCEL |\n"
        "| Usar dados centrados em objetos (E2O/O2O) em vez de um único caso | Modelo de objetos com 10 tipos, cardinalidades O2O e eventos compartilhados (etapa 1 e simulador) |\n"
        "| Achatamento gera deficiência, convergência e divergência | Diagnóstico por tipo de objeto (etapa 2, aba *Qualidade do achatamento*) |\n"
        "| Chegadas de objetos não são independentes; restringir o comportamento “subespecificado” da OCPN | Objetos transacionais nascem de outros objetos (SOI→IL; RP/PO pelo estado do IL), respeitando as cardinalidades |\n"
        "| Primeiro reproduzir o processo do log; só depois perguntar “e se?” | Etapa 6: cenário Atual × log |\n"
        "| Visão unificada: minerar o log simulado como o original (as-is × to-be) | Etapa 7: OCEL simulado lido pelo PM4Py, lado a lado com o original |\n"
        "| Simulação de curto prazo a partir do estado atual (análise transitória) | Etapa 5: estado inicial = fim do log (estoques e POs em aberto) |")
    st.subheader("Modelo de objetos")
    st.image(str(ROOT / "assets" / "modelo_objetos.png"), use_container_width=True,
             caption="Modelo de objetos do processo (tipos, atributos, atividades e cardinalidades).")
    st.markdown(
        '<div class="note"><b>Referências do método.</b> Knopp, B.; van der Aalst, W. — <i>Object-Centric Process '
        'Simulation</i> (OCPS, repositório beneknopp/OCPS); van der Aalst, W. — <i>Toward More Realistic Simulation '
        'Models Using Object-Centric Process Mining</i> (2023): geração do modelo de objetos a partir das '
        'cardinalidades do log, descoberta do modelo de controle de fluxo por tipo de objeto, previsão de '
        'atividades/atrasos a partir do log e avaliação do OCEL simulado contra o original. '
        'Efeito chicote: Wang &amp; Disney (2016); Dejonckheere et al. (2003).</div>',
        unsafe_allow_html=True)

# ===================================================================== 1
# ETAPA 1 — Dados OCEL: tipos de objeto, matriz atividade × tipo, relações
# O2O com cardinalidades e atributos de cada tipo.
elif page == PAGES[1]:
    header("Etapa 1", "Dados OCEL", "O que existe no log: objetos, atividades e como eles se relacionam.")
    st.subheader("Tipos de objeto")
    ots = pm.object_type_summary(D)
    st.dataframe(ots, use_container_width=True)

    st.subheader("Atividades × tipos de objeto (nº de relações E2O)")
    st.caption("Cada célula indica quantas vezes objetos daquele tipo participam da atividade.")
    st.dataframe(pm.activity_summary(D), use_container_width=True, height=520)

    st.subheader("Modelo de objetos — relações O2O e cardinalidades")
    cards = P.cardinalities
    show = cards[["origem", "qualificador", "destino", "ligações", "cardinalidade",
                  "média destino/origem", "média origem/destino"]]
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("Cardinalidade lida como no diagrama: (destino por origem) : (origem por destino). "
               "Ex.: SalesOrderItem —consumes-from→ ItemLocation: cada SOI consome de 1 IL; cada IL é "
               "consumido por n SOIs.")
    with st.expander("Co-ocorrência de objetos em eventos (grafo de tipos de objetos)"):
        st.dataframe(pm.object_interactions(D), use_container_width=True, hide_index=True)

    st.subheader("Atributos por tipo de objeto")
    attrs = pm.object_type_attributes(D)
    ot = st.selectbox("Tipo de objeto", [t for t in pm.OT_ORDER if t in attrs])
    st.dataframe(pm.static_attributes(D, ot).head(200), use_container_width=True)

# ===================================================================== 2
# ETAPA 2 — Mineração de processos com PM4Py: OC-DFG, Petri net centrada em
# objetos, variantes/desempenho por tipo e qualidade do achatamento.
elif page == PAGES[2]:
    header("Etapa 2", "Mineração de processos centrada em objetos (PM4Py)",
           "Descoberta do comportamento de cada tipo de objeto e de como os tipos se sincronizam nos eventos.")
    t1, t2, t3, t4 = st.tabs(["OC-DFG", "Petri net centrada em objetos", "Por tipo de objeto",
                              "Qualidade do achatamento"])
    # aba 4: deficiência / convergência / divergência do achatamento
    with t4:
        st.markdown(
            "Para medir durações usamos o log **achatado** por tipo de objeto (cada tipo vira a noção de caso). "
            "Van der Aalst aponta três distorções do achatamento: **deficiência** (eventos sem objeto do tipo "
            "somem), **convergência** (um evento ligado a vários objetos do tipo é replicado) e **divergência** "
            "(a mesma atividade se repete no caso por se referir a objetos de outro tipo, tornando os eventos "
            "indistinguíveis). A tabela mostra quanto cada uma afeta este log.")
        st.dataframe(pm.flattening_diagnostics(D).round(2), use_container_width=True, hide_index=True)
        st.caption("Convergência nula: nenhum evento envolve dois objetos do mesmo tipo, então as durações "
                   "por tipo não são infladas por replicação. As durações da simulação vêm dos tipos "
                   "transacionais (SalesOrderItem, PurchaseOrderItem, ReplenishmentProposal, TransferOrder), "
                   "com divergência ≈ 0. Para ItemLocation (divergência alta, pois cada IL participa de muitos "
                   "pedidos) só usamos pares bem definidos (Approved → PO Created, Data Gap Started → Ended).")
    # aba 1: OC-DFG (frequência ou desempenho), com filtro de tipos de objeto
    with t1:
        all_ot = [t for t in pm.OT_ORDER if t in set(D.objects[pm.OTYPE])]
        c1, c2 = st.columns([1, 2])
        ann = c1.radio("Anotação", ["frequency", "performance"], horizontal=True,
                       format_func=lambda x: {"frequency": "frequência", "performance": "desempenho (tempo médio)"}[x])
        sel_ot = c2.multiselect("Tipos de objeto exibidos", all_ot,
                                default=["SalesOrderItem", "ReplenishmentProposal", "PurchaseOrderItem", "TransferOrder"])
        svg = svg_file(path, KEY, "ocdfg", ann, tuple(sorted(sel_ot)) if sel_ot and len(sel_ot) < len(all_ot) else ())
        show_svg(svg, 700)
        st.download_button("Baixar OC-DFG (SVG)", svg.encode("utf-8"), "ocdfg.svg", "image/svg+xml")
        st.caption("Cada cor é um tipo de objeto; os arcos mostram a sequência de atividades por tipo.")
    # aba 2: Petri net centrada em objetos (gerada sob demanda, pois é mais lenta)
    with t2:
        if st.button("Gerar Petri net centrada em objetos (≈10 s)"):
            st.session_state["show_ocpn"] = True
        if st.session_state.get("show_ocpn"):
            sel2 = st.multiselect("Tipos de objeto", [t for t in pm.OT_ORDER if t in set(D.objects[pm.OTYPE])],
                                  default=["SalesOrderItem", "ReplenishmentProposal", "PurchaseOrderItem"], key="ocpn_ot")
            show_svg(svg_file(path, KEY, "ocpn", "", tuple(sorted(sel2))), 700)
    # aba 3: log achatado por um tipo → variantes, tempo de vida e DFG de desempenho
    with t3:
        ot = st.selectbox("Tipo de objeto", [t for t in pm.OT_ORDER if t in set(D.objects[pm.OTYPE])], index=0)
        fl = flat_cached(path, KEY, ot)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("**Variantes (log achatado por tipo)**")
            st.dataframe(pm.variants(fl), use_container_width=True)
        with c2:
            lt = pm.object_lifetimes(fl)
            fig = go.Figure(go.Histogram(x=lt, nbinsx=40, marker=dict(color=SERIES[0], line=dict(color="white", width=1))))
            st.plotly_chart(style(fig, 300, f"Tempo de vida dos objetos {ot}", xtitle="dias", ytitle="objetos",
                                  legend=False), use_container_width=True)
        st.markdown("**DFG de desempenho (tempos entre atividades consecutivas do mesmo objeto)**")
        st.dataframe(pm.performance_dfg_table(fl), use_container_width=True, hide_index=True)

# ===================================================================== 3
# ETAPA 3 — Distribuições: mostra tudo o que foi ajustado em parameters.py
# (cada aba corresponde a um grupo de parâmetros).
elif page == PAGES[3]:
    header("Etapa 3", "Distribuições calculadas a partir do log",
           "Para cada variável numérica testamos normal, exponencial, gama, lognormal, Weibull e uniforme "
           "(máxima verossimilhança) e escolhemos a de **menor AIC**; o teste de **Kolmogorov–Smirnov** mede a "
           "aderência. Com menos de 15 observações usamos a distribuição **empírica** (reamostragem). "
           "Variáveis categóricas usam as frequências observadas.")
    tabs = st.tabs(["Durações entre atividades", "Demanda por ItemLocation", "Atributos dos objetos",
                    "Ocorrências de atividades por objeto", "Tempo de vida dos objetos",
                    "Modelo de objetos & probabilidades", "Notas"])
    # durações entre atividades + histograma com a curva ajustada + lead time por fornecedor
    with tabs[0]:
        dt = P.delay_table()
        dt_show = dt.copy()
        dt_show["média (h)"] = dt_show["média"] * 24
        st.dataframe(dt_show[["parâmetro", "tipo de objeto", "de", "para", "distribuição", "n", "média (h)",
                              "KS", "KS p-valor"]], use_container_width=True, hide_index=True)
        k = st.selectbox("Ver ajuste", list(P.delays))
        dist = P.delays[k]
        if dist.values is not None and len(dist.values) > 1:
            st.plotly_chart(hist_with_fit(dist.values, dist, f"{k}: observado × ajustado", "dias"),
                            use_container_width=True)
            if dist.ranking:
                st.dataframe(pd.DataFrame(dist.ranking, columns=["distribuição", "AIC", "KS", "KS p-valor"]),
                             hide_index=True)
        st.markdown("**Lead time por fornecedor** (Purchase Order Item Created → Goods Receipt)")
        st.dataframe(pd.DataFrame([{"fornecedor": s, **dd.to_row()} for s, dd in P.supplier_lead_time.items()]),
                     hide_index=True, use_container_width=True)
        if P.receipt_ratio is not None:
            st.markdown(f"**Razão recebido/pedido** (atendimento parcial): {P.receipt_ratio.label()} — "
                        f"média {P.receipt_ratio.mean:.2f}")
    # demanda por ItemLocation: tabela, histograma da quantidade, sazonalidade e calendário
    with tabs[1]:
        rows = []
        for il, ip in P.item_locations.items():
            rows.append({"IL": il, "material": ip.material, "location": ip.location, "classe": ip.material_class,
                         "P(pedido/dia)": ip.p_order, "quantidade": ip.qty.label(), "demanda média/dia": ip.daily_mean,
                         "desvio/dia": ip.daily_std, "tendência %/ano": ip.trend_per_day * 365 * 100,
                         "lead time total (d)": ip.lead_time_mean, "s (log)": ip.s, "S (log)": ip.S,
                         "estoque inicial": ip.initial_on_hand, "fornecedores": ", ".join(ip.supplier_probs)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=360)
        il = st.selectbox("ItemLocation", list(P.item_locations))
        ip = P.item_locations[il]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(hist_with_fit(ip.qty.values, ip.qty, f"Quantidade pedida — {il}", "unidades", 30),
                            use_container_width=True)
        with c2:
            fig = go.Figure(go.Bar(x=["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"],
                                   y=ip.season_index, marker=dict(color=SERIES[0], cornerradius=4)))
            fig.add_hline(y=1, line=dict(color=MUTED, dash="dot"))
            st.plotly_chart(style(fig, title=f"Índice sazonal — classe {ip.material_class}", ytitle="índice",
                                  legend=False), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"], y=P.calendar_dow,
                                   marker=dict(color=SERIES[0], cornerradius=4)))
            fig.add_hline(y=1, line=dict(color=MUTED, dash="dot"))
            st.plotly_chart(style(fig, 300, "Índice por dia da semana (demanda total)", ytitle="índice", legend=False),
                            use_container_width=True)
        with c2:
            fig = go.Figure(go.Bar(x=list(range(1, 32)), y=P.calendar_dom, marker=dict(color=SERIES[0], cornerradius=3)))
            fig.add_hline(y=1, line=dict(color=MUTED, dash="dot"))
            st.plotly_chart(style(fig, 300, "Índice por dia do mês (demanda total)", ytitle="índice", legend=False),
                            use_container_width=True)
        st.caption("Modelo de demanda: a cada dia, um pedido de venda ocorre com probabilidade P(pedido/dia) "
                   "(no log há no máximo 1 pedido por IL por dia, por volta das 8 h); a quantidade segue a "
                   "distribuição ajustada do IL, multiplicada pelos índices de tendência, sazonalidade, dia da "
                   "semana e dia do mês estimados no log.")
    # atributos de cada tipo de objeto (sorteados ou calculados pelo estado)
    with tabs[2]:
        ot = st.selectbox("Tipo de objeto ", [t for t in pm.OT_ORDER if t in P.attr_dists])
        rows = []
        for a, dd in P.attr_dists[ot].items():
            r = {"atributo": a, "origem na simulação": "estado da simulação" if a in STATE_ATTRIBUTES.get(ot, set())
                 else "amostrado da distribuição", **dd.to_row()}
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Atributos temporais (due_time, planned_receipt_time…) são modelados como deslocamento em dias "
                   "a partir do primeiro evento do objeto. Atributos marcados como 'estado da simulação' são "
                   "calculados pelo modelo (ex.: on_hand, quantidades, status).")
    # nº de vezes que cada atividade ocorre em cada objeto
    with tabs[3]:
        ot = st.selectbox("Tipo de objeto  ", [t for t in pm.OT_ORDER if t in P.activity_counts])
        rows = []
        for a, dd in P.activity_counts[ot].items():
            vals = ", ".join(f"{v}: {p:.1%}" for v, p in zip(dd.values, dd.probs)) if dd.probs is not None else dd.label()
            rows.append({"atividade": a, "média por objeto": dd.mean, "distribuição P(nº ocorrências)": vals})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    # tempo de vida (primeiro → último evento) por tipo de objeto
    with tabs[4]:
        st.dataframe(pd.DataFrame([{"tipo de objeto": k, **v.to_row()} for k, v in P.lifetimes.items()]),
                     use_container_width=True, hide_index=True)
    # probabilidades de caminho, calibração da política atual, exceções e planejadores
    with tabs[5]:
        st.markdown("**Probabilidades de roteamento (frequências no log)**")
        st.dataframe(pd.DataFrame([{"evento": k, "probabilidade": v} for k, v in P.probs.items()]), hide_index=True)
        st.markdown(f"**Política atual** — probabilidade de gerar proposta quando IP ≤ s na revisão diária: "
                    f"`{P.asis_review_prob:.3f}`")
        st.markdown("**Taxas dos eventos de exceção** (por IL · dia)")
        st.dataframe(pd.DataFrame([{"atividade": k, "taxa": v, "esperado em 2 anos (48 IL)": v * 48 * P.horizon_days}
                                   for k, v in P.exception_rates.items()]), hide_index=True)
        st.markdown("**Planejador por atividade (E2O)**")
        st.dataframe(pd.DataFrame([{"atividade": k, "distribuição": v.label()} for k, v in P.planner_by_activity.items()]),
                     hide_index=True, use_container_width=True)
    # observações sobre particularidades do log
    with tabs[6]:
        for n in P.notes:
            st.markdown(f"- {n}")

# ===================================================================== 4
# ETAPA 4 — Cenários: o usuário escolhe quais políticas simular e seus
# parâmetros. O resultado é uma lista de ScenarioConfig guardada em
# st.session_state["scenarios"], lida pela etapa 5.
elif page == PAGES[4]:
    header("Etapa 4", "Cenários de reposição",
           "Cada cenário é uma regra de reposição aplicada a todos os ItemLocations. Os parâmetros de cada IL "
           "(demanda média e desvio, lead time) vêm do log; os custos e o nível de serviço podem ser ajustados.")
    st.latex(r"z=\Phi^{-1}(NS),\quad \sigma_{T}=\sqrt{T\,\sigma_d^2+\bar d^2\sigma_L^2},\quad "
             r"Q=\max\!\Big(\text{lote},\ \Big\lceil\sqrt{\tfrac{2DK}{h\,c}}\Big\rceil_{\text{lote}}\Big)")
    with st.expander("Como cada regra decide o pedido", expanded=True):
        st.markdown(
            "| Cenário | Revisão | Gatilho | Quantidade |\n|---|---|---|---|\n"
            "| **Atual (mín–máx do log)** | diária, com prob. calibrada no log | IP ≤ s (=20) | S − IP (S = 80/90) |\n"
            "| **(s, Q) Lote fixo e ponto de pedido** | contínua | IP ≤ s = d̄·L + z·σ_L | Q (EOQ) |\n"
            "| **(R, S) Revisão periódica, intervalo fixo** | a cada R dias | sempre | S − IP, S = F·(L+R) + z·σ_(L+R) |\n"
            "| **(R, s, S) Revisão periódica com ponto de pedido** | a cada R dias | IP ≤ s = F·(L+R) + z·σ_(L+R) | S − IP, S = s + Q |\n"
            "| **Lote a lote** | diária | sempre | necessidade líquida do dia (base-stock fixo ⇒ repõe exatamente o consumido) |\n\n"
            "**F** é a previsão de demanda diária: *estática* (média do log) ou *dinâmica* (média móvel / "
            "suavização exponencial, atualizada todo dia). A previsão dinâmica é um dos mecanismos clássicos do "
            "efeito chicote (Dejonckheere et al., 2003).")
    # na primeira visita, começa com os 5 cenários padrão
    if "scenarios" not in st.session_state:
        from ocsim.policies import default_scenarios
        st.session_state["scenarios"] = default_scenarios()
    scs: list[ScenarioConfig] = st.session_state["scenarios"] or [ScenarioConfig()]

    # parâmetros comuns a todos os cenários (custos, nível de serviço, sensibilidade)
    st.subheader("Parâmetros globais")
    g = st.columns(4)
    K = g[0].number_input("Custo por pedido K", 0.0, 10000.0, float(scs[0].order_cost), 10.0)
    h = g[1].number_input("Custo de estocagem h (%/ano)", 0.0, 200.0, float(scs[0].holding_rate * 100), 1.0) / 100
    b = g[2].number_input("Custo de falta (por un·dia)", 0.0, 1000.0, float(scs[0].backorder_cost), 1.0)
    use_loc = g[3].toggle("Nível de serviço da Location (log)", value=scs[0].service_level is None)
    ns = None if use_loc else st.slider("Nível de serviço alvo", 0.80, 0.995, 0.95, 0.005)
    g2 = st.columns(3)
    ltf = g2[0].slider("Fator de lead time (sensibilidade)", 0.25, 2.0, float(scs[0].lead_time_factor), 0.05)
    dmf = g2[1].slider("Fator de demanda (sensibilidade)", 0.5, 2.0, float(scs[0].demand_factor), 0.05)
    stag = g2[2].toggle("Revisões escalonadas entre ILs", value=scs[0].staggered_reviews,
                        help="Se desligado, todos os ILs revisam no mesmo dia (sincronização aumenta o chicote no fornecedor).")

    # um bloco por política: incluir ou não + parâmetros específicos (R, previsão...)
    st.subheader("Cenários")
    new = []
    for pol in ["ATUAL", "SQ", "RS", "RSS", "L4L"]:
        cur = next((s for s in scs if s.policy == pol), ScenarioConfig(policy=pol))
        with st.expander(POLICY_LABELS[pol], expanded=pol in ("RS", "RSS")):
            c = st.columns(5)
            on = c[0].checkbox("Incluir", value=any(s.policy == pol for s in scs), key=f"on_{pol}")
            R = cur.review_period
            fc = cur.forecast
            acc = cur.backorder_accounting
            if pol in ("RS", "RSS"):
                R = c[1].number_input("R (dias)", 1, 60, int(cur.review_period), key=f"R_{pol}")
            if pol in ("SQ", "RS", "RSS"):
                fc = c[2].selectbox("Previsão F", ["estatico", "media_movel", "suav_exponencial"],
                                    index=["estatico", "media_movel", "suav_exponencial"].index(cur.forecast),
                                    format_func=lambda x: {"estatico": "estática (log)", "media_movel": "média móvel",
                                                           "suav_exponencial": "suavização exponencial"}[x],
                                    key=f"F_{pol}")
            ma, al = cur.ma_window, cur.es_alpha
            if fc == "media_movel":
                ma = c[3].number_input("Janela (dias)", 2, 120, int(cur.ma_window), key=f"ma_{pol}")
            if fc == "suav_exponencial":
                al = c[3].slider("α", 0.01, 1.0, float(cur.es_alpha), key=f"al_{pol}")
            if pol == "ATUAL":
                acc = c[1].selectbox("Contabilidade de backorder", ["log", "correta"],
                                     index=["log", "correta"].index(cur.backorder_accounting),
                                     format_func=lambda x: {"log": "como no log (validação)",
                                                            "correta": "física correta"}[x], key="acc_ATUAL")
                st.caption("No log, o recebimento abate o backorder sem consumir o estoque. 'como no log' reproduz "
                           "isso para validar o simulador; 'física correta' é a contabilidade real.")
            if on:
                new.append(ScenarioConfig(policy=pol, review_period=int(R), forecast=fc, ma_window=int(ma),
                                          es_alpha=float(al), service_level=ns, order_cost=K, holding_rate=h,
                                          backorder_cost=b, lead_time_factor=ltf, demand_factor=dmf,
                                          staggered_reviews=stag, backorder_accounting=acc))
    st.session_state["scenarios"] = new   # guarda a escolha para a etapa 5

    # prévia: s, S e Q iniciais de cada IL no cenário selecionado

    st.subheader("Parâmetros resultantes por ItemLocation")
    if new:
        sel = st.selectbox("Cenário", [s.label() for s in new])
        cfg = next(s for s in new if s.label() == sel)
        rows = []
        for il, ip in P.item_locations.items():
            ps = make_policy_state(ip, cfg, ip.lead_time_mean * cfg.lead_time_factor, ip.lead_time_std * cfg.lead_time_factor)
            rows.append({"IL": il, "d̄/dia": ip.daily_mean * cfg.demand_factor, "σ/dia": ip.daily_std * cfg.demand_factor,
                         "L (dias)": ps.L, "z": ps.z, "s": ps.s, "S": ps.S, "Q (EOQ)": ps.Q})
        st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True, height=360)
        st.caption("Com previsão dinâmica, s e S mudam ao longo da simulação; a tabela mostra os valores iniciais.")

# ===================================================================== 5
# ETAPA 5 — Simulação: opções de execução (nº de replicações, horizonte,
# estado inicial...) e o botão que chama experiment.run_experiment.
# Os resultados ficam em st.session_state["results"].
elif page == PAGES[5]:
    header("Etapa 5", "Simulação (SimPy)",
           "Cada replicação simula todos os objetos e eventos da cadeia ao longo do horizonte. A replicação *r* "
           "de todos os cenários usa a mesma semente (números aleatórios comuns), o que torna a comparação "
           "entre cenários mais precisa.")
    scs = st.session_state.get("scenarios")
    if not scs:
        from ocsim.policies import default_scenarios
        scs = default_scenarios()
        st.session_state["scenarios"] = scs
    st.markdown("**Cenários selecionados:** " + " · ".join(s.label() for s in scs))
    c = st.columns(4)
    n_reps = c[0].number_input("Nº de simulações (replicações) por cenário", 1, 500, 10, 1)
    horizon = c[1].number_input("Horizonte (dias)", 60, 3650, int(round(P.horizon_days)), 30)
    init = st.radio("Estado inicial", ["inicio_log", "estado_atual"], horizontal=True,
                    format_func=lambda x: {"inicio_log": "início do log (estado estacionário / validação)",
                                           "estado_atual": "estado atual = fim do log (simulação de curto prazo)"}[x],
                    help="Curto prazo (van der Aalst): todas as replicações partem do estado conhecido no fim do log "
                         "— estoque de cada IL e POs em aberto — e exploram o futuro próximo (análise transitória).")
    inherit = False
    if init == "estado_atual":
        cs = P.current_state
        st.caption(f"Estado em {cs['date']:%d/%m/%Y}: {len(cs['open_pos'])} POs em aberto herdados do log.")
        inherit = st.toggle("Herdar backorders acumulados do log", False,
                            help="No log, o backorder acumulado é em parte artefato da contabilidade (o recebimento "
                                 "abate backorder sem consumir estoque). Desligado: parte apenas do estoque em mão e dos POs abertos.")
    warm = c[2].number_input("Aquecimento descartado (dias)", 0, 365, 0 if init == "estado_atual" else 56, 7)
    seed = c[3].number_input("Semente base", 0, 10_000_000, 42, 1)
    c = st.columns(4)
    mode = c[0].selectbox("Amostragem", ["parametric", "empirical"],
                          format_func=lambda x: {"parametric": "distribuições ajustadas",
                                                 "empirical": "empírica (reamostragem do log)"}[x])
    dm = c[1].selectbox("Modelo de demanda", ["tendencia_sazonal", "estacionario"],
                        format_func=lambda x: {"tendencia_sazonal": "com tendência, sazonalidade e calendário",
                                               "estacionario": "estacionário"}[x])
    exc = c[2].toggle("Incluir eventos de exceção", True,
                      help="Ajustes de inventário, contagens, transferências, lacunas de dados, alteração de limites.")
    # paralelismo ligado por padrão só no Linux (no Windows/macOS pode travar dentro do Streamlit)
    par = c[3].toggle("Usar vários núcleos", X.parallel_supported(),
                      help="Distribui as replicações entre os núcleos do processador. No Windows e no macOS vem "
                           "desligado; se ligado e travar, o app volta sozinho ao modo sequencial após ~90 s.")
    bucket = st.select_slider("Agregação para o efeito chicote", options=[1, 7, 14, 30], value=7,
                              format_func=lambda x: {1: "diária", 7: "semanal", 14: "quinzenal", 30: "mensal"}[x])
    rec = st.multiselect("Gerar OCEL simulado (replicação 1) para — usado nas etapas 7 (visão unificada) e 9 (impacto entre objetos)",
                         [s.label() for s in scs], default=[s.label() for s in scs if s.policy == "ATUAL"])
    if init == "estado_atual":
        st.info("No curto prazo, o efeito chicote dos cenários diferentes do Atual inclui o **efeito de transição** "
                "(os pedidos de recomposição feitos logo após a troca de política). Para comparar regimes "
                "permanentes, use o início do log com horizonte longo.")
    total = n_reps * len(scs)
    st.caption(f"Total: {total} replicações (≈1–2 s cada em um núcleo).")
    # ao clicar: monta as opções, roda todas as replicações e guarda os resultados
    if st.button("▶ Executar simulação", type="primary", use_container_width=True):
        bar = st.progress(0.0, "Iniciando…")
        opts = {"horizon_days": float(horizon), "warmup_days": int(warm), "bucket_days": int(bucket),
                "include_exceptions": exc, "sampling_mode": mode, "demand_model": dm,
                "record_ocel_scenarios": rec, "initial_state": init, "inherit_backorders": inherit}
        # "progress=lambda f, m: ..." passa uma pequena função que a simulação chama a
        # cada replicação concluída, para avançar a barra de progresso (f = fração, m = mensagem)
        res = X.run_experiment(P, scs, int(n_reps), int(seed), opts,
                               progress=lambda f, m: bar.progress(min(f, 1.0), m), parallel=par)
        st.session_state["results"] = res
        st.session_state["opts"] = opts
        st.session_state["run_scenarios"] = scs
        bar.progress(1.0, "Concluído")
        st.success("Simulação concluída. Veja as etapas 6 (validação), 7 (visão unificada), 8 (comparação) e 9 (impacto entre objetos).")
    if "results" in st.session_state:
        df = X.results_frame(st.session_state["results"])
        st.subheader("Resultados por replicação")
        st.dataframe(df, use_container_width=True, hide_index=True, height=320)

# ===================================================================== 6
# ETAPA 6 — Validação: o cenário Atual deve reproduzir o log (como no OCPS).
# Compara KPIs, frequência das atividades, lead time, variantes e BWE por IL.
elif page == PAGES[6]:
    header("Etapa 6", "Validação do simulador: cenário Atual × log",
           "Como no OCPS, o log simulado com a política observada deve reproduzir o log original. "
           "Diferenças grandes indicam onde o modelo simplifica a realidade.")
    res = st.session_state.get("results")
    if not res or not any(r["policy"] == "ATUAL" for r in res):
        st.info("Execute a simulação (etapa 5) incluindo o cenário **Atual**.")
        st.stop()
    opts = st.session_state["opts"]
    if opts.get("initial_state") == "estado_atual" or abs(opts.get("horizon_days", P.horizon_days) - P.horizon_days) > 1:
        st.warning("A validação compara contagens com o log inteiro: rode com **estado inicial = início do log** e "
                   f"**horizonte = {P.horizon_days:.0f} dias** para uma comparação justa.")
    lb = load_log_baseline(path, KEY, opts["warmup_days"], opts["bucket_days"])
    asis = [r for r in res if r["policy"] == "ATUAL"]
    vt = X.validation_table(lb, asis)
    c = st.columns(4)
    sim_k = pd.DataFrame([r["kpis"] for r in asis]).mean()
    c[0].metric("Fill rate — log", f"{lb['kpis']['nível de serviço (fill rate)']:.1%}",
                f"simulado {sim_k['nível de serviço (fill rate)']:.1%}", delta_color="off", delta_arrow="off")
    c[1].metric("% SO com backorder — log", f"{lb['kpis']['% pedidos de venda com backorder']:.1f}%",
                f"simulado {sim_k['% pedidos de venda com backorder']:.1f}%", delta_color="off", delta_arrow="off")
    c[2].metric("Pedidos de compra — log", lb["kpis"]["nº pedidos de compra"],
                f"simulado {sim_k['nº pedidos de compra']:.0f}", delta_color="off", delta_arrow="off")
    bw_sim = np.mean([r["bw_summary"]["BWE rede"] for r in asis])
    c[3].metric("Efeito chicote (rede) — log", f"{lb['bw']['summary']['BWE rede']:.2f}",
                f"simulado {bw_sim:.2f}", delta_color="off", delta_arrow="off")

    # barras horizontais: nº de eventos por atividade, log × simulado
    st.subheader("Frequência das atividades")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=vt["atividade"], x=vt["log"], name="log", orientation="h",
                         marker=dict(color=LOG_COLOR, cornerradius=3)))
    fig.add_trace(go.Bar(y=vt["atividade"], x=vt["simulado (média)"], name="simulado (Atual)", orientation="h",
                         marker=dict(color=SERIES[0], cornerradius=3)))
    fig.update_xaxes(type="log")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 620, xtitle="eventos (escala log)"), use_container_width=True)
    st.dataframe(vt, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Lead time (PO → recebimento)")
        sim_lt = np.concatenate([r["lead_times"] for r in asis])
        log_lt = P.supplier_lead_time["__all__"].values
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=log_lt, histnorm="probability density", name="log", nbinsx=40,
                                   marker=dict(color=LOG_COLOR), opacity=0.6))
        fig.add_trace(go.Histogram(x=sim_lt, histnorm="probability density", name="simulado", nbinsx=40,
                                   marker=dict(color=SERIES[0]), opacity=0.6))
        fig.update_layout(barmode="overlay")
        st.plotly_chart(style(fig, 340, xtitle="dias", ytitle="densidade"), use_container_width=True)
        from scipy.stats import ks_2samp
        ks = ks_2samp(log_lt, sim_lt)
        st.caption(f"KS de duas amostras: D = {ks.statistic:.3f} (p = {ks.pvalue:.3g})")
    with c2:
        st.subheader("Variantes do SalesOrderItem")
        fl = flat_cached(path, KEY, "SalesOrderItem")
        lv = fl.groupby("case")["act"].apply(lambda s: " → ".join(s)).value_counts(normalize=True)
        mapping = {"Sales Order Item Created → Reservation Created → Goods Issue": "Created → Reservation → Goods Issue",
                   "Sales Order Item Created → Backorder Registered": "Created → Backorder",
                   "Sales Order Item Created → Reservation Created → Goods Issue → Backorder Registered":
                       "Created → Reservation → Goods Issue → Backorder"}
        sv = pd.DataFrame([r["so_variants"] for r in asis]).fillna(0).sum()
        sv = sv / sv.sum()
        rows = [{"variante": short, "log %": lv.get(full, 0) * 100, "simulado %": sv.get(short, 0) * 100}
                for full, short in mapping.items()]
        st.dataframe(pd.DataFrame(rows).round(1), hide_index=True, use_container_width=True)

    # dispersão: cada ponto é um IL; perto da diagonal = simulador reproduz o log
    st.subheader("Efeito chicote por IL — log × simulado (Atual)")
    lp = lb["bw"]["per_il"][["il", "BWE"]].rename(columns={"BWE": "log"})
    sp = pd.concat([pd.DataFrame(r["bw_per_il"]) for r in asis]).groupby("il")["BWE"].mean().rename("simulado")
    m = lp.merge(sp, on="il").dropna()
    fig = go.Figure(go.Scatter(x=m["log"], y=m["simulado"], mode="markers", text=m["il"],
                               marker=dict(size=9, color=SERIES[0], line=dict(color="white", width=1.5)),
                               hovertemplate="%{text}<br>log: %{x:.2f}<br>simulado: %{y:.2f}<extra></extra>"))
    mx = float(np.nanmax([m["log"].max(), m["simulado"].max(), 1]))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=MUTED, dash="dot"), name="y = x"))
    st.plotly_chart(style(fig, 380, xtitle="BWE no log", ytitle="BWE simulado", legend=False), use_container_width=True)

    st.caption("Para analisar o OCEL simulado com as mesmas técnicas de mineração usadas no log original "
               "(OC-DFG, variantes, desempenho), veja a etapa 7.")

# ===================================================================== 7
# ETAPA 7 — Visão unificada (van der Aalst): o OCEL simulado é gravado em
# disco e lido pelo PM4Py exatamente como o original, permitindo comparar
# original × simulado com as mesmas técnicas de mineração.
elif page == PAGES[7]:
    header("Etapa 7", "Visão unificada: log original × log simulado",
           "Van der Aalst (2023) propõe analisar o log simulado **com as mesmas técnicas** de mineração de "
           "processos usadas no log original: assim o processo “as is” e os possíveis “to be” são vistos pela mesma "
           "lente (mineração comparativa). Aqui o OCEL simulado de cada cenário é lido pelo PM4Py exatamente "
           "como o original.")
    res = st.session_state.get("results")
    recs = [r for r in (res or []) if r.get("ocel_json")]
    if not recs:
        st.info("Na etapa 5, escolha ao menos um cenário em “Gerar OCEL simulado” e execute a simulação.")
        st.stop()
    sc = st.selectbox("Log simulado", [r["scenario"] for r in recs])
    rr = next(r for r in recs if r["scenario"] == sc)
    sim_key = hashlib.md5(rr["ocel_json"][:2000].encode() + str(len(rr["ocel_json"])).encode()).hexdigest()[:12]
    sim_path = CACHE / f"sim_{sim_key}.json"
    if not sim_path.exists():
        sim_path.write_text(rr["ocel_json"])
    DS = load_data(str(sim_path), sim_key)
    days_orig, days_sim = D.horizon_days, max(DS.horizon_days, 1)
    c = st.columns(4)
    c[0].metric("Eventos (original)", f"{len(D.events):,}".replace(",", "."))
    c[1].metric("Eventos (simulado)", f"{len(DS.events):,}".replace(",", "."))
    c[2].metric("Horizonte original (dias)", f"{days_orig:.0f}")
    c[3].metric("Horizonte simulado (dias)", f"{days_sim:.0f}")

    tabs = st.tabs(["Atividades por dia", "Variantes por tipo de objeto", "Desempenho (DFG)", "OC-DFG simulado",
                    "Modelo de objetos"])
    with tabs[0]:
        a = (D.events[pm.ACT].value_counts() / days_orig).rename("original")
        b = (DS.events[pm.ACT].value_counts() / days_sim).rename("simulado")
        t = pd.concat([a, b], axis=1).fillna(0).sort_values("original", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=t.index, x=t["original"], name="original", orientation="h", marker=dict(color=LOG_COLOR, cornerradius=3)))
        fig.add_trace(go.Bar(y=t.index, x=t["simulado"], name=f"simulado — {POLICY_SHORT.get(rr['policy'], sc)}",
                             orientation="h", marker=dict(color=color_for(rr["policy"]), cornerradius=3)))
        fig.update_xaxes(type="log")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 620, "Eventos por dia (normalizado pelo horizonte)", xtitle="eventos/dia (escala log)"),
                        use_container_width=True)
    with tabs[1]:
        ot = st.selectbox("Tipo de objeto", ["SalesOrderItem", "PurchaseOrderItem", "ReplenishmentProposal", "TransferOrder"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.dataframe(pm.variants(flat_cached(path, KEY, ot), 10).round(2), use_container_width=True)
        with c2:
            st.markdown("**Simulado**")
            if ot in set(DS.objects[pm.OTYPE]):
                st.dataframe(pm.variants(flat_cached(str(sim_path), sim_key, ot), 10).round(2), use_container_width=True)
            else:
                st.caption("Nenhum objeto deste tipo no log simulado.")
    with tabs[2]:
        ot2 = st.selectbox("Tipo de objeto ", ["SalesOrderItem", "PurchaseOrderItem", "ReplenishmentProposal"])
        pr = pm.performance_dfg_table(flat_cached(path, KEY, ot2))
        if ot2 in set(DS.objects[pm.OTYPE]):
            ps = pm.performance_dfg_table(flat_cached(str(sim_path), sim_key, ot2))
            m = pr[["de", "para", "frequência", "média (h)", "mediana (h)"]].merge(
                ps[["de", "para", "frequência", "média (h)", "mediana (h)"]], on=["de", "para"], how="outer",
                suffixes=(" original", " sim"))
            st.dataframe(m.round(2), use_container_width=True, hide_index=True)
            mm = m.dropna(subset=["média (h) original", "média (h) sim"])
            fig = go.Figure(go.Scatter(x=mm["média (h) original"], y=mm["média (h) sim"], mode="markers",
                                       text=mm["de"] + " → " + mm["para"],
                                       marker=dict(size=10, color=SERIES[0], line=dict(color="white", width=1.5)),
                                       hovertemplate="%{text}<br>original: %{x:.2f} h<br>sim: %{y:.2f} h<extra></extra>"))
            mx = float(max(mm["média (h) original"].max(), mm["média (h) sim"].max(), 1))
            fig.add_trace(go.Scatter(x=[0.001, mx], y=[0.001, mx], mode="lines", line=dict(color=MUTED, dash="dot")))
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
            st.plotly_chart(style(fig, 380, "Tempo médio entre atividades: original × simulado", xtitle="original (h)",
                                  ytitle="simulado (h)", legend=False), use_container_width=True)
    with tabs[3]:
        ots = [t for t in pm.OT_ORDER if t in set(DS.objects[pm.OTYPE])]
        sel = st.multiselect("Tipos de objeto", ots, default=[t for t in ["SalesOrderItem", "ReplenishmentProposal",
                                                                          "PurchaseOrderItem"] if t in ots], key="ocdfg_sim")
        ann = st.radio("Anotação ", ["frequency", "performance"], horizontal=True,
                       format_func=lambda x: {"frequency": "frequência", "performance": "desempenho"}[x])
        show_svg(svg_file(str(sim_path), sim_key, "ocdfg", ann, tuple(sorted(sel))), 650)
    with tabs[4]:
        st.markdown("Objetos gerados por tipo e cardinalidades O2O no log simulado (compare com a etapa 1).")
        c1, c2 = st.columns(2)
        c1.dataframe(D.objects[pm.OTYPE].value_counts().rename("original"), use_container_width=True)
        c2.dataframe(DS.objects[pm.OTYPE].value_counts().rename("simulado"), use_container_width=True)
        cs_ = pm.o2o_cardinalities(DS)
        st.dataframe(cs_[["origem", "qualificador", "destino", "ligações", "cardinalidade", "média destino/origem"]],
                     use_container_width=True, hide_index=True)

# ===================================================================== 8
# ETAPA 8 — Comparação de cenários: efeito chicote (rede, por IL, por nível),
# serviço, custo, trade-offs e séries temporais, com IC 95%.
elif page == PAGES[8]:
    header("Etapa 8", "Comparação de cenários",
           "Efeito chicote = Var(pedidos ao fornecedor) / Var(demanda do cliente). Valores acima de 1 indicam "
           "amplificação da variabilidade rumo a montante. Barras mostram a média das replicações e o IC 95%.")
    res = st.session_state.get("results")
    if not res:
        st.info("Execute a simulação na etapa 5.")
        st.stop()
    opts = st.session_state["opts"]
    lb = load_log_baseline(path, KEY, opts["warmup_days"], opts["bucket_days"])
    df = X.results_frame(res)
    pol_of = {r["scenario"]: r["policy"] for r in res}
    order = list(dict.fromkeys(df["cenário"]))
    pols = [pol_of[s] for s in order]

    def agg(col):
        """Média e IC 95% de uma métrica para cada cenário (alimenta os gráficos de barras)."""
        rows = []
        for sc in order:
            mm, lo, hi = X.ci95(df.loc[df["cenário"] == sc, col])
            rows.append({"cenário": POLICY_SHORT.get(pol_of[sc], sc), "full": sc, "m": mm, "lo": lo, "hi": hi})
        return pd.DataFrame(rows)

    # ---- KPI tiles: melhor cenário em cada critério
    st.subheader("Resumo")
    summ = X.summary_table(df, ["BWE rede", "BWE mediano IL", "NSAmp mediano IL", "nível de serviço (fill rate)",
                                "% pedidos de venda com backorder", "estoque médio (un/IL)", "nº pedidos de compra",
                                "custo total"])
    best_bw = summ.loc[summ["BWE rede"].idxmin(), "cenário"]
    best_ns = summ.loc[summ["nível de serviço (fill rate)"].idxmax(), "cenário"]
    best_c = summ.loc[summ["custo total"].idxmin(), "cenário"]
    c = st.columns(3)
    c[0].metric("Menor efeito chicote (rede)", POLICY_SHORT.get(pol_of[best_bw], best_bw),
                f"BWE {summ['BWE rede'].min():.2f}", delta_color="off", delta_arrow="off")
    c[1].metric("Maior nível de serviço", POLICY_SHORT.get(pol_of[best_ns], best_ns),
                f"{summ['nível de serviço (fill rate)'].max():.1%}", delta_color="off", delta_arrow="off")
    c[2].metric("Menor custo total", POLICY_SHORT.get(pol_of[best_c], best_c),
                f"{summ['custo total'].min():,.0f}".replace(",", "."), delta_color="off", delta_arrow="off")

    t = st.tabs(["Efeito chicote", "Por nível da cadeia", "Por ItemLocation", "Serviço e custo",
                 "Séries temporais", "Tabela"])
    # aba: BWE da rede e mediano por IL + distribuição entre ILs
    with t[0]:
        c1, c2 = st.columns(2)
        a = agg("BWE rede")
        c1.plotly_chart(bar_ci(a, "m", "lo", "hi", "cenário", pols, "BWE da rede (todas as ILs)", "BWE",
                               ref=lb["bw"]["summary"]["BWE rede"]), use_container_width=True)
        a = agg("BWE mediano IL")
        c2.plotly_chart(bar_ci(a, "m", "lo", "hi", "cenário", pols, "BWE mediano por ItemLocation", "BWE",
                               ref=lb["bw"]["summary"]["BWE mediano IL"]), use_container_width=True)
        st.caption("Linha tracejada: valor medido no log original. Linha pontilhada: BWE = 1 (sem amplificação). "
                   "O BWE da rede agrega todas as ILs (as flutuações individuais se compensam); o BWE mediano "
                   "por IL mostra a amplificação em cada elo.")
        fig = go.Figure()
        for i, sc in enumerate(order):
            v = pd.concat([pd.DataFrame(r["bw_per_il"]) for r in res if r["scenario"] == sc])["BWE"].dropna()
            fig.add_trace(go.Box(y=v, name=POLICY_SHORT.get(pol_of[sc], sc), marker_color=color_for(pol_of[sc], i),
                                 boxpoints="outliers", line=dict(width=1.5)))
        fig.add_trace(go.Box(y=lb["bw"]["per_il"]["BWE"].dropna(), name="Log original", marker_color=LOG_COLOR))
        fig.update_yaxes(type="log")
        st.plotly_chart(style(fig, 420, "Distribuição do BWE entre ItemLocations (todas as replicações)",
                              ytitle="BWE (escala log)", legend=False), use_container_width=True)
    # aba: BWE por nível de agregação (Location, fornecedor, rede)
    with t[1]:
        lv = pd.DataFrame([{"cenário": POLICY_SHORT.get(r["policy"], r["scenario"]), "policy": r["policy"], **r["bw_levels"]}
                           for r in res])
        levels = [c_ for c_ in lv.columns if c_ not in ("cenário", "policy")]
        mean = lv.groupby(["cenário", "policy"], sort=False)[levels].mean().reset_index()
        fig = go.Figure()
        for i, (_, r) in enumerate(mean.iterrows()):
            fig.add_trace(go.Bar(x=levels, y=[r[l_] for l_ in levels], name=r["cenário"],
                                 marker=dict(color=color_for(r["policy"], i), cornerradius=3)))
        fig.add_trace(go.Scatter(x=levels, y=[lb["bw"]["levels"].get(l_, np.nan) for l_ in levels], mode="markers",
                                 name="Log original", marker=dict(symbol="line-ew", size=26, color=LOG_COLOR,
                                                              line=dict(width=3, color=LOG_COLOR))))
        fig.update_yaxes(type="log")
        st.plotly_chart(style(fig, 440, "BWE por nível de agregação", ytitle="BWE (escala log)"),
                        use_container_width=True)
        st.caption("Location: soma das ILs de cada armazém. Fornecedor: pedidos recebidos pelo fornecedor ÷ "
                   "demanda final das ILs que ele atende.")
    # aba: tabela de BWE de cada IL em cada cenário
    with t[2]:
        per = pd.concat([pd.DataFrame(r["bw_per_il"]).assign(cenário=POLICY_SHORT.get(r["policy"], r["scenario"]))
                         for r in res])
        piv = per.groupby(["il", "cenário"])["BWE"].mean().unstack()[[POLICY_SHORT.get(p, p) for p in pols]]
        piv["Log original"] = lb["bw"]["per_il"].set_index("il")["BWE"]
        info = pd.DataFrame({il: {"classe": ip.material_class, "location": ip.location,
                                  "demanda/dia": ip.daily_mean} for il, ip in P.item_locations.items()}).T
        st.dataframe(info.join(piv).round(2), use_container_width=True, height=500)
    # aba: serviço, custo, estoque, nº de pedidos e o trade-off BWE × serviço
    with t[3]:
        c1, c2 = st.columns(2)
        a = agg("nível de serviço (fill rate)")
        c1.plotly_chart(bar_ci(a, "m", "lo", "hi", "cenário", pols, "Nível de serviço (fill rate)", "fill rate",
                               ref=lb["kpis"]["nível de serviço (fill rate)"]), use_container_width=True)
        a = agg("custo total")
        c2.plotly_chart(bar_ci(a, "m", "lo", "hi", "cenário", pols, "Custo total (estocagem + pedidos + falta)",
                               "custo"), use_container_width=True)
        c1, c2 = st.columns(2)
        a = agg("estoque médio (un/IL)")
        c1.plotly_chart(bar_ci(a, "m", "lo", "hi", "cenário", pols, "Estoque médio por IL", "unidades"),
                        use_container_width=True)
        a = agg("nº pedidos de compra")
        c2.plotly_chart(bar_ci(a, "m", "lo", "hi", "cenário", pols, "Pedidos de compra emitidos", "pedidos",
                               ref=lb["kpis"]["nº pedidos de compra"]), use_container_width=True)
        st.markdown("**Trade-off: efeito chicote × nível de serviço** (cada ponto é uma replicação)")
        fig = go.Figure()
        for i, sc in enumerate(order):
            g_ = df[df["cenário"] == sc]
            fig.add_trace(go.Scatter(x=g_["BWE rede"], y=g_["nível de serviço (fill rate)"], mode="markers",
                                     name=POLICY_SHORT.get(pol_of[sc], sc),
                                     marker=dict(size=10, color=color_for(pol_of[sc], i), line=dict(color="white", width=1.5))))
        st.plotly_chart(style(fig, 400, xtitle="BWE da rede", ytitle="fill rate"), use_container_width=True)
    # aba: séries semanais de demanda × pedidos (o chicote "visível") e estoque líquido
    with t[4]:
        c1, c2 = st.columns([1, 1])
        sc = c1.selectbox("Cenário", order, format_func=lambda s: POLICY_SHORT.get(pol_of[s], s))
        il_opt = ["Rede (total)"] + list(P.item_locations)
        il = c2.selectbox("ItemLocation", il_opt)
        r0 = next(r for r in res if r["scenario"] == sc and r["rep"] == 0)
        idx = None if il == "Rede (total)" else r0["ils"].index(il)
        wd = r0["weekly_demand"].sum(0) if idx is None else r0["weekly_demand"][idx]
        wo = r0["weekly_orders"].sum(0) if idx is None else r0["weekly_orders"][idx]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=wd, mode="lines", name="demanda do cliente", line=dict(color=LOG_COLOR, width=2)))
        fig.add_trace(go.Scatter(y=wo, mode="lines", name="pedidos ao fornecedor",
                                 line=dict(color=color_for(pol_of[sc]), width=2)))
        st.plotly_chart(style(fig, 380, f"Semanal — {il} — {POLICY_SHORT.get(pol_of[sc], sc)} (replicação 1)",
                              xtitle="semana", ytitle="unidades"), use_container_width=True)
        ns = r0["daily_net_stock"].sum(0) if idx is None else r0["daily_net_stock"][idx]
        fig = go.Figure(go.Scatter(y=ns, mode="lines", name="estoque líquido", line=dict(color=color_for(pol_of[sc]), width=2)))
        fig.add_hline(y=0, line=dict(color=MUTED, dash="dot"))
        st.plotly_chart(style(fig, 300, "Estoque líquido diário (em mão − backorder)", xtitle="dia",
                              ytitle="unidades", legend=False), use_container_width=True)
        lw = lb["weekly_demand"].sum(0) if idx is None else lb["weekly_demand"][lb["ils"].index(il)]
        lo_ = lb["weekly_orders"].sum(0) if idx is None else lb["weekly_orders"][lb["ils"].index(il)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=lw, mode="lines", name="demanda (log)", line=dict(color=LOG_COLOR, width=2)))
        fig.add_trace(go.Scatter(y=lo_, mode="lines", name="pedidos (log)", line=dict(color=SERIES[6], width=2)))
        st.plotly_chart(style(fig, 300, f"Referência: log original — {il}", xtitle="semana", ytitle="unidades"),
                        use_container_width=True)
    # aba: tabela-resumo com IC 95% e download
    with t[5]:
        st.dataframe(summ, use_container_width=True, hide_index=True)
        st.download_button("Baixar resultados por replicação (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "resultados_replicacoes.csv", "text/csv")

# ===================================================================== 9
# ETAPA 9 — Impacto entre objetos: mede como um tipo de objeto afeta outro
# (fornecedor → cliente, cadeia de reposição → falta de estoque, demanda →
# pedidos, lead time do fornecedor → efeito chicote). Funciona para o log
# original e para os logs simulados (impact.py).
elif page == PAGES[9]:
    header("Etapa 9", "Impacto entre objetos",
           "Estas análises só são possíveis porque o OCEL liga objetos de tipos diferentes (relações E2O e O2O). "
           "Com um log tradicional, de um único tipo de caso, a demanda (SalesOrderItem) e os pedidos ao "
           "fornecedor (PurchaseOrderItem) ficariam em logs separados e o vínculo entre eles — o ItemLocation — "
           "se perderia. É o argumento central para usar mineração de processos **centrada em objetos** no "
           "estudo do efeito chicote.")
    res = st.session_state.get("results") or []

    # --- logs disponíveis: o original + os OCELs simulados gravados na etapa 5
    sources = {"Log original": (path, KEY, None)}
    for rr in res:
        if rr.get("ocel_json"):
            sp, sk = simulated_ocel_path(rr)
            sources[f"Simulado — {POLICY_SHORT.get(rr['policy'], rr['scenario'])}"] = (sp, sk, rr["policy"])
    if len(sources) == 1:
        st.info("Por enquanto só o log original está disponível. Para comparar com logs simulados, escolha "
                "cenários em “Gerar OCEL simulado” na etapa 5 e execute a simulação.")
    impacts = {name: impact_cached(p_, k_) for name, (p_, k_, _) in sources.items()}

    def src_color(name, i):
        pol = sources[name][2]
        return LOG_COLOR if pol is None else color_for(pol, i)

    t1, t2, t3, t4 = st.tabs(["1 · Fornecedor → cliente", "2 · Reposição → falta de estoque",
                              "3 · Demanda → pedidos", "4 · Lead time → efeito chicote"])

    # ------------------------------------------------------------- análise 1
    with t1:
        st.markdown(
            "**Pergunta:** a situação dos **pedidos de compra** (PurchaseOrderItem) de um ItemLocation afeta os "
            "**pedidos de venda** (SalesOrderItem) do mesmo ItemLocation?\n\n"
            "No instante em que cada pedido de venda é criado, o IL é classificado em: *sem pedido de compra em "
            "aberto*, *pedido em aberto no prazo* ou *pedido em aberto atrasado* (idade maior que a mediana do lead "
            "time). Para não confundir ILs de demanda alta e baixa, a medida principal é o **excesso de backorder "
            "em relação à média do próprio IL**, em pontos percentuais (p.p.).")
        fig = go.Figure()
        for i, (name, imp) in enumerate(impacts.items()):
            sm = imp["fornecedor_cliente"]["summary"]
            fig.add_trace(go.Bar(x=sm["situação do abastecimento"], y=sm["excesso_vs_media_do_il"], name=name,
                                 marker=dict(color=src_color(name, i), cornerradius=3),
                                 hovertemplate="%{x}<br>%{y:+.1f} p.p.<extra>" + name + "</extra>"))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        st.plotly_chart(style(fig, 380, "Excesso de backorder (p.p.) conforme a situação do abastecimento",
                              ytitle="p.p. acima/abaixo da média do IL"), use_container_width=True)
        orig = impacts["Log original"]["fornecedor_cliente"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Log original — detalhe**")
            st.dataframe(orig["summary"].rename(columns={
                "pedidos_de_venda": "pedidos de venda", "pct_backorder_bruto": "% backorder",
                "excesso_vs_media_do_il": "excesso (p.p.)"}).round(1), hide_index=True, use_container_width=True)
            st.caption(f"Pedido “atrasado”: em aberto há mais de {orig['overdue_days']:.1f} dias (mediana do lead "
                       f"time). Teste de Mann–Whitney (atrasado > no prazo): p = {orig['test_p']:.3g}.")
        with c2:
            fig = go.Figure()
            for i, (name, imp) in enumerate(impacts.items()):
                bw = imp["fornecedor_cliente"]["by_wait"]
                fig.add_trace(go.Scatter(x=bw["faixa de espera"].astype(str), y=bw["excesso_vs_media_do_il"],
                                         mode="lines+markers", name=name,
                                         line=dict(color=src_color(name, i), width=2), marker=dict(size=8)))
            fig.add_hline(y=0, line=dict(color=MUTED, width=1))
            st.plotly_chart(style(fig, 330, "Excesso de backorder × há quanto tempo o IL espera o fornecedor",
                                  xtitle="idade do pedido de compra mais antigo em aberto", ytitle="p.p."),
                            use_container_width=True)
        st.markdown(
            '<div class="note"><b>Como ler.</b> Valores positivos indicam que, naquela situação, os pedidos de '
            'venda do IL viram backorder com mais frequência que o normal para aquele IL. A diferença entre '
            '“sem pedido em aberto” e “pedido em aberto” mostra o vínculo entre os dois tipos de objeto: a falta '
            'de estoque que o cliente sente coincide com o período em que o IL espera a reposição. Se a curva '
            'do simulado acompanha a do original, o simulador reproduz também essa interação entre objetos.</div>',
            unsafe_allow_html=True)

    # ------------------------------------------------------------- análise 2
    with t2:
        st.markdown(
            "**Pergunta:** quais características da **cadeia de reposição** de cada ItemLocation (lead time dos "
            "PurchaseOrderItem, tempo de aprovação das ReplenishmentProposal, nº de pedidos de compra) estão "
            "associadas à **falta de estoque** sentida pelos SalesOrderItem? Cada ponto é um IL; a medida é a "
            "correlação de Spearman (ρ), que capta relações monotônicas.")
        rows = []
        for name, imp in impacts.items():
            cr = imp["reposicao_falta"]["correlations"].copy()
            cr.insert(0, "log", name)
            rows.append(cr)
        st.dataframe(pd.concat(rows).round(3), hide_index=True, use_container_width=True)
        sel = st.selectbox("Log", list(impacts))
        pi = impacts[sel]["reposicao_falta"]["per_il"]
        xvar = st.radio("Eixo x", ["pedidos de venda/dia", "lead time médio (dias)", "aprovação média (dias)"],
                        horizontal=True)
        fig = go.Figure(go.Scatter(x=pi[xvar], y=pi["% backorder"], mode="markers", text=pi["il"],
                                   marker=dict(size=10, color=src_color(sel, 0), line=dict(color="white", width=1.5)),
                                   hovertemplate="%{text}<br>x: %{x:.2f}<br>% backorder: %{y:.1f}<extra></extra>"))
        st.plotly_chart(style(fig, 380, f"% de pedidos de venda em backorder por IL — {sel}", xtitle=xvar,
                              ytitle="% backorder", legend=False), use_container_width=True)
        st.caption("Correlação não é causalidade: ILs de demanda alta também emitem mais pedidos de compra. Por "
                   "isso a análise 4 usa o simulador para isolar o efeito do lead time.")

    # ------------------------------------------------------------- análise 3
    with t3:
        st.markdown(
            "**Pergunta:** como a demanda dos **SalesOrderItem** se transmite aos **PurchaseOrderItem** em cada "
            "política? Mostramos a correlação entre a demanda da semana *t* e os pedidos ao fornecedor da semana "
            "*t + k*. Um pico em *k* indica que a política reage à demanda com *k* semanas de atraso; correlações "
            "baixas com pedidos muito variáveis indicam que a variabilidade dos pedidos **não vem da demanda**, e "
            "sim da própria regra de reposição — a origem do efeito chicote.")
        level = st.radio("Nível", ["correlação mediana por IL", "correlação rede"], horizontal=True)
        fig = go.Figure()
        lr = IMP.lagged_response(load_log_baseline(path, KEY, 56, 7)["weekly_demand"],
                                 load_log_baseline(path, KEY, 56, 7)["weekly_orders"])
        fig.add_trace(go.Scatter(x=lr["defasagem (semanas)"], y=lr[level], mode="lines+markers", name="Log original",
                                 line=dict(color=LOG_COLOR, width=2, dash="dash")))
        if res:
            scen = list(dict.fromkeys(r["scenario"] for r in res))
            for i, sc in enumerate(scen):
                reps = [r for r in res if r["scenario"] == sc]
                curves = [IMP.lagged_response(r["weekly_demand"], r["weekly_orders"])[level].to_numpy() for r in reps]
                fig.add_trace(go.Scatter(x=lr["defasagem (semanas)"], y=np.nanmean(curves, axis=0),
                                         mode="lines+markers", name=POLICY_SHORT.get(reps[0]["policy"], sc),
                                         line=dict(color=color_for(reps[0]["policy"], i), width=2)))
        else:
            st.info("Execute a simulação (etapa 5) para comparar as políticas.")
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        st.plotly_chart(style(fig, 400, "Correlação demanda(t) × pedidos(t + k)", xtitle="defasagem k (semanas)",
                              ytitle="correlação"), use_container_width=True)

    # ------------------------------------------------------------- análise 4
    with t4:
        st.markdown(
            "**Pergunta:** quanto o comportamento do **Supplier** (lead time) muda o efeito chicote e o serviço dos "
            "**ItemLocation**? Aqui o simulador funciona como experimento controlado: só o lead time do fornecedor "
            "muda (multiplicado pelo fator), todo o resto fica igual (mesmas sementes).")
        scs = st.session_state.get("scenarios")
        if not scs:
            from ocsim.policies import default_scenarios
            scs = default_scenarios()
        c = st.columns(3)
        factors = c[0].multiselect("Fatores de lead time", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                                   default=[0.5, 1.0, 1.5, 2.0])
        n_rep = c[1].number_input("Replicações por ponto", 1, 50, 3)
        par = c[2].toggle("Usar vários núcleos", X.parallel_supported(), key="par_sens")
        st.caption(f"{len(factors) * len(scs) * n_rep} replicações no total.")
        if st.button("▶ Rodar análise de sensibilidade", type="primary") and factors:
            bar = st.progress(0.0, "Iniciando…")
            out = []
            for j, f in enumerate(sorted(factors)):
                # cópia de cada cenário, mudando apenas o fator de lead time
                sc_f = [dataclasses.replace(sc, lead_time_factor=float(f)) for sc in scs]
                rr = X.run_experiment(P, sc_f, int(n_rep), 42, {"warmup_days": 56, "bucket_days": 7},
                                      parallel=par,
                                      progress=lambda fr, m, j=j: bar.progress(min((j + fr) / len(factors), 1.0),
                                                                               f"fator {f}: {m}"))
                dfr = X.results_frame(rr)
                dfr["fator de lead time"] = f
                out.append(dfr)
            st.session_state["sens"] = pd.concat(out)
            bar.progress(1.0, "Concluído")
        sens = st.session_state.get("sens")
        if sens is not None:
            c1, c2 = st.columns(2)
            for col, metric, ttl in [(c1, "BWE rede", "Efeito chicote da rede"),
                                     (c2, "nível de serviço (fill rate)", "Nível de serviço (fill rate)")]:
                fig = go.Figure()
                for i, (sc, g) in enumerate(sens.groupby("cenário", sort=False)):
                    m = g.groupby("fator de lead time")[metric].mean()
                    pol = g["política"].iloc[0]
                    fig.add_trace(go.Scatter(x=m.index, y=m.values, mode="lines+markers",
                                             name=POLICY_SHORT.get(pol, sc),
                                             line=dict(color=color_for(pol, i), width=2), marker=dict(size=8)))
                col.plotly_chart(style(fig, 380, ttl, xtitle="fator do lead time do fornecedor", ytitle=metric),
                                 use_container_width=True)
            st.dataframe(sens.groupby(["cenário", "fator de lead time"], sort=False)[
                ["BWE rede", "BWE mediano IL", "nível de serviço (fill rate)", "custo total"]].mean().round(3),
                use_container_width=True)

# ===================================================================== 10
# ETAPA 10 — Exportar: CSVs de resultados e parâmetros e OCELs simulados.
elif page == PAGES[10]:
    header("Etapa 10", "Exportar")
    res = st.session_state.get("results")
    if res:
        df = X.results_frame(res)
        st.download_button("Resultados por replicação (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "resultados_replicacoes.csv", "text/csv")
        per = pd.concat([pd.DataFrame(r["bw_per_il"]).assign(cenário=r["scenario"], replicação=r["rep"]) for r in res])
        st.download_button("Efeito chicote por IL e replicação (CSV)", per.to_csv(index=False).encode("utf-8"),
                           "bwe_por_il.csv", "text/csv")
        for r in res:
            if r.get("ocel_json"):
                st.download_button(f"OCEL simulado — {r['scenario']} (JSON, OCEL 2.0)", r["ocel_json"].encode("utf-8"),
                                   f"ocel_simulado_{r['policy']}.json", "application/json")
    else:
        st.info("Execute a simulação para exportar resultados.")
    st.download_button("Parâmetros de duração ajustados (CSV)", P.delay_table().to_csv(index=False).encode("utf-8"),
                       "distribuicoes_duracoes.csv", "text/csv")
    rows = [{"IL": il, "p_order": ip.p_order, "quantidade": ip.qty.label(), "media_dia": ip.daily_mean,
             "desvio_dia": ip.daily_std, "lead_time": ip.lead_time_mean, "fornecedores": json.dumps(ip.supplier_probs)}
            for il, ip in P.item_locations.items()]
    st.download_button("Parâmetros de demanda por IL (CSV)", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                       "parametros_demanda_il.csv", "text/csv")
