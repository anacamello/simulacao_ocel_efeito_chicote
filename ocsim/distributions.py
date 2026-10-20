"""
distributions.py — Ajuste e amostragem de distribuições de probabilidade
=========================================================================

PAPEL NO PROJETO
----------------
Toda grandeza aleatória da simulação (tempo entre atividades, quantidade
pedida, atributo de um objeto, nº de vezes que uma atividade ocorre...) é
representada por um objeto ``Dist``. O simulador nunca usa um número fixo:
ele sempre chama ``dist.sample(rng)`` para sortear um valor "parecido com o
log".

COMO O AJUSTE É FEITO (função ``fit_numeric``)
---------------------------------------------
1. Recebe as observações do log (ex.: os 575 tempos "proposta criada →
   proposta aprovada").
2. Testa 6 distribuições candidatas: normal, exponencial, gama, lognormal,
   Weibull e uniforme. Para cada uma, encontra os parâmetros (média, forma,
   escala...) pelo método da MÁXIMA VEROSSIMILHANÇA: escolhe os valores dos
   parâmetros que tornam MAIS PROVÁVEL ter observado exatamente esses dados.
   Quem faz essa conta é a função ``fit`` da biblioteca SciPy.
3. Compara as candidatas pelo AIC (Critério de Informação de Akaike):
        AIC = 2k − 2·ln(L)
   onde k = nº de parâmetros da distribuição e L = verossimilhança (o quanto
   ela "explica" os dados). O AIC premia o bom ajuste e penaliza
   distribuições com mais parâmetros. Quanto MENOR o AIC, melhor.
4. Fica com a de menor AIC e registra o teste de Kolmogorov–Smirnov (KS), que
   mede a maior distância entre a curva acumulada dos dados e a da
   distribuição (estatística D) e dá um p-valor. p-valor baixo (< 0,05) indica
   que a distribuição não descreve perfeitamente os dados — comum com
   milhares de observações, quando qualquer pequena diferença vira
   "significativa". Por isso a interface também oferece o modo empírico.
5. Casos especiais:
   - nenhuma observação           → constante 0;
   - todas as observações iguais  → constante;
   - menos de 15 observações      → distribuição EMPÍRICA: em vez de uma curva
     teórica, sorteia diretamente um dos valores observados (técnica chamada
     "bootstrap" ou reamostragem). Com tão poucos pontos, estimar 2 ou 3
     parâmetros seria pouco confiável.

Variáveis categóricas (ex.: ``carrier``) usam as frequências relativas
observadas (``fit_categorical``); contagens inteiras (ex.: nº de "Goods
Receipt" por pedido) usam a distribuição discreta empírica (``fit_discrete``).

Termos de programação (classe, dataclass, gerador de números aleatórios...)
estão explicados no glossário do arquivo ocsim/__init__.py.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Distribuições candidatas: nome em português → objeto do scipy.stats
# ---------------------------------------------------------------------------
CANDIDATES = {
    "normal": stats.norm,
    "exponencial": stats.expon,
    "gama": stats.gamma,
    "lognormal": stats.lognorm,
    "weibull": stats.weibull_min,
    "uniforme": stats.uniform,
}

# abaixo deste nº de observações usamos a distribuição empírica
MIN_SAMPLES_PARAMETRIC = 15
# Com muitas observações (ex.: 16 mil pedidos de venda) o ajuste fica lento.
# Nesses casos usamos 4000 observações sorteadas ao acaso (uma "subamostra"):
# o resultado é praticamente o mesmo e o cálculo fica muito mais rápido.
MAX_FIT_SAMPLES = 4000


# ===========================================================================
# Classe Dist: guarda o resultado do ajuste e sabe sortear valores
# ===========================================================================
@dataclass
class Dist:
    """Distribuição ajustada a partir do log.

    ``kind`` indica o tipo:
      * 'const'       – valor fixo (variável sem variação no log);
      * 'empirical'   – sorteia um dos valores observados;
      * 'parametric'  – distribuição teórica (``name`` + ``params``);
      * 'categorical' – valores textuais com probabilidades;
      * 'discrete'    – valores inteiros com probabilidades.
    """

    kind: str
    name: str = ""                       # nome da distribuição paramétrica escolhida
    params: tuple = ()                   # parâmetros estimados (ordem do scipy)
    values: np.ndarray | None = None     # observações originais (usadas no modo empírico e nos gráficos)
    probs: np.ndarray | None = None      # probabilidades (categórica/discreta)
    lower: float | None = None           # menor valor observado (limite para o sorteio)
    upper: float | None = None           # maior valor observado
    integer: bool = False                # arredondar o valor sorteado para inteiro?
    ks_stat: float | None = None         # estatística D do teste KS
    ks_p: float | None = None            # p-valor do teste KS
    aic: float | None = None             # AIC da distribuição escolhida
    ranking: list = field(default_factory=list)  # todas as candidatas: (nome, AIC, KS, p)
    n: int = 0                           # nº de observações usadas
    mode: str = "parametric"             # modo de sorteio preferido ('parametric' ou 'empirical')

    # ----------------------------------------------------------- sorteio
    def sample(self, rng: np.random.Generator, mode: str | None = None) -> Any:
        """Sorteia UM valor desta distribuição.

        ``rng`` é o gerador de números aleatórios da replicação. Cada replicação
        tem o seu, criado a partir de uma semente: com a mesma semente, os
        sorteios se repetem exatamente (resultado reproduzível).
        ``mode`` permite forçar o sorteio 'empirical' (valores do log) ou
        'parametric' (curva ajustada)."""
        mode = mode or self.mode

        # (a) valor constante
        if self.kind == "const":
            return self.params[0]

        # (b) categórica / discreta: sorteio pela probabilidade ACUMULADA.
        #     Exemplo com carrier: probs = [0,98 (SIM), 0,02 (FORCED)].
        #     A acumulada é [0,98; 1,00], que divide o intervalo de 0 a 1 em
        #     faixas: [0 a 0,98) → SIM e [0,98 a 1) → FORCED. Sorteamos um número
        #     ao acaso entre 0 e 1 e vemos em que faixa ele cai (searchsorted
        #     faz essa busca). Assim SIM sai em 98% das vezes.
        #     A acumulada é calculada só na primeira vez e guardada em _cum,
        #     porque este sorteio é feito milhares de vezes por replicação.
        if self.kind in ("categorical", "discrete"):
            cum = self.__dict__.get("_cum")
            if cum is None:
                cum = np.cumsum(self.probs)
                cum[-1] = 1.0  # evita erro de arredondamento no último intervalo
                self.__dict__["_cum"] = cum
            return self.values[int(np.searchsorted(cum, rng.random(), side="right"))]

        # (c) empírica: escolhe ao acaso uma das observações do log (reamostragem)
        if self.kind == "empirical" or mode == "empirical":
            x = float(self.values[rng.integers(len(self.values))])
        # (d) paramétrica: sorteia da curva teórica ajustada (ex.: Weibull do lead time)
        else:
            x = _fast_rvs(self.name, self.params, rng)
            # limita ao intervalo observado no log, evitando valores absurdos
            # gerados nas caudas longas (ex.: lead time de 2 anos)
            if self.lower is not None:
                x = max(x, self.lower)
            if self.upper is not None:
                x = min(x, self.upper)

        if self.integer:
            x = int(round(x))
        return x

    # ----------------------------------------------------------- descrição (para tabelas da interface)
    @property
    def mean(self) -> float:
        """Média da distribuição (calculada sobre as observações)."""
        if self.kind == "const":
            return float(self.params[0]) if _isnum(self.params[0]) else float("nan")
        if self.kind in ("categorical",):
            return float("nan")  # média não faz sentido para texto
        if self.kind == "discrete":
            return float(np.dot(self.values.astype(float), self.probs))
        return float(np.mean(self.values)) if self.values is not None and len(self.values) else float("nan")

    @property
    def std(self) -> float:
        """Desvio-padrão. ``ddof=1`` = desvio "amostral" (divide por n − 1),
    o usual quando os dados são uma amostra e não a população inteira."""
        if self.kind == "const":
            return 0.0
        if self.kind == "discrete":
            v = self.values.astype(float)
            m = np.dot(v, self.probs)
            return float(math.sqrt(np.dot((v - m) ** 2, self.probs)))
        if self.kind == "categorical" or self.values is None or len(self.values) < 2:
            return float("nan")
        return float(np.std(self.values, ddof=1))

    def label(self) -> str:
        """Texto legível, ex.: 'weibull(forma=1.29, loc=0, escala=23.8)'."""
        if self.kind == "const":
            return f"constante({_fmt(self.params[0])})"
        if self.kind == "categorical":
            items = ", ".join(f"{v}: {p:.1%}" for v, p in zip(self.values, self.probs))
            return f"categórica({items})"
        if self.kind == "discrete":
            return f"discreta empírica ({len(self.values)} valores)"
        if self.kind == "empirical":
            return f"empírica (n={self.n})"
        pn = _param_names(self.name)
        ps = ", ".join(f"{k}={_fmt(v)}" for k, v in zip(pn, self.params))
        return f"{self.name}({ps})"

    def pdf(self, x: np.ndarray) -> np.ndarray | None:
        """Densidade teórica (usada para desenhar a curva sobre o histograma)."""
        if self.kind != "parametric":
            return None
        return CANDIDATES[self.name].pdf(x, *self.params)

    def to_row(self) -> dict:
        """Linha de tabela com o resumo do ajuste."""
        return {
            "distribuição": self.label(),
            "n": self.n,
            "média": self.mean,
            "desvio": self.std,
            "AIC": self.aic,
            "KS": self.ks_stat,
            "KS p-valor": self.ks_p,
        }


# ===========================================================================
# Funções auxiliares
# ===========================================================================
def _fast_rvs(name: str, p: tuple, rng: np.random.Generator) -> float:
    """Sorteia um valor de uma distribuição paramétrica, de forma rápida.

    O SciPy tem uma função própria de sorteio, mas ela é cerca de 50 vezes
    mais lenta por chamada. Como cada replicação faz centenas de milhares de
    sorteios, usamos o numpy diretamente, aplicando a mesma convenção de
    parâmetros do SciPy:
      loc    = deslocamento (quanto a curva é empurrada para a direita);
      escala = fator que estica ou encolhe a curva;
      forma  = parâmetro que muda o formato (ex.: assimetria da Weibull)."""
    if name == "normal":        # params = (μ, σ)
        return p[0] + p[1] * rng.standard_normal()
    if name == "exponencial":   # params = (loc, escala)
        return p[0] + p[1] * rng.standard_exponential()
    if name == "gama":          # params = (forma, loc, escala)
        return p[1] + p[2] * rng.standard_gamma(p[0])
    if name == "lognormal":     # params = (s, loc, escala): X = loc + escala·exp(s·Z)
        return p[1] + p[2] * math.exp(p[0] * rng.standard_normal())
    if name == "weibull":       # params = (forma, loc, escala)
        return p[1] + p[2] * rng.weibull(p[0])
    if name == "uniforme":      # params = (loc, amplitude)
        return p[0] + p[1] * rng.random()
    return float(CANDIDATES[name].rvs(*p, random_state=rng))


def _isnum(x) -> bool:
    """True se o valor pode ser convertido em número."""
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _fmt(v) -> str:
    """Formata números com 4 algarismos significativos."""
    if _isnum(v):
        v = float(v)
        return f"{v:.4g}"
    return str(v)


def _param_names(name: str) -> list[str]:
    """Nomes legíveis dos parâmetros de cada distribuição (para os rótulos)."""
    return {
        "normal": ["μ", "σ"],
        "exponencial": ["loc", "escala"],
        "gama": ["forma", "loc", "escala"],
        "lognormal": ["s", "loc", "escala"],
        "weibull": ["forma", "loc", "escala"],
        "uniforme": ["loc", "amplitude"],
    }.get(name, [])


# ===========================================================================
# AJUSTE de variáveis NUMÉRICAS (contínuas ou inteiras)
# ===========================================================================
def fit_numeric(
    samples,
    integer: bool = False,
    nonneg: bool = True,
    candidates: list[str] | None = None,
    seed: int = 0,
) -> Dist:
    """Ajusta a melhor distribuição (menor AIC) a amostras numéricas.

    Parâmetros
    ----------
    samples    : observações do log.
    integer    : se True, os valores sorteados serão arredondados (ex.: quantidades).
    nonneg     : se True, impede que as curvas que só valem para números
                 positivos (gama, lognormal, Weibull, exponencial) sejam
                 deslocadas para valores negativos — tempos e quantidades
                 nunca são negativos.
    candidates : restringe as distribuições testadas (opcional).
    seed       : semente usada para sortear a subamostra (mesma semente → mesmo ajuste).
    """
    # 1) limpa valores ausentes/infinitos
    x = np.asarray([v for v in samples if v is not None and np.isfinite(v)], dtype=float)
    n = len(x)

    # 2) casos degenerados
    if n == 0:
        return Dist("const", params=(0.0,), integer=integer, n=0)
    if np.ptp(x) < 1e-12:  # ptp = máximo − mínimo; se é zero, todos os valores são iguais
        return Dist("const", params=(int(x[0]) if integer else float(x[0]),), values=x, integer=integer, n=n)
    lower = float(x.min())
    upper = float(x.max())
    if n < MIN_SAMPLES_PARAMETRIC:
        return Dist("empirical", values=x, integer=integer, n=n, lower=lower, upper=upper, mode="empirical")

    # 3) subamostra para acelerar (o resultado praticamente não muda)
    rng = np.random.default_rng(seed)
    xf = x if n <= MAX_FIT_SAMPLES else rng.choice(x, MAX_FIT_SAMPLES, replace=False)

    # 4) ajusta cada candidata e calcula AIC e KS
    names = candidates or list(CANDIDATES)
    ranking = []
    for nm in names:
        dist = CANDIDATES[nm]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # o scipy emite avisos em ajustes difíceis
                if nm in ("exponencial", "gama", "lognormal", "weibull") and nonneg:
                    # floc = "loc fixo": em vez de deixar o SciPy escolher o
                    # deslocamento da curva, fixamos em 0 (ou logo abaixo do menor
                    # valor observado). Isso garante que a curva não gere valores
                    # negativos e deixa o ajuste mais estável.
                    params = dist.fit(xf, floc=min(0.0, xf.min() - 1e-9))
                else:
                    params = dist.fit(xf)
                # log-verossimilhança: soma do logaritmo da densidade de cada observação
                # (quanto maior, melhor a curva explica os dados)
                ll = np.sum(dist.logpdf(xf, *params))
                if not np.isfinite(ll):
                    continue
                k = len(params)
                aic = 2 * k - 2 * ll                        # critério de Akaike
                ks = stats.kstest(xf, dist.cdf, args=params)  # aderência
                ranking.append((nm, float(aic), float(ks.statistic), float(ks.pvalue), params))
        except Exception:
            # algumas candidatas não conseguem ser ajustadas a certos dados
            # (o cálculo numérico não converge); nesse caso, apenas pulamos para a próxima
            continue

    # 5) se nenhuma candidata funcionou, recorre à empírica
    if not ranking:
        return Dist("empirical", values=x, integer=integer, n=n, lower=lower, upper=upper, mode="empirical")

    # 6) ordena as candidatas pelo AIC (2º item de cada registro) e fica com a primeira
    ranking.sort(key=lambda r: r[1])
    nm, aic, ks, ksp, params = ranking[0]
    return Dist(
        "parametric",
        name=nm,
        params=tuple(float(p) for p in params),
        values=x,
        lower=lower,
        upper=upper,
        integer=integer,
        ks_stat=ks,
        ks_p=ksp,
        aic=aic,
        ranking=[(r[0], r[1], r[2], r[3]) for r in ranking],
        n=n,
    )


# ===========================================================================
# AJUSTE de variáveis CATEGÓRICAS (texto) e DISCRETAS (contagens)
# ===========================================================================
def fit_categorical(values) -> Dist:
    """Frequência relativa de cada categoria (ex.: carrier SIM 98%, FORCED 1,2%...)."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return Dist("const", params=(None,), n=0)
    u, c = np.unique(np.asarray(vals, dtype=object).astype(str), return_counts=True)
    order = np.argsort(-c)              # ordena da mais para a menos frequente
    u, c = u[order], c[order]
    if len(u) == 1:
        return Dist("const", params=(u[0],), values=u, probs=np.array([1.0]), n=len(vals))
    return Dist("categorical", values=u, probs=c / c.sum(), n=len(vals))


def fit_discrete(values) -> Dist:
    """Distribuição discreta empírica de contagens.

    Usada para "quantas vezes cada atividade ocorre em cada objeto":
    ex.: 'Goods Receipt' por PurchaseOrderItem → {0: 7%, 1: 93%, 2: 0,2%}."""
    vals = np.asarray(list(values), dtype=int)
    if len(vals) == 0:
        return Dist("const", params=(0,), integer=True)
    u, c = np.unique(vals, return_counts=True)
    if len(u) == 1:
        return Dist("const", params=(int(u[0]),), values=u, probs=np.array([1.0]), integer=True, n=len(vals))
    return Dist("discrete", values=u, probs=c / c.sum(), integer=True, n=len(vals))


def bernoulli(p: float) -> Dist:
    """Distribuição sim/não com probabilidade p (utilitário)."""
    p = float(min(max(p, 0.0), 1.0))
    return Dist("discrete", values=np.array([False, True], dtype=object), probs=np.array([1 - p, p]), n=0)
