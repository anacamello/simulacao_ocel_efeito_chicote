"""
Pacote ocsim — Simulação centrada em objetos da cadeia de suprimentos (efeito chicote)

Ordem em que os módulos são usados:
  1. process_mining.py  – lê o OCEL e faz a mineração de processos (PM4Py)
  2. distributions.py   – ajusta distribuições de probabilidade (AIC + KS)
  3. parameters.py      – extrai do log todos os parâmetros da simulação
  4. policies.py        – regras de reposição (cenários)
  5. simulation.py      – simulador de eventos discretos (SimPy)
  6. metrics.py         – efeito chicote e indicadores
  7. experiment.py      – replicações, validação e exportação OCEL
  8. ui_charts.py       – gráficos da interface
"""
