"""
Pacote ocsim — Simulação centrada em objetos da cadeia de suprimentos (efeito chicote)
=====================================================================================

Um "pacote" em Python é uma pasta com vários arquivos de código (os "módulos").
Este arquivo (__init__.py) é lido quando o pacote é carregado; aqui ele serve
só de documentação: a ordem dos módulos e um glossário dos termos de
programação que aparecem nos comentários.

ORDEM EM QUE OS MÓDULOS SÃO USADOS
----------------------------------
  1. process_mining.py  – lê o arquivo OCEL e faz a mineração de processos (PM4Py)
  2. distributions.py   – ajusta distribuições de probabilidade aos dados (AIC + teste KS)
  3. parameters.py      – extrai do log todos os parâmetros que a simulação usa
  4. policies.py        – regras de reposição de estoque (os cenários)
  5. simulation.py      – o simulador de eventos discretos (SimPy)
  6. metrics.py         – cálculo do efeito chicote e dos indicadores
  7. experiment.py      – roda as replicações, valida e exporta o OCEL simulado
  8. impact.py          – mede o impacto de um tipo de objeto sobre outro
  9. ui_charts.py       – padrão visual dos gráficos da interface

GLOSSÁRIO (termos de programação usados nos comentários)
--------------------------------------------------------
Biblioteca
    Conjunto de código pronto, escrito por outras pessoas, que o projeto usa.
    Ex.: PM4Py (mineração de processos), SimPy (simulação), pandas (tabelas),
    numpy (cálculo numérico), SciPy (estatística), Streamlit (interface web).

Importar (import)
    Carregar uma biblioteca ou módulo para poder usar suas funções.
    Algumas bibliotecas, como o PM4Py, demoram alguns segundos para carregar.

Função
    Bloco de código com nome, que recebe valores (parâmetros), faz um cálculo e
    devolve um resultado. Ex.: ``fit_numeric(observacoes)`` devolve a melhor
    distribuição para aquelas observações.

Docstring
    Texto entre três aspas logo abaixo do nome de uma função ou no topo de um
    arquivo. É a documentação daquele trecho.

Classe / objeto
    Uma classe é um "molde" que agrupa dados e funções relacionados; cada
    objeto criado a partir dela guarda seus próprios valores. Ex.: a classe
    ``Dist`` guarda uma distribuição ajustada e sabe sortear valores dela.
    (Não confundir com os "objetos" do OCEL, como pedidos e fornecedores.)

dataclass
    Forma curta de escrever uma classe que serve principalmente para guardar
    dados, como uma ficha com campos (ex.: ``ILParams`` guarda os parâmetros de
    um ItemLocation).

Dicionário
    Estrutura que associa chaves a valores, como um índice.
    Ex.: {"SUP-001": 0.8, "SUP-003": 0.2} = proporção de pedidos por fornecedor.

Lista / tupla
    Sequências de valores. A lista pode ser alterada depois de criada; a tupla, não.

DataFrame (pandas)
    Uma tabela, como uma planilha: linhas, colunas com nome e um índice
    (o "rótulo" de cada linha).

Operações de tabela do pandas que aparecem muito:
    groupby  – agrupa as linhas por uma coluna e calcula algo por grupo,
               como uma tabela dinâmica do Excel. Ex.: total de demanda por IL.
    merge    – junta duas tabelas por uma coluna em comum, como o PROCV do Excel.
    unstack  – "gira" a tabela: transforma valores de uma coluna em colunas.
               Ex.: linhas (IL, dia) viram uma matriz com ILs nas linhas e dias nas colunas.
    reindex  – reorganiza as linhas/colunas numa ordem fixa e preenche as que
               faltam (ex.: dias sem pedido recebem 0).
    shift    – desloca uma coluna uma linha para cima/baixo; usado para colocar,
               lado a lado, um evento e o evento seguinte do mesmo objeto.

Array (numpy)
    Vetor ou matriz de números, com cálculos rápidos. As séries diárias da
    simulação (demanda, pedidos, estoque) são matrizes ILs × dias.

Gerador de números aleatórios / semente (seed)
    O computador produz números "aleatórios" a partir de um número inicial, a
    semente. Mesma semente → mesma sequência de sorteios → mesmo resultado.
    Isso torna a simulação reproduzível.

Função geradora e ``yield`` (SimPy)
    Uma função que pode "pausar" no meio e continuar depois. No SimPy, cada
    processo da simulação (ex.: o ciclo de vida de um pedido) é uma função
    assim: ``yield env.timeout(2)`` significa "espere 2 dias de simulação e
    então continue daqui".

Cache
    Memória de resultados já calculados. Se a mesma conta for pedida de novo,
    o resultado guardado é devolvido sem recalcular. A interface usa cache para
    não reler o arquivo OCEL de 60 MB a cada clique.

Processos paralelos
    Cópias do Python rodando ao mesmo tempo, cada uma em um núcleo do
    processador, para fazer várias replicações simultaneamente. Os modos de
    criar essas cópias diferem entre sistemas operacionais:
      • no Linux ("fork"), a cópia nasce como um clone do programa que já está
        rodando — rápido e sem surpresas;
      • no Windows e no macOS ("spawn"), cada cópia é um Python novo que
        precisa carregar tudo de novo desde o começo.

Saída do terminal (stdout)
    O texto que um programa escreve no terminal/console com ``print``.

JSON
    Formato de arquivo de texto para dados estruturados; o OCEL 2.0 usa JSON.
"""
