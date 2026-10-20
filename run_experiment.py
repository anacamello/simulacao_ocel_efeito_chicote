"""
run_experiment.py — execução em lote (sem interface)
====================================================

Roda todo o pipeline pelo terminal, útil para muitas replicações:

    python run_experiment.py --ocel caminho/do/arquivo_ocel.json --reps 30
    python run_experiment.py --ocel caminho/do/arquivo_ocel.json --reps 100 --out resultados

O caminho do OCEL (--ocel) é obrigatório.

Arquivos gerados em --out:
  distribuicoes_duracoes.csv    – distribuições ajustadas das durações entre atividades
  resultados_replicacoes.csv    – uma linha por (cenário, replicação) com todas as métricas
  resumo_cenarios.csv           – média e IC 95% por cenário
  validacao_atual_vs_log.csv    – frequência das atividades: log × cenário Atual
"""
import argparse
import os
from pathlib import Path

from ocsim import experiment as X
from ocsim import process_mining as pm
from ocsim.parameters import extract_parameters
from ocsim.policies import default_scenarios


def main():
    # -------------------------------------------------------------------
    # Argumentos de linha de comando
    # -------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Simulação OCEL — efeito chicote")
    ap.add_argument("--ocel", required=True, help="arquivo OCEL 2.0 (JSON) — obrigatório")
    ap.add_argument("--reps", type=int, default=10, help="replicações por cenário")
    ap.add_argument("--seed", type=int, default=42, help="semente base")
    ap.add_argument("--warmup", type=int, default=56, help="dias de aquecimento descartados")
    ap.add_argument("--bucket", type=int, default=7, help="agregação do efeito chicote (dias)")
    ap.add_argument("--sampling", choices=["parametric", "empirical"], default="parametric",
                    help="parametric = distribuições ajustadas; empirical = reamostragem do log")
    ap.add_argument("--no-parallel", action="store_true", help="desliga o uso de vários núcleos")
    ap.add_argument("--out", default="resultados", help="pasta de saída")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)

    # 1) leitura do OCEL (etapa 1)
    print("1/4 Lendo OCEL…")
    d = pm.load_ocel(a.ocel)

    # 2) mineração + ajuste das distribuições (etapas 2 e 3)
    print("2/4 Extraindo parâmetros e distribuições…")
    P = extract_parameters(d, progress=lambda m: print("   ", m))
    P.delay_table().to_csv(out / "distribuicoes_duracoes.csv", index=False)

    # 3) simulação dos 5 cenários padrão (etapas 4 e 5)
    print("3/4 Simulando…")
    scs = default_scenarios()
    opts = {"warmup_days": a.warmup, "bucket_days": a.bucket, "sampling_mode": a.sampling}
    res = X.run_experiment(P, scs, a.reps, a.seed, opts, parallel=not a.no_parallel,
                           progress=lambda f, m: print(f"   {m}", end="\r"))
    print()

    # 4) resumo, comparação com o log e validação (etapas 6 e 8)
    print("4/4 Resumo")
    df = X.results_frame(res)
    df.to_csv(out / "resultados_replicacoes.csv", index=False)
    cols = ["BWE rede", "BWE mediano IL", "nível de serviço (fill rate)", "estoque médio (un/IL)", "custo total"]
    summ = X.summary_table(df, cols)
    summ.to_csv(out / "resumo_cenarios.csv", index=False)
    lb = X.log_baseline(d, P, a.warmup, a.bucket)
    print(summ[["cenário"] + cols].round(3).to_string(index=False))
    print(f"\nLog original: BWE rede = {lb['bw']['summary']['BWE rede']:.3f}; "
          f"BWE mediano IL = {lb['bw']['summary']['BWE mediano IL']:.3f}")
    X.validation_table(lb, [r for r in res if r["policy"] == "ATUAL"]).to_csv(out / "validacao_atual_vs_log.csv", index=False)
    print(f"Arquivos salvos em {out.resolve()}")


# executa main() apenas quando o arquivo é chamado diretamente
# (necessário também para o paralelismo funcionar no Windows)
if __name__ == "__main__":
    main()
