"""Execução em lote (sem interface).

Exemplo:
    python run_experiment.py --ocel data/inventory-management_ocel.json --reps 30 --out resultados
"""
import argparse
import os
from pathlib import Path

from ocsim import experiment as X
from ocsim import process_mining as pm
from ocsim.parameters import extract_parameters
from ocsim.policies import default_scenarios


def main():
    ap = argparse.ArgumentParser(description="Simulação OCEL — efeito chicote")
    ap.add_argument("--ocel", default="data/inventory-management_ocel.json")
    ap.add_argument("--reps", type=int, default=10, help="replicações por cenário")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=56)
    ap.add_argument("--bucket", type=int, default=7)
    ap.add_argument("--sampling", choices=["parametric", "empirical"], default="parametric")
    ap.add_argument("--no-parallel", action="store_true")
    ap.add_argument("--out", default="resultados")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    print("1/4 Lendo OCEL…")
    d = pm.load_ocel(a.ocel)
    print("2/4 Extraindo parâmetros e distribuições…")
    P = extract_parameters(d, progress=lambda m: print("   ", m))
    P.delay_table().to_csv(out / "distribuicoes_duracoes.csv", index=False)
    print("3/4 Simulando…")
    scs = default_scenarios()
    opts = {"warmup_days": a.warmup, "bucket_days": a.bucket, "sampling_mode": a.sampling}
    res = X.run_experiment(P, scs, a.reps, a.seed, opts, parallel=not a.no_parallel,
                           progress=lambda f, m: print(f"   {m}", end="\r"))
    print()
    print("4/4 Resumo")
    df = X.results_frame(res)
    df.to_csv(out / "resultados_replicacoes.csv", index=False)
    cols = ["BWE rede", "BWE mediano IL", "nível de serviço (fill rate)", "estoque médio (un/IL)", "custo total"]
    summ = X.summary_table(df, cols)
    summ.to_csv(out / "resumo_cenarios.csv", index=False)
    lb = X.log_baseline(d, P, a.warmup, a.bucket)
    print(summ[["cenário"] + cols].round(3).to_string(index=False))
    print(f"\nLog real: BWE rede = {lb['bw']['summary']['BWE rede']:.3f}; "
          f"BWE mediano IL = {lb['bw']['summary']['BWE mediano IL']:.3f}")
    X.validation_table(lb, [r for r in res if r["policy"] == "ATUAL"]).to_csv(out / "validacao_atual_vs_log.csv", index=False)
    print(f"Arquivos salvos em {out.resolve()}")


if __name__ == "__main__":
    main()
