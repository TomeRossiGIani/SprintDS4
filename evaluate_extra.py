"""Auditoria sem ajuste: desafios, custos, estabilidade por família e limiares."""

import json
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, brier_score_loss
from pipeline import Analyzer, ROOT, TASKS
from train import metrics


def main():
    a = Analyzer()
    df = pd.read_csv(ROOT / "data/corpus.csv")
    val = df[df.particao == "validacao"]
    test = df[df.particao == "teste"]
    cases = json.loads((ROOT / "data/casos_desafio.json").read_text())
    outputs = []
    for case in cases:
        outputs.append(
            {
                "id": case["id"],
                "categoria": case["categoria"],
                "texto": case["texto"],
                "esperado": {t: case[t] for t in TASKS},
                "resultado": a.analyze(case["texto"]),
            }
        )
    (ROOT / "results/desafios_resultados.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2)
    )
    challenges = {
        t: metrics([r[t] for r in cases], [r["resultado"]["classificacoes"][t] for r in outputs])
        for t in TASKS
    }
    family_rows = []
    for task, model in a.bundle["models"].items():
        for fam, g in test.groupby("familia"):
            family_rows.append(
                {
                    "tarefa": task,
                    "familia": fam,
                    "n": len(g),
                    **metrics(g[task], model.predict(g.texto)),
                }
            )
    pd.DataFrame(family_rows).to_csv(ROOT / "results/estabilidade_familias.csv", index=False)
    thresholds = []
    briers = {}
    for task, positive in [("churn", "risco"), ("oportunidade", "oportunidade")]:
        model = a.bundle["models"][task]
        idx = list(model.classes_).index(positive)
        probs = model.predict_proba(val.texto)[:, idx]
        real = (val[task] == positive).astype(int)
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            pred = probs >= threshold
            thresholds.append(
                {
                    "tarefa": task,
                    "limiar": threshold,
                    "precision": precision_score(real, pred, zero_division=0),
                    "recall": recall_score(real, pred, zero_division=0),
                    "f1": f1_score(real, pred, zero_division=0),
                    "alertas": int(pred.sum()),
                    "n": len(pred),
                }
            )
        y = (test[task] == positive).astype(int)
        prob = model.predict_proba(test.texto)[:, idx]
        briers[task] = {
            "brier": brier_score_loss(y, prob),
            "brier_constante_05": brier_score_loss(y, np.full(len(y), 0.5)),
        }
    pd.DataFrame(thresholds).to_csv(ROOT / "results/sensibilidade_validacao.csv", index=False)
    # Produto é extração por dicionário: verificar conjuntos esperados explícitos.
    expected = [
        ["RM"],
        ["Protheus"],
        ["Fluig"],
        ["RM", "Fluig"],
        ["Não identificado"],
        ["Não identificado"],
        ["RM"],
        ["Protheus"],
    ]
    exact = sum(set(row["resultado"]["produtos"]) == set(p) for row, p in zip(outputs, expected))
    audit = {
        "desafios": challenges,
        "desafios_n": len(cases),
        "produtos_exatos": {"acertos": exact, "total": len(expected)},
        "brier_teste": briers,
        "nota": "Brier apenas diagnóstico em corpus sintético; não comprova calibração real. Variação por 4 famílias descreve esta amostra, sem inferência populacional.",
    }
    (ROOT / "results/auditoria.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
