"""Robustez da seleção de modelos: validação cruzada agrupada por família e IC do teste.

Por que este script existe
--------------------------
A seleção oficial (train.py) usa uma única partição de validação com 4 famílias.
Com tão poucas famílias, os candidatos empatam (sentimento: F1 = 1,000 nos três)
ou ficam separados por diferenças mínimas (churn: 0,630 x 0,629), e a validação
de churn (0,63) diverge do teste (0,94).

Este script NÃO altera o modelo entregue nem usa o conjunto de teste para escolher
nada. Ele mede, nas 24 famílias de desenvolvimento (treino + validação), quanto o
desempenho de cada candidato varia entre grupos de famílias (GroupKFold) e calcula
um intervalo de confiança bootstrap para o F1 do modelo final no teste.

Saídas
------
results/selecao_cv_folds.csv    F1 macro e recall da classe crítica por fold e candidato
results/selecao_cv_resumo.csv   média, desvio e vitórias pareadas por candidato
results/selecao_cv.json         resumo legível + IC bootstrap do teste
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold

from modeling import CANDIDATES, make_model
from pipeline import ROOT, TASKS

N_FOLDS = 6  # 24 famílias de desenvolvimento -> 4 famílias por fold
N_BOOTSTRAP = 2000
SEED = 42

# Classe cujo erro é mais caro para o negócio em cada tarefa.
CRITICAL_CLASS = {
    "sentimento": "negativo",  # insatisfação não detectada
    "churn": "risco",  # cliente em saída sem ação de retenção
    "oportunidade": "oportunidade",  # expansão perdida
}


def cross_validate(dev):
    """Treina cada candidato em 5/6 das famílias e avalia nas famílias restantes."""
    splitter = GroupKFold(n_splits=N_FOLDS)
    rows = []
    for task in TASKS:
        critical = CRITICAL_CLASS[task]
        folds = splitter.split(dev.texto, dev[task], groups=dev.familia)
        for fold, (train_idx, test_idx) in enumerate(folds, start=1):
            train, held_out = dev.iloc[train_idx], dev.iloc[test_idx]
            for name in CANDIDATES:
                model = make_model(name).fit(train.texto, train[task])
                pred = model.predict(held_out.texto)
                rows.append(
                    {
                        "tarefa": task,
                        "fold": fold,
                        "familias_avaliadas": ",".join(sorted(held_out.familia.unique())),
                        "modelo": name,
                        "f1_macro": f1_score(held_out[task], pred, average="macro"),
                        f"recall_{critical}": recall_score(
                            held_out[task], pred, labels=[critical], average="macro"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize(folds, selected):
    """Média, desvio e comparação pareada (mesmos folds) contra o modelo entregue."""
    rows = []
    for task in TASKS:
        critical_col = f"recall_{CRITICAL_CLASS[task]}"
        task_folds = folds[folds.tarefa == task]
        pivot = task_folds.pivot(index="fold", columns="modelo", values="f1_macro")
        chosen = selected[task]
        for name in CANDIDATES:
            scores = task_folds[task_folds.modelo == name]
            diff = pivot[chosen] - pivot[name]
            rows.append(
                {
                    "tarefa": task,
                    "modelo": name,
                    "entregue": name == chosen,
                    "f1_media": scores.f1_macro.mean(),
                    "f1_desvio": scores.f1_macro.std(ddof=1),
                    "f1_min": scores.f1_macro.min(),
                    "recall_classe_critica_media": scores[critical_col].mean(),
                    "dif_media_entregue_menos_este": diff.mean() if name != chosen else 0.0,
                    "folds_entregue_maior_ou_igual": (
                        int((diff >= -1e-12).sum()) if name != chosen else N_FOLDS
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_test(models, test):
    """IC 95% do F1 macro no teste, reamostrando transcrições com reposição."""
    rng = np.random.default_rng(SEED)
    result = {}
    for task in TASKS:
        y = test[task].to_numpy()
        pred = models[task].predict(test.texto)
        samples = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, len(y), len(y))
            samples.append(f1_score(y[idx], pred[idx], average="macro"))
        low, high = np.percentile(samples, [2.5, 97.5])
        result[task] = {
            "f1_macro": round(float(f1_score(y, pred, average="macro")), 3),
            "ic95": [round(float(low), 3), round(float(high), 3)],
        }
    return result


def verdict(summary, selected):
    """Texto curto por tarefa, para o notebook e o relatório."""
    lines = {}
    for task in TASKS:
        task_rows = summary[summary.tarefa == task].set_index("modelo")
        chosen = selected[task]
        best = task_rows.f1_media.idxmax()
        gap = task_rows.f1_media.max() - task_rows.loc[chosen, "f1_media"]
        spread = task_rows.loc[chosen, "f1_desvio"]
        if gap <= 1e-9:
            tied = [
                m
                for m in task_rows.index
                if task_rows.f1_media.max() - task_rows.f1_media[m] <= 1e-9
            ]
            if len(tied) > 1:
                text = (
                    f"Empate em F1 médio ({task_rows.f1_media.max():.3f}) entre {', '.join(tied)}; "
                    f"{chosen} é mantido pela regra de desempate fixada antes da avaliação."
                )
            else:
                text = f"{chosen} também tem a maior média na validação cruzada."
        elif gap <= spread:
            text = (
                f"{best} tem média {gap:.3f} maior, diferença menor que o desvio entre folds "
                f"({spread:.3f}); a diferença está dentro da variação entre folds, então as escolhas são equivalentes nesta amostra."
            )
        else:
            text = (
                f"{best} supera {chosen} em {gap:.3f}, acima do desvio entre folds "
                f"({spread:.3f}); vale reavaliar a escolha com dados reais."
            )
        lines[task] = text
    return lines


def main():
    corpus = pd.read_csv(ROOT / "data/corpus.csv")
    dev = corpus[corpus.particao.isin(["treino", "validacao"])].reset_index(drop=True)
    test = corpus[corpus.particao == "teste"]
    bundle = joblib.load(ROOT / "results/modelos.joblib")  # gerado localmente por train.py
    selected = bundle["selected"]

    folds = cross_validate(dev)
    summary = summarize(folds, selected)
    output = {
        "protocolo": (
            f"GroupKFold com {N_FOLDS} folds sobre {dev.familia.nunique()} famílias de "
            "desenvolvimento (treino + validação). O teste não participa. "
            "Diagnóstico de robustez: não altera o modelo entregue."
        ),
        "modelos_entregues": selected,
        "classe_critica": CRITICAL_CLASS,
        "conclusao": verdict(summary, selected),
        "teste_bootstrap": bootstrap_test(bundle["models"], test),
        "n_bootstrap": N_BOOTSTRAP,
    }

    out = ROOT / "results"
    folds.round(4).to_csv(out / "selecao_cv_folds.csv", index=False)
    summary.round(4).to_csv(out / "selecao_cv_resumo.csv", index=False)
    (out / "selecao_cv.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.round(3).to_string(index=False))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
