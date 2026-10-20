"""Seleção em validação; teste e novas transcrições não participam do ajuste."""

import json, platform, hashlib
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from pipeline import ROOT, TASKS, normalize, Analyzer


def metrics(y, p):
    pr, re, f, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(
        accuracy=float(accuracy_score(y, p)),
        precision_macro=float(pr),
        recall_macro=float(re),
        f1_macro=float(f),
    )


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "protocolo.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "corpus_sha256": hashlib.sha256(
                    (ROOT / "data/corpus.csv").read_bytes()
                ).hexdigest(),
                "modeling_sha256": hashlib.sha256((ROOT / "modeling.py").read_bytes()).hexdigest(),
                "novas_sha256": hashlib.sha256(
                    (ROOT / "data/novas_transcricoes.json").read_bytes()
                ).hexdigest(),
                "politica": "Legado da primeira versão é desenvolvimento. Teste novo e novas transcrições não ajustam candidatos. TF-IDF apenas treino.",
            },
            indent=2,
        )
    )
    df = pd.read_csv(ROOT / "data/corpus.csv")
    parts = {s: df[df.particao == s] for s in ["treino", "validacao", "teste"]}
    for a, b in [("treino", "validacao"), ("treino", "teste"), ("validacao", "teste")]:
        assert not set(parts[a].familia) & set(parts[b].familia)
        assert not set(parts[a].texto.map(normalize)) & set(parts[b].texto.map(normalize))
    from modeling import make_model, CANDIDATES

    candidates = CANDIDATES
    comparison = []
    models = {}
    selected = {}
    reports = {}
    errors = []
    for task in TASKS:
        fitted = {}
        for name in candidates:
            model = make_model(name)
            model.fit(parts["treino"].texto, parts["treino"][task])
            fitted[name] = model
            comparison.append(
                {
                    "tarefa": task,
                    "modelo": name,
                    **metrics(parts["validacao"][task], model.predict(parts["validacao"].texto)),
                }
            )
        best = max(
            [r for r in comparison if r["tarefa"] == task],
            key=lambda r: (r["f1_macro"], r["recall_macro"], r["modelo"] == "Logistica_palavras"),
        )
        selected[task] = best["modelo"]
        models[task] = fitted[best["modelo"]]
        # Não retreinar: scores de validação e teste correspondem ao mesmo artefato.
        pred = models[task].predict(parts["teste"].texto)
        labels = list(models[task].classes_)
        baseline = DummyClassifier(strategy="most_frequent").fit(
            np.zeros((len(parts["treino"]), 1)), parts["treino"][task]
        )
        reports[task] = {
            "modelo": best["modelo"],
            "teste": metrics(parts["teste"][task], pred),
            "baseline_teste": metrics(
                parts["teste"][task], baseline.predict(np.zeros((len(pred), 1)))
            ),
            "classes": labels,
            "matriz_confusao": confusion_matrix(parts["teste"][task], pred, labels=labels).tolist(),
            "relatorio_por_classe": classification_report(
                parts["teste"][task], pred, output_dict=True, zero_division=0
            ),
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(
            parts["teste"][task], pred, display_labels=labels, cmap="Blues", ax=ax, colorbar=False
        )
        ax.set(title=task, xlabel="Predito", ylabel="Real")
        fig.tight_layout()
        fig.savefig(out / f"matriz_{task}.png", dpi=160)
        plt.close(fig)
        for (_, row), p in zip(parts["teste"].iterrows(), pred):
            if p != row[task]:
                errors.append(
                    {
                        "conjunto": "teste",
                        "id": row.id,
                        "tarefa": task,
                        "real": row[task],
                        "predito": p,
                        "texto": row.texto,
                    }
                )
    from pipeline import client_text

    term_vectorizer = TfidfVectorizer(
        preprocessor=client_text, ngram_range=(1, 2), sublinear_tf=True
    ).fit(parts["treino"].texto)
    joblib.dump(
        {"models": models, "selected": selected, "seed": 42, "term_vectorizer": term_vectorizer},
        out / "modelos.joblib",
    )
    analyzer = Analyzer()
    novel = json.loads((ROOT / "data/novas_transcricoes.json").read_text())
    results = []
    for row in novel:
        result = analyzer.analyze(row["texto"])
        results.append(
            {
                "id": row["id"],
                "esperado": {t: row[t] for t in TASKS},
                "resultado": result,
                "texto": row["texto"],
            }
        )
        for t in TASKS:
            if result["classificacoes"][t] != row[t]:
                errors.append(
                    {
                        "conjunto": "novas",
                        "id": row["id"],
                        "tarefa": t,
                        "real": row[t],
                        "predito": result["classificacoes"][t],
                        "texto": row["texto"],
                    }
                )
    novel_metrics = {
        t: metrics([r[t] for r in novel], [r["resultado"]["classificacoes"][t] for r in results])
        for t in TASKS
    }
    pd.DataFrame(comparison).to_csv(out / "comparacao_validacao.csv", index=False)
    pd.DataFrame(errors, columns=["conjunto", "id", "tarefa", "real", "predito", "texto"]).to_csv(
        out / "erros.csv", index=False
    )
    (out / "novas_resultados.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    summary = {
        "origem": "DADOS SINTÉTICOS, não são reuniões reais TOTVS",
        "particoes": {s: len(p) for s, p in parts.items()},
        "distribuicao": {t: df.groupby(["particao", t]).size().to_dict().__str__() for t in TASKS},
        "selecao": "Maior F1 macro na validação; desempate recall macro, Logistica_palavras e ordem fixa dos candidatos. Sem ajuste após teste.",
        "comparacao_validacao": comparison,
        "avaliacao": reports,
        "novas_metricas": novel_metrics,
        "novas_n": len(novel),
        "total_erros": len(errors),
        "ambiente": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        "hash_corpus_sha256": hashlib.sha256((ROOT / "data/corpus.csv").read_bytes()).hexdigest(),
    }
    (out / "metricas.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "modelos": selected,
                "teste": {t: reports[t]["teste"] for t in TASKS},
                "novas": novel_metrics,
                "erros": len(errors),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
