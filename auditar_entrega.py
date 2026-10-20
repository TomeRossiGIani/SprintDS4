"""Audita evidências dos sete critérios, sem atribuir nota acadêmica."""

import json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from pipeline import Analyzer, ROOT, TASKS, normalize
from train import metrics

CRITERIA = [
    ("pipeline", "Pipeline completo de processamento", 15),
    ("modelo", "Modelo final selecionado e justificativa", 15),
    ("scoring", "Sistema de scoring/indicadores", 20),
    ("prototipo", "Protótipo funcional", 20),
    ("validacao", "Validação com novas transcrições", 15),
    ("recomendacoes", "Recomendações de negócio", 10),
    ("limites", "Limitações e próximos passos", 5),
]


def main():
    checks = {key: [] for key, _, _ in CRITERIA}

    def check(area, name, condition):
        checks[area].append({"verificacao": name, "aprovado": bool(condition)})

    a = Analyzer()
    m = json.loads((ROOT / "results/metricas.json").read_text())
    protocol = json.loads((ROOT / "results/protocolo.json").read_text())
    df = pd.read_csv(ROOT / "data/corpus.csv")
    test = df[df.particao == "teste"]
    val = df[df.particao == "validacao"]
    standard = json.loads((ROOT / "data/novas_transcricoes.json").read_text())
    challenging = json.loads((ROOT / "data/casos_desafio.json").read_text())
    stored = json.loads((ROOT / "results/novas_resultados.json").read_text())
    results = {r["id"]: a.analyze(r["texto"]) for r in standard + challenging}
    check(
        "pipeline",
        "Entrada nova gera produtos, classes, scores, termos, evidências e ações",
        all(
            set(["produtos", "classificacoes", "scores", "termos", "evidencias", "acoes"]) <= set(r)
            for r in results.values()
        ),
    )
    check(
        "pipeline",
        "Evidências preservam os trechos exatos e suas posições",
        all(
            row["texto"][e["inicio_caractere"] : e["inicio_caractere"] + len(e["trecho"])]
            == e["trecho"]
            for row in standard + challenging
            for e in results[row["id"]]["evidencias"]
        ),
    )
    check(
        "pipeline",
        "Famílias inteiras pertencem a uma única partição",
        df.groupby("familia").particao.nunique().max() == 1,
    )
    check(
        "modelo",
        "Hashes de dados e candidatos correspondem ao protocolo",
        all(
            hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == protocol[k]
            for p, k in [
                ("data/corpus.csv", "corpus_sha256"),
                ("data/novas_transcricoes.json", "novas_sha256"),
                ("modeling.py", "modeling_sha256"),
            ]
        ),
    )
    for task in TASKS:
        model = a.bundle["models"][task]
        candidates = [r for r in m["comparacao_validacao"] if r["tarefa"] == task]
        selected = max(
            candidates,
            key=lambda r: (r["f1_macro"], r["recall_macro"], r["modelo"] == "Logistica_palavras"),
        )
        check(
            "modelo",
            task + ": seleção obedece à validação",
            selected["modelo"] == a.bundle["selected"][task],
        )
        check(
            "modelo",
            task + ": métricas de validação conferem com o modelo entregue",
            all(
                np.isclose(v, selected[k])
                for k, v in metrics(val[task], model.predict(val.texto)).items()
            ),
        )
        check(
            "modelo",
            task + ": métricas de teste recalculadas conferem com as publicadas",
            all(
                np.isclose(v, m["avaliacao"][task]["teste"][k])
                for k, v in metrics(test[task], model.predict(test.texto)).items()
            ),
        )
        matrix = confusion_matrix(
            test[task], model.predict(test.texto), labels=m["avaliacao"][task]["classes"]
        ).tolist()
        check(
            "modelo",
            task + ": matriz corresponde às predições",
            matrix == m["avaliacao"][task]["matriz_confusao"],
        )
    check(
        "scoring",
        "Todos os scores estão entre 0 e 100",
        all(0 <= v <= 100 for r in results.values() for v in r["scores"].values()),
    )
    check(
        "scoring",
        "Distribuições dos classificadores somam aproximadamente um",
        all(
            abs(sum(p.values()) - 1) < 1e-5
            for r in results.values()
            for p in r["probabilidades_modelo"].values()
        ),
    )
    check(
        "scoring",
        "Prioridade respeita a fórmula documentada",
        all(
            abs(
                r["scores"]["prioridade"]
                - (
                    0.55 * r["scores"]["risco_churn"]
                    + 0.30 * r["scores"]["oportunidade"]
                    + 0.15 * r["scores"]["sentimento_negativo"]
                )
            )
            < 0.11
            for r in results.values()
        ),
    )
    check(
        "scoring",
        "Auditoria de sensibilidade e Brier está incluída",
        all(
            (ROOT / p).is_file()
            for p in ["results/sensibilidade_validacao.csv", "results/auditoria.json"]
        ),
    )
    log = (ROOT / "results/testes.txt").read_text()
    browser = (ROOT / "results/teste_navegador.txt").read_text()
    check(
        "prototipo",
        "Testes Python/HTTP executados e aprovados",
        "Ran 20 tests" in log and "\nOK\n" in log,
    )
    check(
        "prototipo",
        "Navegador aprovou exemplo, TXT, análise, JSON, mobile e entrada vazia",
        browser.startswith("PASS:") and "zero erros JavaScript" in browser,
    )
    check(
        "prototipo",
        "Interface e CLI estão incluídas",
        all(
            (ROOT / p).is_file()
            for p in ["app.py", "cli.py", "tests/run_browser.py", "tests/browser_smoke.cjs"]
        ),
    )
    check(
        "validacao",
        "24 exemplos inéditos e 8 desafios disponíveis",
        len(standard) == 24 and len(challenging) == 8,
    )
    check(
        "validacao",
        "Exemplos novos não são duplicatas do corpus",
        not set(df.texto.map(normalize)) & {normalize(r["texto"]) for r in standard + challenging},
    )
    check(
        "validacao",
        "Predições armazenadas coincidem com a inferência entregue",
        all(results[r["id"]]["classificacoes"] == r["resultado"]["classificacoes"] for r in stored),
    )
    for task in TASKS:
        measured = metrics(
            [r[task] for r in standard],
            [results[r["id"]]["classificacoes"][task] for r in standard],
        )
        check(
            "validacao",
            task + ": métricas inéditas conferem",
            all(np.isclose(v, m["novas_metricas"][task][k]) for k, v in measured.items()),
        )
    check(
        "recomendacoes",
        "Ações informam responsável, prazo, descrição e score",
        all(
            set(["tipo", "responsavel", "prazo_horas", "acao", "score"]) <= set(action)
            and action["prazo_horas"] > 0
            for r in results.values()
            for action in r["acoes"]
        ),
    )
    check(
        "recomendacoes",
        "Risco acima do corte tem ação de retenção e responsável CS",
        all(
            any(x["tipo"] == "retencao" and x["responsavel"] == "CS" for x in r["acoes"])
            for r in results.values()
            if r["probabilidades_modelo"]["churn"]["risco"] >= 0.5
        ),
    )
    check(
        "recomendacoes",
        "Oportunidade acima do corte tem ação comercial",
        all(
            any(x["tipo"] == "expansao" for x in r["acoes"])
            for r in results.values()
            if r["probabilidades_modelo"]["oportunidade"]["oportunidade"] >= 0.5
        ),
    )
    check(
        "recomendacoes",
        "Retenção antecede expansão quando os sinais coexistem",
        all(
            [x["tipo"] for x in r["acoes"]].index("retencao")
            < [x["tipo"] for x in r["acoes"]].index("expansao")
            for r in results.values()
            if {"retencao", "expansao"} <= {x["tipo"] for x in r["acoes"]}
        ),
    )
    readme = (ROOT / "README.md").read_text().lower()
    check(
        "limites",
        "Origem sintética, calibração, erros e dados reais estão explicitados",
        all(term in readme for term in ["sintética", "calibrad", "erros", "dados reais"]),
    )
    check(
        "limites",
        "Revisão humana é exigida em toda análise",
        all(r["revisao_humana"] for r in results.values()),
    )
    check(
        "limites",
        "Casos de desenvolvimento conhecido são identificados separadamente",
        (ROOT / "data/casos_desenvolvimento_legacy.json").is_file(),
    )
    areas = [
        {
            "id": key,
            "criterio": title,
            "peso_oficial": weight,
            "evidencias_conferidas": all(c["aprovado"] for c in checks[key]),
            "verificacoes": checks[key],
        }
        for key, title, weight in CRITERIA
    ]
    output = {
        "criterios": areas,
        "verificacoes_total": sum(len(v) for v in checks.values()),
        "verificacoes_aprovadas": sum(c["aprovado"] for v in checks.values() for c in v),
        "criterios_evidenciados": sum(x["evidencias_conferidas"] for x in areas),
        "aviso": "Auditoria de integridade e cobertura das evidências. Não atribui nota nem substitui a correção do professor.",
    }
    (ROOT / "results/aceitacao_rubrica.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {k: v for k, v in output.items() if k != "criterios"}, ensure_ascii=False, indent=2
        )
    )
    return 0 if all(x["evidencias_conferidas"] for x in areas) else 1


if __name__ == "__main__":
    sys.exit(main())
