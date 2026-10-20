"""Reexecuta as células Python e salva saídas reais em um notebook padrão nbformat 4."""

import base64, contextlib, hashlib, io, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = [
    (
        "markdown",
        "# Kaivi AI · Sprint 4\nAnálise reproduzível de dados sintéticos. Execute antes `python train.py`, `python evaluate_extra.py` e `python selecao_cv.py`.\nOs testes por famílias e as transcrições novas são avaliações distintas.",
    ),
    (
        "code",
        "from pathlib import Path\nimport json, pandas as pd\nroot = Path.cwd()\nmetrics = json.loads((root / 'results/metricas.json').read_text())\ndf = pd.read_csv(root / 'data/corpus.csv')\nprint(df.groupby('particao').size().to_string())\nprint('Total:', len(df))",
    ),
    (
        "code",
        "for task in ['sentimento','churn','oportunidade']:\n    print('\\n', task)\n    print(pd.crosstab(df.particao,df[task]).to_string())",
    ),
    (
        "markdown",
        "## Proteção contra vazamento\nCada família inteira fica em uma partição. Nenhuma transcrição nova é usada para treino. TF-IDF ajustado apenas no treino.",
    ),
    (
        "code",
        "print(df.groupby('familia').particao.nunique().to_string())\nassert df.groupby('familia').particao.nunique().max() == 1\nassert not df.texto.duplicated().any()",
    ),
    (
        "code",
        "print(pd.read_csv(root/'results/comparacao_validacao.csv').round(3).to_string(index=False))",
    ),
    (
        "markdown",
        "## Por que estes modelos\n"
        "- **Métrica de seleção: F1 macro.** As classes estão balanceadas e todas importam; F1 macro pesa igualmente cada classe e equilibra precision e recall. Accuracy sozinha esconderia uma classe ignorada.\n"
        "- **Churn: recall da classe risco é a métrica crítica.** Deixar passar um cliente em saída custa mais do que um alerta falso, que é filtrado pela revisão humana. O modelo entregue recupera todos os casos de risco do teste (recall 1,000) com precision 0,889.\n"
        "- **Oportunidade:** a Regressão Logística híbrida (palavras + caracteres) teve a maior F1 na validação, coerente com variações morfológicas como expandir/expansão.\n"
        "- **Sentimento:** empate perfeito na validação; a regra de desempate fixada antes do teste favorece a Regressão Logística de palavras, mais simples e interpretável.\n\n"
        "Como a validação tem só 4 famílias, conferimos a escolha com validação cruzada agrupada por família (abaixo). O teste não participa dessa conferência.",
    ),
    (
        "code",
        "cv = json.loads((root/'results/selecao_cv.json').read_text())\n"
        "resumo = pd.read_csv(root/'results/selecao_cv_resumo.csv')\n"
        "print(cv['protocolo'], '\\n')\n"
        "print(resumo[['tarefa','modelo','entregue','f1_media','f1_desvio','f1_min','recall_classe_critica_media']].round(3).to_string(index=False))\n"
        "print()\n"
        "for tarefa, texto in cv['conclusao'].items():\n"
        "    print(f'{tarefa}: {texto}')\n"
        "print('\\nIC 95% bootstrap do F1 macro no teste:')\n"
        "for tarefa, v in cv['teste_bootstrap'].items():\n"
        "    print(f\"  {tarefa}: {v['f1_macro']:.3f} [{v['ic95'][0]:.3f} - {v['ic95'][1]:.3f}]\")",
    ),
    (
        "code",
        "for task, result in metrics['avaliacao'].items():\n    print('\\n',task,result['modelo'])\n    print('Teste:',result['teste'])\n    print('Baseline:',result['baseline_teste'])\n    print('Matriz (linhas reais / colunas preditas):',result['classes'])\n    print(result['matriz_confusao'])",
    ),
    (
        "markdown",
        "## Matrizes de confusão (teste)\nLinhas são classes reais; colunas são predições.",
    ),
    (
        "code",
        "__imagens__ = [root/f'results/matriz_{t}.png' for t in ['sentimento','churn','oportunidade']]\n"
        "print('Matrizes geradas por train.py a partir do conjunto de teste.')",
    ),
    (
        "code",
        "print('Novas transcrições:', metrics['novas_n'])\nprint(json.dumps(metrics['novas_metricas'], indent=2))\nerrors = pd.read_csv(root/'results/erros.csv')\nprint(errors.groupby(['conjunto','tarefa']).size().to_string())\nprint(errors[errors.conjunto == 'novas'][['id','tarefa','real','predito']].to_string(index=False))",
    ),
    (
        "code",
        "from pipeline import Analyzer\nexample = (root/'data/exemplo.txt').read_text()\nprint(json.dumps(Analyzer().analyze(example),ensure_ascii=False,indent=2))",
    ),
    (
        "markdown",
        "## Recomendações de negócio\n"
        "1. **Usar como triagem pós-reunião, não como decisão automática.** Cada análise gera ações com responsável e prazo: risco de churn vai para CS em 24h, oportunidade para o vendedor em 48h, insatisfação para o suporte em 24h. Quando risco e oportunidade coexistem, a retenção vem primeiro.\n"
        "2. **Rodar um piloto assistido antes de investir em escala.** Com 5 vendedores por 8 a 12 semanas, medir tempo de revisão com e sem o Kaivi, alertas confirmados, riscos perdidos encontrados em auditoria e aceitação das ações.\n"
        "3. **Ajustar o limiar de churn pelo custo do erro.** Reduzir o corte de 0,50 para 0,30 aumenta o recall de risco na validação, ao custo de mais alertas. A escolha depende da capacidade do time de CS e deve ser feita com dados reais.\n"
        "4. **Coletar transcrições reais autorizadas e anonimizadas,** com dupla anotação, antes de retreinar. É a principal condição para expandir.\n"
        "5. **Critério para integrar ao CRM:** F1 macro ≥ 0,80 por tarefa e recall de risco ≥ 0,85 em avaliação externa por cliente e período. São metas propostas, ainda não comprovadas.",
    ),
    (
        "markdown",
        "## Conclusão\n"
        "O pipeline e o protótipo funcionam de ponta a ponta. No teste com famílias nunca vistas, o F1 macro fica próximo de 0,94 nas três tarefas, bem acima do baseline, e a validação cruzada por família mostra quanto a escolha dos modelos depende da partição usada. "
        "A validação de churn mais baixa mostra sensibilidade à formulação. Os resultados vêm de dados sintéticos e não comprovam desempenho em reuniões reais; toda recomendação exige revisão humana. "
        "Próximo passo: corpus real autorizado, dupla anotação e avaliação externa por cliente e período.",
    ),
]
cells.insert(
    -2,
    (
        "code",
        "audit = json.loads((root/'results/auditoria.json').read_text())\nprint(json.dumps(audit,ensure_ascii=False,indent=2))\nprint(pd.read_csv(root/'results/estabilidade_familias.csv').round(3).to_string(index=False))\nprint(pd.read_csv(root/'results/sensibilidade_validacao.csv').round(3).to_string(index=False))",
    ),
)
cells.insert(
    -2,
    (
        "code",
        "acceptance = json.loads((root/'results/aceitacao_rubrica.json').read_text())\nclean = json.loads((root/'results/reproducao_limpa.json').read_text())\nprint('Critérios evidenciados:', acceptance['criterios_evidenciados'])\nprint('Verificações aprovadas:', acceptance['verificacoes_aprovadas'], '/', acceptance['verificacoes_total'])\nprint('Reprodução em ambiente novo:', clean['todos_aprovados'])",
    ),
)


def main():
    os.chdir(ROOT)
    scope = {}
    out = []
    n = 0
    for typ, source in cells:
        cell = {
            "id": hashlib.sha256(source.encode()).hexdigest()[:12],
            "cell_type": typ,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if typ == "code":
            n += 1
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(compile(source, "notebook", "exec"), scope)
            outputs = [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": buffer.getvalue().splitlines(keepends=True),
                }
            ]
            # Imagens embutidas em base64 para renderizar no GitHub sem depender de caminhos.
            for image in scope.pop("__imagens__", []):
                outputs.append(
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {
                            "image/png": base64.b64encode(Path(image).read_bytes()).decode(),
                            "text/plain": [f"<{Path(image).name}>"],
                        },
                    }
                )
            cell.update(execution_count=n, outputs=outputs)
        out.append(cell)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": out,
    }
    (ROOT / "analise.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=2))
    print(f"Notebook executado: {n} células de código sem erros.")


if __name__ == "__main__":
    main()
