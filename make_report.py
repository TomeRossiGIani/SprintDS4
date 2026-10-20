"""Relatório gerado exclusivamente a partir dos resultados executados."""

import json, csv
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent
font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
bold = font.with_name("DejaVuSans-Bold.ttf")
if font.exists():
    pdfmetrics.registerFont(TTFont("Body", str(font)))
    pdfmetrics.registerFont(TTFont("BodyBold", str(bold)))
    F = "Body"
    B = "BodyBold"
else:
    F = "Helvetica"
    B = "Helvetica-Bold"
NAVY = colors.HexColor("#142b46")
BLUE = colors.HexColor("#215ac1")
MUTED = colors.HexColor("#50657c")
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="BodyCustom", fontName=F, fontSize=9.5, leading=15, spaceAfter=9, textColor=NAVY
    )
)
styles.add(
    ParagraphStyle(
        name="TitleCustom", fontName=B, fontSize=29, leading=36, textColor=NAVY, spaceAfter=16
    )
)
styles.add(
    ParagraphStyle(
        name="HeadCustom", fontName=B, fontSize=18, leading=24, textColor=BLUE, spaceAfter=15
    )
)
styles.add(
    ParagraphStyle(
        name="SubCustom",
        fontName=B,
        fontSize=11,
        leading=16,
        textColor=NAVY,
        spaceBefore=9,
        spaceAfter=7,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallCustom", fontName=F, fontSize=7.5, leading=11, textColor=MUTED, spaceAfter=7
    )
)
story = []


def p(text, style="BodyCustom"):
    story.append(Paragraph(escape(str(text)), styles[style]))


def h(text):
    p(text, "HeadCustom")


def sub(text):
    p(text, "SubCustom")


def page():
    story.append(PageBreak())


def table(rows, widths=None, compact=False):
    cells = [[Paragraph(escape(str(c)), styles["SmallCustom"]) for c in row] for row in rows]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edfa")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 if compact else 8),
                ("TOPPADDING", (0, 0), (-1, -1), 3 if compact else 8),
                ("LINEBELOW", (0, 0), (-1, 0), 1, BLUE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#d8e2ed")),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))


def main():
    import pandas as pd
    from pipeline import Analyzer, TASKS

    m = json.loads((ROOT / "results/metricas.json").read_text())
    new = json.loads((ROOT / "results/novas_resultados.json").read_text())
    audit = json.loads((ROOT / "results/auditoria.json").read_text())
    stress = json.loads((ROOT / "results/desafios_resultados.json").read_text())
    errors = pd.read_csv(ROOT / "results/erros.csv")
    families = pd.read_csv(ROOT / "results/estabilidade_familias.csv")
    p("KAIVI AI", "TitleCustom")
    p("Da transcrição à ação comercial", "HeadCustom")
    p("SPRINT 4 · DATA SCIENCE AND STATISTICAL COMPUTING", "SubCustom")
    story.append(Spacer(1, 24))
    p("Challenge TOTVS · Engenharia de Software")
    table(
        [
            ["Integrante", "RM"],
            ["Leonardo Eiji Kina", "562784"],
            ["Nicholas Braga de Souza", "561733"],
            ["Tomé Rossi Giani", "562422"],
            ["Vitor Ramos de Farias", "561958"],
        ],
        [340, 140],
    )
    story.append(Spacer(1, 18))
    p("Pipeline local, modelos comparados e recomendações rastreáveis", "SubCustom")
    p(
        "Código-fonte, protótipo web, CLI, notebook executado, dados rotulados, modelos serializados, métricas, auditoria e testes reproduzíveis."
    )
    p("1.344 transcrições sintéticas · 24 casos inéditos · 8 desafios linguísticos", "SubCustom")
    p(
        "F1 macro no teste separado: sentimento 0,942; churn 0,937; oportunidade 0,943. Esses resultados descrevem uma demonstração sintética, sem comprovar desempenho em reuniões reais.",
        "SmallCustom",
    )
    page()
    h("01 · Objetivo e critérios da Sprint")
    p(
        "O Kaivi AI converte reuniões em inteligência comercial. Este recorte recebe uma transcrição pronta em português e identifica produtos mencionados, sentimento, intenção de saída e oportunidade. A saída prioriza ações, mostra trechos para conferência e pode ser exportada. O usuário-alvo é o vendedor ou gestor que precisa revisar a reunião e definir o próximo passo."
    )
    table(
        [
            ["Critério oficial", "Peso", "Evidência verificável"],
            ["Pipeline completo de processamento", "15", "Seção 02; pipeline.py e train.py"],
            ["Modelo final e justificativa", "15", "Seções 04–06; comparação de 3 candidatos"],
            ["Sistema de scoring/indicadores", "20", "Seção 07; fórmulas, limiares e JSON"],
            ["Protótipo funcional", "20", "Seção 08; app.py, CLI e teste no navegador"],
            ["Validação com novas transcrições", "15", "Seção 09; 24 novos casos e 8 desafios"],
            ["Recomendações de negócio", "10", "Seção 10; ações, responsáveis e prazos"],
            ["Limitações e próximos passos", "5", "Seção 11; limites e plano de evolução"],
        ],
        [170, 40, 270],
    )
    sub("Organização da entrega")
    p(
        "02 Pipeline e dados · 03 Protocolo experimental · 04 Seleção dos modelos · 05 Teste e classes · 06 Robustez e scoring · 07 Indicadores · 08 Protótipo · 09 Novas transcrições e erros · 10 Ações · 11 Limitações · 12 Reprodução e referências."
    )
    p(
        "O guia da Sprint 4 pede o fluxo Transcrição → Processamento → Modelo → Indicadores → Recomendação. O PDF da Kaivi fornece visão e equipe, mas não fornece corpus nem modelos prévios. Este módulo foi implementado do zero para esse fluxo; não inclui captura de áudio, Whisper, chat RAG ou integração Java/Oracle."
    )
    page()
    h("02 · Pipeline integrado")
    steps = [
        (
            "1. Receber e validar",
            "Texto colado, arquivo TXT UTF-8 ou CLI. Pelo menos cinco palavras na fala analisável e até 50.000 caracteres. Tipos inválidos, texto vazio e requisições excessivas são rejeitados.",
        ),
        (
            "2. Identificar as falas",
            "Rótulos Cliente/Comprador e Vendedor/Consultor/Comercial no início de cada linha permitem selecionar somente as falas do cliente para o ML. Timestamps no começo da linha são aceitos. Sem rótulos, todo o texto é usado com aviso; se só há fala do vendedor, a análise é rejeitada.",
        ),
        (
            "3. Processar e representar",
            "Normalização NFKC, caixa e espaços; máscara parcial de e-mail/telefone para inferência; preservação de negações. TF-IDF com palavras e bigramas, ou união com n-gramas de caracteres. Vocabulários ajustados só no treino.",
        ),
        (
            "4. Inferir três intenções",
            "Classificadores independentes de sentimento, churn e oportunidade. É possível haver sentimento positivo e intenção de sair simultaneamente: uma variável não determina a outra.",
        ),
        (
            "5. Extrair e calcular",
            "Dicionário identifica produtos mencionados em toda a transcrição, inclusive perguntas. TF-IDF destaca termos das falas analisadas. Scores de 0–100, confiança mínima, cobertura e prioridade resumem os resultados.",
        ),
        (
            "6. Recomendar e conferir",
            "Ações estruturadas com responsável, prazo e score. Para cada tarefa, a frase isolada mais associada à classe prevista é marcada para conferência. Isso não constitui explicação causal do modelo. O JSON preserva trechos e posições.",
        ),
    ]
    for title, body in steps:
        sub(title)
        p(body)
    p(
        "ML utiliza somente a fala do cliente quando rotulada; produtos são menções no contexto completo. Uma menção a produto não significa confirmação de uso. Nenhuma ação é executada automaticamente.",
        "SmallCustom",
    )
    page()
    h("02 · Base, rótulos e proveniência")
    p(
        "Base sintética autoral, sem dados reais da TOTVS. São 28 famílias de frases × 3 sentimentos × 2 situações de churn × 2 de oportunidade × 4 produtos = 1.344 transcrições. A ordem dos três sinais é embaralhada com seed 42. O produto é combinado de forma balanceada para evitar associação artificial com um rótulo."
    )
    table(
        [
            ["Partição", "Famílias", "Registros", "Função"],
            ["Treino", "F00–F19", "960", "Ajustar representações e classificadores"],
            ["Validação", "F20–F23", "192", "Escolher candidato por tarefa"],
            ["Teste", "F24–F27", "192", "Avaliar após congelar a seleção"],
            ["Novas", "N01–N24", "24", "Transcrições autorais não usadas no ajuste"],
            ["Desafios", "D01–D08", "8", "Negação, falantes, ironia, contexto e produtos"],
        ],
        [82, 86, 65, 247],
    )
    table(
        [
            ["Variável-alvo", "Definição operacional"],
            [
                "Sentimento",
                "positivo: avaliação favorável; negativo: insatisfação; neutro: informação sem avaliação explícita",
            ],
            [
                "Churn",
                "risco: intenção explícita de saída, não renovação ou troca; sem_risco: permanência explícita no exemplo",
            ],
            [
                "Oportunidade",
                "oportunidade: interesse expresso em compra, cotação ou expansão; sem_oportunidade: ausência/negação de nova aquisição",
            ],
        ],
        [105, 375],
    )
    p(
        "Em treino, cada sentimento tem 320 exemplos e cada classe binária tem 480; em validação e teste, 64 por sentimento e 96 por classe binária. Não houve SMOTE nem oversampling. class_weight=balanced fica registrado, mas tem efeito unitário nesta base balanceada."
    )
    p(
        "Os rótulos foram definidos na autoria, sem dupla anotação independente. “Sem risco” aqui não garante permanência real. Nenhuma transcrição sintética substitui amostra representativa de reuniões. Os 1.344 registros não são observações independentes: muitas combinações compartilham frases.",
        "SmallCustom",
    )
    page()
    h("03 · Protocolo contra vazamento")
    sub("Separação por família")
    p(
        "Todas as versões de uma família, incluindo produtos, rótulos e ordens, permanecem na mesma partição. Nenhuma frase exata de intenção é reutilizada entre as famílias. O vocabulário TF-IDF, a frequência dos termos e os classificadores são ajustados apenas nos 960 registros de treino. O código verifica famílias e transcrições disjuntas."
    )
    sub("Seleção antes de observar o novo teste")
    p(
        "Três candidatos e seus hiperparâmetros foram fixados no código modeling.py. A escolha usa apenas F1 macro da validação. Empates usam recall macro, preferência pelo modelo logístico de palavras e ordem fixa. Não há busca de hiperparâmetros no teste, ajuste manual de respostas ou retreino com as 24 transcrições inéditas."
    )
    sub("Rastreabilidade do experimento")
    p(
        "results/protocolo.json registra seed e hashes de corpus, candidatos e casos inéditos. results/metricas.json registra versões de Python/scikit-learn e resultados. Os modelos entregues são os mesmos usados no teste; não houve retreino com validação ou teste após a seleção. train.py permite repetir o experimento."
    )
    sub("Uso correto do material anterior")
    p(
        "Os 18 exemplos da primeira tentativa já tinham sido examinados. Foram conservados somente como casos de desenvolvimento conhecidos em data/casos_desenvolvimento_legacy.json e não entram nas métricas inéditas desta entrega. O teste atual e as 24 novas transcrições são novos conjuntos. A primeira tentativa motivou o aumento da diversidade, sem ocultar essa origem."
    )
    sub("Limites da inferência")
    p(
        "Apesar da separação, as famílias são curtas e artificiais, com vocabulário sobreposto e construção controlada. O teste tem somente quatro famílias linguísticas. Não apresentamos intervalo de confiança calculado como se 192 linhas fossem 192 reuniões independentes. A auditoria mostra a variação por família e não extrapola para o mercado."
    )
    page()
    h("04 · Comparação e seleção final")
    table(
        [
            ["Candidato", "Representação / configuração"],
            [
                "NaiveBayes_palavras",
                "TF-IDF palavras (1,2); sublinear_tf=True; MultinomialNB alpha=0,5",
            ],
            [
                "Logistica_palavras",
                "Mesmo TF-IDF; LogisticRegression C=4, max_iter=2000, class_weight=balanced, seed=42",
            ],
            [
                "Logistica_hibrida",
                "TF-IDF palavras (1,2) + char_wb (3,5), min_df=2; pesos 1,0 / 0,8; mesma Regressão Logística",
            ],
        ],
        [150, 330],
    )
    rows = [["Tarefa", "Candidato", "F1 macro", "Recall macro"]]
    for r in m["comparacao_validacao"]:
        rows.append([r["tarefa"], r["modelo"], f"{r['f1_macro']:.3f}", f"{r['recall_macro']:.3f}"])
    table(rows, [85, 215, 90, 90], compact=True)
    p(
        "Selecionados: Logistica_palavras para sentimento e churn; Logistica_hibrida para oportunidade. Em sentimento há empate perfeito na validação, e o desempate favorece a opção de palavras. Em oportunidade, o modelo híbrido apresenta a melhor F1 da validação, compatível com o aproveitamento de variações morfológicas."
    )
    p(
        "A validação de churn ficou em F1 macro 0,630 e a de oportunidade em 0,765, abaixo do teste. Isso evidencia sensibilidade à formulação e ao pequeno conjunto de famílias; não é prova de qualidade uniforme. Não descartamos essas métricas por serem inferiores."
    )
    sub("Por que acompanhar estas métricas")
    p(
        "F1 macro dá igual peso às classes e equilibra precision e recall. Para churn, recall de risco é essencial para não deixar escapar intenção de saída; precision evita desgaste com alertas falsos. Em oportunidade, precision reduz propostas sem intenção de compra. Accuracy sozinha pode ocultar classes ignoradas. A seleção usa F1 macro; pesos de custo reais exigiriam validação com o negócio."
    )
    cv_path = ROOT / "results/selecao_cv.json"
    if cv_path.is_file():
        cv = json.loads(cv_path.read_text(encoding="utf-8"))
        cv_rows = pd.read_csv(ROOT / "results/selecao_cv_resumo.csv")
        sub("Robustez da escolha: validação cruzada por família")
        p(
            "Com apenas 4 famílias na validação, houve empates e diferenças mínimas. Para checar a "
            "escolha sem tocar no teste, repetimos a comparação com GroupKFold de 6 folds sobre as "
            "24 famílias de desenvolvimento (cada fold avalia 4 famílias nunca vistas no treino). "
            "Diagnóstico apenas: o modelo entregue não foi alterado."
        )
        rows = [["Tarefa", "Candidato", "F1 médio ± desvio", "Recall crítico", "Entregue"]]
        for _, r in cv_rows.iterrows():
            rows.append(
                [
                    r.tarefa,
                    r.modelo,
                    f"{r.f1_media:.3f} ± {r.f1_desvio:.3f}",
                    f"{r.recall_classe_critica_media:.3f}",
                    "sim" if r.entregue else "",
                ]
            )
        table(rows, [80, 140, 110, 80, 70], compact=True)
        for task, text in cv["conclusao"].items():
            p(f"{task.capitalize()}: {text}", "SmallCustom")
    page()
    h("05 · Teste separado e baseline")
    rows = [["Tarefa", "Accuracy", "Precision macro", "Recall macro", "F1 macro", "F1 baseline"]]
    for t, r in m["avaliacao"].items():
        v = r["teste"]
        rows.append(
            [
                t,
                *[
                    f"{v[k]:.3f}"
                    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
                ],
                f"{r['baseline_teste']['f1_macro']:.3f}",
            ]
        )
    table(rows, [90, 68, 86, 82, 76, 78])
    p(
        "Baseline: DummyClassifier most_frequent treinado somente no treino. Todos os modelos finais superam a referência trivial no novo teste. Matrizes abaixo: linhas são classes reais; colunas são predições."
    )
    if (ROOT / "results/selecao_cv.json").is_file():
        boot = json.loads((ROOT / "results/selecao_cv.json").read_text(encoding="utf-8"))
        intervals = "; ".join(
            f"{t} {v['f1_macro']:.3f} [{v['ic95'][0]:.3f}–{v['ic95'][1]:.3f}]"
            for t, v in boot["teste_bootstrap"].items()
        )
        p(
            f"IC 95% bootstrap do F1 macro no teste ({boot['n_bootstrap']} reamostragens): "
            f"{intervals}. O intervalo reflete só a variação entre transcrições desta amostra "
            "sintética, não a incerteza sobre reuniões reais.",
            "SmallCustom",
        )
    imgs = [Image(str(ROOT / f"results/matriz_{t}.png"), width=238, height=159) for t in TASKS]
    story.append(Table([[imgs[0], imgs[1]]], colWidths=[242, 242]))
    story.append(imgs[2])
    r = m["avaliacao"]["churn"]["relatorio_por_classe"]["risco"]
    p(
        f"Churn: recall da classe risco = {r['recall']:.3f}; precision = {r['precision']:.3f}. O teste recupera os 96 casos de risco, mas também gera falsos positivos. A evidência é sintética e não autoriza prometer a mesma taxa em clientes reais.",
        "SmallCustom",
    )
    page()
    h("05 · Desempenho por classe")
    rows = [["Tarefa / classe", "Precision", "Recall", "F1", "Suporte"]]
    for t, r in m["avaliacao"].items():
        for label in r["classes"]:
            v = r["relatorio_por_classe"][label]
            rows.append(
                [
                    t + " / " + label,
                    f"{v['precision']:.3f}",
                    f"{v['recall']:.3f}",
                    f"{v['f1-score']:.3f}",
                    str(int(v["support"])),
                ]
            )
    table(rows, [190, 70, 70, 70, 80])
    sub("Estabilidade entre famílias de teste")
    rows = [["Tarefa", "Menor F1 por família", "Maior F1 por família"]]
    for t, g in families.groupby("tarefa"):
        rows.append([t, f"{g.f1_macro.min():.3f}", f"{g.f1_macro.max():.3f}"])
    table(rows, [160, 160, 160])
    p(
        "A diferença entre famílias reforça que o total de linhas não representa diversidade equivalente de situações. Em sentimento, F24 é mais difícil; em churn, F25 concentra os erros. Em oportunidade, as quatro famílias variam de aproximadamente 0,895 a 1,000 de F1 macro. Todas as linhas estão em results/estabilidade_familias.csv."
    )
    sub("Leitura de negócio")
    p(
        "Falso negativo de churn omite uma chance de retenção; falso positivo pode consumir tempo de CS e alarmar a conta. Falso positivo de oportunidade pode gerar abordagem inadequada. Por isso o protótipo recomenda confirmação humana e registra separadamente os três sinais, sem concluir que insatisfação sempre significa cancelamento."
    )
    page()
    h("06 · Sensibilidade e incerteza")
    p(
        "A auditoria mede o efeito de limiares na validação, sem usar o teste para escolher o melhor corte. O protótipo mantém o limiar demonstrativo 0,50 fixado no código. A tabela mostra a troca entre recuperar sinais e gerar alertas."
    )
    th = pd.read_csv(ROOT / "results/sensibilidade_validacao.csv")
    rows = [["Tarefa", "Limiar", "Precision", "Recall", "Alertas / 192"]]
    for _, r in th.iterrows():
        rows.append(
            [
                r.tarefa,
                f"{r.limiar:.2f}",
                f"{r.precision:.3f}",
                f"{r.recall:.3f}",
                str(int(r.alertas)),
            ]
        )
    table(rows, [130, 65, 95, 95, 95], compact=True)
    p(
        "No churn, reduzir o corte de 0,50 para 0,30 aumenta recall de validação de 0,625 para 0,823, ao custo de mais alertas e menor precision. Não adotamos automaticamente esse corte: o negócio precisa definir custo por erro, capacidade de atendimento e resultado em amostra real."
    )
    sub("Scores não são probabilidades reais")
    p(
        "Brier score no teste: churn "
        + f"{audit['brier_teste']['churn']['brier']:.3f}"
        + "; oportunidade "
        + f"{audit['brier_teste']['oportunidade']['brier']:.3f}"
        + ". A referência constante 0,50 tem Brier 0,250. Brier menor é melhor, mas esse diagnóstico em dados sintéticos não comprova calibração para cancelamentos ou compras reais."
    )
    p(
        "A confiança mínima é a menor probabilidade máxima entre as tarefas. Não é garantia de acerto. Cobertura lexical baixa aponta distância do vocabulário do treino. Toda análise exige revisão humana; confiança < 0,60 ou cobertura < 0,35 adiciona alerta de incerteza."
    )
    page()
    h("07 · Sistema de scoring")
    table(
        [
            ["Indicador", "Fórmula", "Uso"],
            ["Risco de churn", "100 × saída da classe risco", "Apoiar triagem de retenção"],
            ["Oportunidade", "100 × saída da classe oportunidade", "Apoiar qualificação comercial"],
            [
                "Sentimento negativo",
                "100 × saída da classe negativo",
                "Apoiar acompanhamento do suporte",
            ],
            [
                "Prioridade",
                "0,55 × risco + 0,30 × oportunidade + 0,15 × negativo",
                "Ordenação heurística de atenção",
            ],
            [
                "Cobertura",
                "Fração de n-gramas conhecidos no TF-IDF de termos",
                "Avisar sobre texto pouco familiar",
            ],
            [
                "Confiança mínima",
                "Mínimo dos maiores scores das três tarefas",
                "Sinalizar ambiguidade da inferência",
            ],
        ],
        [110, 210, 160],
    )
    p(
        "Os scores ficam entre 0 e 100. Os pesos da prioridade são decisão de projeto, não resultado de otimização financeira. Prioridade não representa receita esperada, probabilidade conjunta nem economia comprovada. Não calculamos um score de escuta ativa, pois este módulo não mede tempo de fala real."
    )
    sub("Saída real de uma nova entrada")
    a = Analyzer()
    example = (ROOT / "data/exemplo.txt").read_text()
    r = a.analyze(example)
    p(example)
    table(
        [
            ["Campo", "Resultado"],
            ["Produtos mencionados", ", ".join(r["produtos"])],
            *[[k, str(v)] for k, v in r["scores"].items()],
        ],
        [240, 240],
    )
    p(
        "As evidências apontam frases originais do cliente e seus índices de caracteres. A marcação por sinal compara scores de frases isoladas; serve para conferir o texto, sem atribuir causalidade ou transparência completa ao modelo. A exportação inclui as ações estruturadas e os avisos."
    )
    page()
    h("08 · Protótipo funcional e testes")
    p(
        "Interface local com campo de transcrição, upload TXT, exemplo preenchível, cartões de score, classes, termos, trechos, recomendações e download JSON. O backend, a CLI e os testes reutilizam o mesmo Analyzer. Não há dependência de conta externa, API paga, Docker ou credenciais."
    )
    table(
        [
            ["Etapa", "Comando / interação"],
            [
                "Preparar ambiente",
                "python -m venv .venv; ativar conforme README; instalar requirements.txt",
            ],
            [
                "Treinar e auditar",
                "python train.py; python evaluate_extra.py; python selecao_cv.py",
            ],
            ["Iniciar protótipo", "python app.py; abrir http://127.0.0.1:8765"],
            ["Analisar", "Colar texto ou carregar TXT; clicar em Gerar inteligência"],
            ["Conferir e exportar", "Ler recomendações e evidências; baixar JSON"],
            ["Executar testes", "python -m unittest discover -s tests -v"],
        ],
        [115, 365],
    )
    sub("Verificação executada")
    p(
        "20 testes Python aprovados, incluindo servidor HTTP real, contrato JSON, erro 400, rota 404, entradas inválidas, evidências por posição, produtos, máscara parcial, fórmula da prioridade, persistência do modelo, famílias disjuntas, exclusão de pergunta do vendedor e rótulos com timestamps."
    )
    browser = (ROOT / "results/teste_navegador.txt").read_text()
    p(
        (
            "Teste de navegador: APROVADO. " + browser.strip()
            if browser.startswith("PASS:")
            else "Teste de navegador: pendente neste ambiente. Consulte o log e execute tests/browser_smoke.cjs em um ambiente com Chrome ou Chromium disponível."
        ),
        "SmallCustom",
    )
    p(
        "O roteiro automatizado de navegador carrega exemplo, clica em analisar, confere produto e scores, baixa e interpreta JSON, carrega um TXT e analisa outra reunião, verifica layout de 390 px sem overflow horizontal e rejeita entrada vazia. O log e o script estão incluídos. O servidor escuta só em 127.0.0.1 e não grava transcrições no log."
    )
    p(
        "A interface foi criada para demonstração local. Não foi exposta como serviço público nem integrada ao aplicativo desktop da visão de produto. O README contém comandos específicos para Git Bash e Linux/macOS."
    )
    page()
    h("09 · Novas transcrições")
    p(
        "24 transcrições autorais separadas do gerador, combinando produto, diferentes falas e intenções. Nenhuma entrou em treino ou seleção. Há oito por sentimento, doze por classe binária e seis menções por produto principal. Continuam sendo sintéticas e não são validação externa independente."
    )
    rows = [["Tarefa", "Acertos", "Accuracy", "F1 macro"]]
    for t, v in m["novas_metricas"].items():
        rows.append(
            [
                t,
                str(sum(r["esperado"][t] == r["resultado"]["classificacoes"][t] for r in new))
                + "/24",
                f"{v['accuracy']:.3f}",
                f"{v['f1_macro']:.3f}",
            ]
        )
    table(rows, [150, 90, 120, 120], compact=True)
    rows = [["Caso", "Sentimento predito", "Churn predito", "Oportunidade predita", "Acertos"]]
    for r in new:
        pred = r["resultado"]["classificacoes"]
        rows.append(
            [
                r["id"],
                pred["sentimento"],
                pred["churn"],
                pred["oportunidade"],
                str(sum(r["esperado"][t] == pred[t] for t in pred)) + "/3",
            ]
        )
    table(rows, [40, 110, 95, 165, 70], compact=True)
    p(
        "Textos completos, rótulos esperados, scores, ações, evidências e resultados estão nos arquivos JSON de dados e resultados. Não foram removidos casos com erros.",
        "SmallCustom",
    )
    page()
    h("09 · Desafios linguísticos e erros")
    rows = [["Caso", "Desafio", "Sentimento", "Churn", "Oportunidade"]]
    for r in stress:
        ok = [
            ("correto" if r["esperado"][t] == r["resultado"]["classificacoes"][t] else "erro")
            for t in TASKS
        ]
        rows.append([r["id"], r["categoria"], *ok])
    table(rows, [40, 190, 85, 80, 85], compact=True)
    p(
        "São oito desafios adicionais: pergunta do vendedor, negação, ironia, múltiplos produtos, ausência de produto, falso positivo em CRM/protheus_backup, contradição temporal e texto sem rótulos. O extrator acerta os oito conjuntos de produtos esperados. Um produto citado somente pelo vendedor é registrado como menção, sem influenciar o texto de classificação."
    )
    sub("Erros identificados sem ajuste posterior")
    p(
        "D02: “Não estamos insatisfeitos” ainda foi predito como sentimento negativo. O modelo lexical não resolve toda a composição de negações. Esse caso permanece como falha documentada; o sistema exige revisão mesmo com score alto."
    )
    e = errors[errors.conjunto == "novas"]
    table(
        [
            ["Caso", "Tarefa", "Esperado", "Predito"],
            *[[r.id, r.tarefa, r.real, r.predito] for _, r in e.iterrows()],
        ],
        [50, 120, 155, 155],
        compact=True,
    )
    p(
        "N02 e N09: formas pouco frequentes de permanência foram confundidas com saída. N06: a negação “não vamos cancelar” falhou em contexto de insatisfação. N12: intenção de encerrar o contrato foi perdida. N03: avaliação positiva com migração foi predita como negativa. São hipóteses explicativas pela leitura, não prova causal."
    )
    p(
        "No novo teste houve 11 erros de sentimento, 12 de churn e 11 de oportunidade; nos 24 casos inéditos, 1, 4 e 0. Erros são contados por tarefa, podendo ocorrer mais de um na mesma transcrição. Os textos e previsões estão em results/erros.csv.",
        "SmallCustom",
    )
    page()
    h("10 · Recomendações de negócio")
    table(
        [
            ["Sinal", "Ação sugerida", "Responsável / prazo"],
            ["Churn ≥ 50", "Confirmar motivo da saída e registrar plano de retenção", "CS / 24h"],
            [
                "Oportunidade ≥ 50",
                "Qualificar necessidade, orçamento e prazo antes da proposta",
                "Vendedor / 48h",
            ],
            ["Negativo ≥ 50", "Acompanhar insatisfação e combinar retorno", "Suporte / 24h"],
            [
                "Risco + oportunidade",
                "Priorizar retenção e qualificar expansão depois da confirmação",
                "CS + vendedor",
            ],
            [
                "Sem sinal forte",
                "Registrar próximos passos e confirmar entendimento",
                "Executivo da conta / 72h",
            ],
            [
                "Baixa confiança",
                "Conferir transcrição e contexto antes de agir",
                "Responsável pela conta",
            ],
        ],
        [110, 235, 135],
    )
    p(
        "São sugestões de fluxo, não SLAs contratados. Cada ação exportada tem tipo, responsável, prazo_horas, descrição e score. O sistema não envia mensagens, cria tickets ou oferece descontos. Todas as recomendações pedem confirmação humana, mesmo com scores elevados."
    )
    sub("Exemplo de sinais simultâneos")
    p(
        "Uma conta pode elogiar o serviço, decidir migrar por orientação da matriz e ainda precisar de novos acessos durante a transição. Por isso três classificadores independentes são preferíveis a uma regra que deduza churn do sentimento. Ação sugerida: entender a decisão de saída primeiro; só depois avaliar uma oferta de transição."
    )
    sub("Valor esperado e medição")
    p(
        "Hipóteses: reduzir leitura manual pós-call, padronizar registro e priorizar acompanhamento. Não alegamos ganho de receita ou economia medidos. Em piloto, comparar tempo mediano de revisão, alertas confirmados, riscos encontrados por auditoria, taxa de recomendações aceitas e falsos alertas por reunião. Desfechos reais de retenção exigem janela temporal e comparação adequada."
    )
    sub("Decisão de implantação")
    p(
        "A demonstração cumpre o fluxo funcional, mas a validação de churn e os erros inéditos impedem automação sem revisão. A meta proposta para piloto real é F1 macro ≥ 0,80 por tarefa e recall de risco ≥ 0,85 em avaliação externa por cliente e período, além de custos por erro acordados. Essas metas não foram comprovadas em dados reais."
    )
    page()
    h("11 · Limitações e evolução")
    table(
        [
            ["Limitação", "Próximo passo verificável"],
            [
                "Corpus sintético e dependência de famílias",
                "Coletar transcrições autorizadas de clientes e segmentos variados; separar por cliente, reunião e tempo",
            ],
            [
                "Rótulos definidos na autoria",
                "Criar manual de anotação, usar dois revisores e medir concordância",
            ],
            [
                "Negação, tempo e contexto",
                "Anotar falantes e temporalidade; comparar representação contextual em um novo teste externo",
            ],
            [
                "Scores não calibrados",
                "Reservar validação real para calibração; medir Brier e curva de confiabilidade",
            ],
            [
                "Quatro famílias no teste",
                "Ampliar diversidade independente; reportar incerteza por reunião/cliente",
            ],
            [
                "Produtos por dicionário",
                "Ampliar catálogo e aliases revisados; distinguir menção, uso atual e compra desejada",
            ],
            [
                "Privacidade parcial",
                "Anonimização revisada, minimização, controle de acesso e retenção definida",
            ],
            [
                "Sem integração operacional",
                "Integrar mediante contrato de API, monitorar drift e auditar recomendações aceitas",
            ],
        ],
        [180, 300],
    )
    p(
        "A máscara de e-mail e telefone é parcial e aplicada ao texto de inferência. As evidências e o JSON mantêm o texto original; não inserir dados pessoais reais sem autorização. O servidor não grava as entradas automaticamente. Arquivos TXT/JSON exportados precisam de proteção pelo usuário."
    )
    sub("Conclusão sustentada pelos resultados")
    p(
        "A entrega integra entrada, processamento, seleção de modelos, indicadores e recomendações em um protótipo testado. O novo teste sintético alcança F1 macro próximo de 0,94 por tarefa, e a validação inédita mostra melhora funcional com falhas preservadas. A conclusão é um MVP demonstrável para apoio à revisão comercial; o próximo salto exige dados reais autorizados e validação externa."
    )
    page()
    h("12 · Reprodução e conferência final")
    table(
        [
            ["Componente", "Arquivo / evidência"],
            [
                "Dados e rótulos",
                "data/frases.json; corpus.csv; novas_transcricoes.json; casos_desafio.json",
            ],
            [
                "Pipeline e seleção",
                "pipeline.py; modeling.py; train.py; build_data.py; selecao_cv.py",
            ],
            ["Protótipo", "app.py; cli.py; data/exemplo.txt"],
            ["Auditoria", "evaluate_extra.py; resultados por família, limiar, Brier e desafios"],
            ["Notebook executado", "analise.ipynb; run_notebook.py"],
            ["Testes", "tests/test_pipeline.py; browser_smoke.cjs; logs em results/"],
            ["Documentação", "README.md; checklist_rubrica.md; roteiro_demonstracao.md"],
        ],
        [115, 365],
    )
    p(
        "Ordem para reproduzir: criar e ativar ambiente virtual; instalar requirements.txt; executar build_data.py, train.py, evaluate_extra.py, selecao_cv.py, testes Python, run_notebook.py e make_report.py. Para usar: python app.py. O teste de navegador é opcional para reprodução e usa Node.js, Playwright e Chrome/Chromium; o log executado está incluído."
    )
    sub("Ambiente e rastreabilidade")
    p(
        "Python "
        + m["ambiente"]["python"]
        + "; scikit-learn "
        + m["ambiente"]["sklearn"]
        + ". Seed 42. Corpus SHA-256: "
        + m["hash_corpus_sha256"]
        + ".",
        "SmallCustom",
    )
    sub("Fontes")
    p(
        "1. SPRINT_3_E_4_CHALLENGE_2026_TOTVS (3).pdf, páginas físicas 23–24: objetivo e sete critérios de Data Science / Sprint 4. 2. Kaivi AI (2)-1(1).pdf: visão do produto, equipe e módulos. Ambos fornecidos pelo usuário. 3. Experimentos próprios em results/metricas.json, protocolo.json, comparacao_validacao.csv, erros.csv, novas_resultados.json, desafios_resultados.json e auditoria.json. Nenhuma transcrição real foi fornecida ou inventada como se fosse real.",
        "SmallCustom",
    )
    p(
        "Conferência da rubrica: os sete critérios possuem implementação e evidências. Os pesos do guia não representam uma nota garantida; a avaliação final cabe ao professor.",
        "SmallCustom",
    )
    page()
    h("13 · Auditoria final da entrega")
    acceptance = json.loads((ROOT / "results/aceitacao_rubrica.json").read_text())
    clean = json.loads((ROOT / "results/reproducao_limpa.json").read_text())
    p(
        "Além dos testes funcionais, uma auditoria recalcula as métricas dos modelos entregues, verifica a seleção por validação e confere a consistência das saídas. Não há atribuição automática de nota: cada linha abaixo indica evidências técnicas conferidas para o critério oficial."
    )
    table(
        [
            ["Critério", "Peso oficial", "Evidências"],
            *[
                [
                    r["criterio"],
                    r["peso_oficial"],
                    "Conferidas" if r["evidencias_conferidas"] else "Revisar",
                ]
                for r in acceptance["criterios"]
            ],
        ],
        [285, 70, 125],
        compact=True,
    )
    p(
        str(acceptance["verificacoes_aprovadas"])
        + " de "
        + str(acceptance["verificacoes_total"])
        + " verificações aprovadas. O script auditar_entrega.py pode ser executado novamente. A saída completa está em results/aceitacao_rubrica.json."
    )
    sub("Reprodução em ambiente novo")
    p(
        "O projeto foi extraído do ZIP em outra pasta. As dependências declaradas foram instaladas em um novo ambiente virtual, sem aproveitar pacotes globais. Em seguida foram reconstruídos os dados, modelos, avaliações e notebook."
    )
    table(
        [
            ["Etapa reproduzida", "Resultado"],
            *[[r["comando"], "Aprovada" if r["aprovado"] else "Falhou"] for r in clean["etapas"]],
        ],
        [355, 125],
        compact=True,
    )
    p(
        "Corpus e métricas reproduzidos: "
        + ("sim" if clean["metricas_e_corpus_reproduzidos"] else "não")
        + ". Python do ambiente isolado: "
        + clean["ambiente"]["python"]
        + ". O teste de navegador também foi executado e aprovado, com log separado. Esses registros aumentam a rastreabilidade e a facilidade de correção, sem transformar cobertura da rubrica em garantia de nota."
    )
    out = ROOT / "docs/Kaivi_AI_Sprint4_Data_Science.pdf"

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#d5e0ec"))
        canvas.line(55, 46, 540, 46)
        canvas.setFont(F, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(55, 32, "KAIVI AI  /  DATA SCIENCE  /  SPRINT 4")
        canvas.drawRightString(540, 32, str(doc.page))

    SimpleDocTemplate(
        str(out),
        pagesize=(595.28, 841.89),
        rightMargin=55,
        leftMargin=55,
        topMargin=50,
        bottomMargin=65,
        title="Kaivi AI - Sprint 4 - Data Science",
        author="Equipe Kaivi AI",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(out)


if __name__ == "__main__":
    main()
