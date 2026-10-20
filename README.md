# Kaivi AI - Sprint 4 de Data Science

Transcrição → processamento → modelos → indicadores → recomendações.

## Equipe
- Leonardo Eiji Kina - RM 562784
- Nicholas Braga de Souza - RM 561733
- Tomé Rossi Giani - RM 562422
- Vitor Ramos de Farias - RM 561958

## Início rápido
Requer Python 3.11 ou 3.12. Abra o Bash na pasta do projeto:

```bash
python -m venv .venv
# Windows / Git Bash:
source .venv/Scripts/activate
# Linux ou macOS: use source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python app.py
```
Abra **http://127.0.0.1:8765**. Cole texto ou carregue TXT UTF-8, clique em **Gerar inteligência**, confira as ações e os trechos e baixe JSON. Ctrl+C encerra. Porta ocupada: `python app.py --port 8766`.

Sem Docker, conta externa, API paga ou credenciais. Nenhuma alteração na infraestrutura da empresa é necessária. Não use dados reais sem autorização.

Use `Cliente:` e `Vendedor:` no início de linhas diferentes para separar falantes. `Comprador:`, `Consultor:` e `Comercial:` também são aceitos, com timestamp opcional. Sem rótulos, o texto inteiro é analisado e o sistema avisa. Se só há falas do vendedor, a entrada é rejeitada. Produtos são menções em toda a transcrição; o ML considera somente a fala do cliente quando identificada.

## Reproduzir dados, modelos, avaliação e relatório
```bash
python build_data.py
python train.py
python evaluate_extra.py
python selecao_cv.py
python -m unittest discover -s tests -v
python run_notebook.py
python make_report.py
```
Os modelos e resultados já estão incluídos. Se o joblib apresentar incompatibilidade de ambiente, execute `python train.py` para regenerar. Nunca carregue joblib de terceiros. O relatório usa ReportLab e, se disponíveis, fontes DejaVu Sans; existe fallback Helvetica.

`run_notebook.py` executa as células Python e grava suas saídas no notebook padrão, sem exigir Jupyter. Para explorá-lo interativamente, abra `analise.ipynb` no VS Code ou Jupyter e use um kernel do mesmo ambiente.

### Linha de comando
```bash
python cli.py data/exemplo.txt --saida resultado.json
```

### API local
- `GET /health`: status do servidor.
- `POST /api/analyze`: objeto JSON `{"texto":"Cliente: ..."}`.
- Sucesso: produtos, três classes, scores, probabilidades do modelo, confiança, cobertura, ações, recomendações, termos, evidências e avisos.
- Erros: 400 para entrada inválida; 413 para corpo excessivo; 404 para rota inexistente.
- Limite de texto: 50 mil caracteres; pelo menos cinco palavras analisáveis.
- Execução local em 127.0.0.1; nenhuma transcrição é escrita no log ou persistida pelo servidor.

## Avaliação reproduzível
Base autoral **sintética**: 1.344 transcrições, 28 famílias linguísticas, produtos e três rótulos independentes. Famílias inteiras ficam em uma única partição:

| Partição | Famílias | Linhas |
|---|---|---:|
| Treino | F00–F19 | 960 |
| Validação | F20–F23 | 192 |
| Teste | F24–F27 | 192 |

Mais 24 transcrições inéditas e 8 desafios, sem uso no ajuste. Os 18 casos da primeira tentativa estão identificados como desenvolvimento conhecido e não contam como inéditos. Os exemplos não foram fornecidos pela TOTVS e não representam reuniões reais. As combinações compartilham frases; o total de linhas não equivale a reuniões independentes.

Três candidatos fixos: Naive Bayes com palavras; Regressão Logística com palavras; Regressão Logística com palavras e caracteres. Seleção por F1 macro na validação; desempate por recall macro, modelo logístico de palavras e ordem fixa. TF-IDF ajustado só no treino, sem retreino pós-seleção. `results/protocolo.json` registra hashes de corpus, candidatos e casos novos.

| Tarefa | Modelo final | F1 macro validação | F1 macro teste | Acertos inéditos |
|---|---|---:|---:|---:|
| Sentimento | Logistica_palavras | 1,000 | 0,942 | 23/24 |
| Churn | Logistica_palavras | 0,630 | 0,937 | 20/24 |
| Oportunidade | Logistica_hibrida | 0,765 | 0,943 | 24/24 |

O recall de risco no teste é 1,000 e a precision é 0,889. A diferença entre validação e teste mostra sensibilidade linguística. A auditoria inclui baseline, métricas por classe, matrizes, erros, variação por família, Brier e sensibilidade de limiares. Não representa desempenho comprovado em produção.

### Justificativa do modelo final
- **Métrica de seleção: F1 macro.** As classes são balanceadas e todas importam. F1 macro pesa cada classe igualmente e equilibra precision e recall; accuracy sozinha esconderia uma classe ignorada.
- **Churn:** a métrica crítica é o recall da classe `risco`, porque deixar passar um cliente em saída custa mais do que um alerta falso, que é filtrado pela revisão humana. O modelo entregue recupera todos os casos de risco do teste.
- **Oportunidade:** a Regressão Logística híbrida (palavras + caracteres) teve a maior F1 na validação, coerente com variações morfológicas.
- **Sentimento:** empate perfeito na validação; a regra de desempate fixada antes do teste escolhe a Regressão Logística de palavras, mais simples e interpretável.

Como a validação tem só 4 famílias, `selecao_cv.py` repete a comparação com GroupKFold (6 folds sobre as 24 famílias de desenvolvimento) e calcula o IC 95% bootstrap do F1 no teste. O teste não participa da escolha. Resultados em `results/selecao_cv.json`, no notebook e no relatório.

## Testes
20 testes Python/HTTP aprovados. O script `tests/browser_smoke.cjs` testa cliques, geração de análise, upload TXT, download JSON, entrada vazia e largura mobile. Para reproduzir (o script inicia e encerra o servidor):

```bash
npm install playwright
npx playwright install chromium
python tests/run_browser.py
```
Se necessário, informe `CHROME_PATH` com o executável do Chrome. O teste completo foi aprovado, incluindo upload TXT, análise, download JSON e layout mobile sem overflow. O log está em `results/teste_navegador.txt`.

## Organização
- `build_data.py` + `data/frases.json`: gerador e frases rotuladas.
- `modeling.py`: candidatos e hiperparâmetros fixos.
- `train.py`: seleção, teste, baseline, matrizes e novas transcrições.
- `pipeline.py`: entrada, falantes, normalização, inferência, produtos, termos, scores e ações.
- `app.py` / `cli.py`: protótipo e execução por arquivo.
- `evaluate_extra.py`: desafios, estabilidade por família, Brier e limiares.
- `selecao_cv.py`: validação cruzada agrupada por família e IC bootstrap do teste (diagnóstico; não altera o modelo).
- `analise.ipynb` / `run_notebook.py`: notebook com saídas reais e executor.
- `results/`: métricas, modelos, erros, previsões individuais, gráficos e logs.
- `docs/`: relatório, matriz da rubrica e roteiro de apresentação.

## Recomendações de negócio
1. **Triagem pós-reunião, não decisão automática.** Risco de churn vai para CS em 24h, oportunidade para o vendedor em 48h e insatisfação para o suporte em 24h. Com risco e oportunidade juntos, a retenção vem primeiro.
2. **Piloto assistido antes de escalar:** 5 vendedores por 8 a 12 semanas, medindo tempo de revisão com e sem o Kaivi, alertas confirmados, riscos perdidos e aceitação das ações.
3. **Limiar de churn pelo custo do erro:** baixar o corte de 0,50 para 0,30 aumenta o recall de risco na validação, com mais alertas. A decisão depende da capacidade do time de CS e de dados reais.
4. **Dados reais autorizados e anonimizados,** com dupla anotação, antes de retreinar.
5. **Critério para integrar ao CRM:** F1 macro ≥ 0,80 por tarefa e recall de risco ≥ 0,85 em avaliação externa por cliente e período (metas propostas, não comprovadas).

## Interpretação e limites
Scores são saídas não calibradas de modelos sintéticos multiplicadas por 100. Não equivalem a chance real de cancelamento ou compra. A prioridade usa 55% risco, 30% oportunidade e 15% sentimento negativo, pesos heurísticos sem validação financeira. O limiar de ação 0,50 é demonstrativo e não foi escolhido pelo teste.

Toda saída exige revisão humana. Confiança mínima menor que 0,60 ou cobertura abaixo de 0,35 gera aviso adicional. Negações, temporalidade e contextos mistos ainda podem falhar. Os erros estão preservados, inclusive “não estamos insatisfeitos”. Trechos associados a sinais são escolhidos pelo score da frase isolada e não são explicações causais.

A máscara de e-mail/telefone é parcial. Evidências e JSON usam texto original, que pode conter dados pessoais. Não inserir material real sem autorização. Produto é uma menção por dicionário, não confirmação de uso ou aquisição. Não inclui STT, áudio, RAG, Java ou banco de dados; a Sprint de Data Science começa na transcrição pronta.

## Referências

- TOTVS / FIAP. Guia Challenge 2026 – Engenharia de Software, Sprints 3 e 4 (critérios de avaliação de Data Science, Sprint 4).
- Equipe Kaivi.AI. Documento de visão do produto, backlog e protótipos (Sprint 3).
- Documentação do scikit-learn (TF-IDF, Naive Bayes, Regressão Logística e métricas de classificação).
- Os critérios oficiais estão mapeados em `docs/checklist_rubrica.md`.
## Auditoria de integridade da entrega
```bash
python auditar_entrega.py
```
O comando executa 36 conferências ligadas aos sete critérios: hashes, separação dos dados, seleção dos modelos, recálculo de métricas, matrizes, probabilidades, prioridade, transcrições inéditas, evidências e ações. Saída: `results/aceitacao_rubrica.json`. São verificações técnicas de cobertura e consistência; o script não atribui nota acadêmica. A reprodução completa em um ambiente virtual novo está registrada em `results/reproducao_limpa.json` e no log correspondente.
