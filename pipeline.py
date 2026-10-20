"""Texto -> falas do cliente -> ML -> sinais rastreáveis -> ação assistida."""

import re, unicodedata
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
TASKS = ("sentimento", "churn", "oportunidade")
SPEAKER = re.compile(
    r"(?im)(?:^|(?<=\n))\s*(?:(?:\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*))?(cliente|comprador|vendedor|consultor|comercial)\s*:"
)


def normalize(text):
    if not isinstance(text, str):
        raise ValueError("A transcrição deve ser um texto.")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " contato_anonimizado ", text)
    text = re.sub(
        r"(?<!\w)(?:\+?55\s*)?\(?\d{2}\)?[\s-]*9?\d{4}[\s-]?\d{4}(?!\w)",
        " telefone_anonimizado ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def client_segments(text):
    matches = list(SPEAKER.finditer(text))
    if not matches:
        return [{"texto": text, "inicio": 0}], "sem_rotulos"
    result = []
    for i, m in enumerate(matches):
        if m.group(1).lower() in ("cliente", "comprador"):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            raw = text[start:end]
            trim = len(raw) - len(raw.lstrip())
            value = raw.strip()
            if value:
                result.append({"texto": value, "inicio": start + trim})
    return result, "cliente_identificado" if result else "sem_fala_cliente"


def client_text(text):
    parts, _ = client_segments(text)
    return normalize(" ".join(p["texto"] for p in parts))


def validate(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Insira uma transcrição não vazia.")
    if len(text) > 50000:
        raise ValueError("Limite de 50.000 caracteres por transcrição.")
    if len(re.findall(r"\w+", text)) < 5:
        raise ValueError("Insira pelo menos cinco palavras para análise.")


def products(text):
    return [
        p
        for p in ["Protheus", "RM", "Fluig", "TOTVS"]
        if re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text, re.I)
    ]


def evidence_sentences(parts):
    result = []
    for part in parts:
        for m in re.finditer(r"[^.!?\n]+[.!?]?", part["texto"]):
            raw = m.group()
            value = raw.strip()
            if value:
                result.append(
                    {
                        "trecho": value,
                        "inicio_caractere": part["inicio"]
                        + m.start()
                        + len(raw)
                        - len(raw.lstrip()),
                    }
                )
    return result


class Analyzer:
    def __init__(self, path=None):
        path = Path(path or ROOT / "results/modelos.joblib")
        if not path.exists():
            raise FileNotFoundError("Execute python train.py antes de iniciar o protótipo.")
        self.bundle = joblib.load(path)  # Somente arquivo local confiável gerado por train.py.

    def analyze(self, text):
        validate(text)
        parts, speaker_status = client_segments(text)
        if not parts:
            raise ValueError(
                "Não foi encontrada fala do cliente. Use Cliente: no início de uma linha."
            )
        relevant = " ".join(p["texto"] for p in parts)
        if len(re.findall(r"\w+", relevant)) < 5:
            raise ValueError("A fala do cliente precisa de pelo menos cinco palavras.")
        predictions = {}
        probabilities = {}
        evidences = evidence_sentences(parts)
        sentences = [e["trecho"] for e in evidences]
        for task in TASKS:
            model = self.bundle["models"][task]
            probs = model.predict_proba([text])[0]
            idx = int(probs.argmax())
            predictions[task] = str(model.classes_[idx])
            probabilities[task] = {
                str(c): round(float(p), 6) for c, p in zip(model.classes_, probs)
            }
            # Rastreabilidade: frase isolada mais associada à classe prevista (não explicação causal).
            if sentences:
                sentence_probs = model.predict_proba(sentences)[:, idx]
                best = int(sentence_probs.argmax())
                evidences[best].setdefault("sinais", []).append(task)
        vec = self.bundle["term_vectorizer"]
        row = vec.transform([text])
        names = vec.get_feature_names_out()
        indices = row.indices
        ranked = sorted(indices, key=lambda i: row[0, i], reverse=True)[:8]
        terms = [str(names[i]) for i in ranked]
        tokens = vec.build_analyzer()(text)
        cov = sum(t in vec.vocabulary_ for t in tokens) / max(1, len(tokens))
        risk = probabilities["churn"]["risco"]
        opp = probabilities["oportunidade"]["oportunidade"]
        neg = probabilities["sentimento"]["negativo"]
        confidence = min(max(p.values()) for p in probabilities.values())
        low = confidence < 0.60 or cov < 0.35
        indicators = {
            "risco_churn": round(100 * risk, 1),
            "oportunidade": round(100 * opp, 1),
            "sentimento_negativo": round(100 * neg, 1),
            "prioridade": round(100 * (0.55 * risk + 0.30 * opp + 0.15 * neg), 1),
        }
        recommendations = [
            "Validação humana obrigatória: confirmar os sinais com o cliente antes de executar as ações sugeridas."
        ]
        actions = []
        if low:
            recommendations.append(
                "Revisar a transcrição antes de agir: baixa confiança ou vocabulário pouco conhecido."
            )
        if speaker_status == "sem_rotulos":
            recommendations.append(
                "Sem identificação de falantes: o texto inteiro foi analisado. Confirme quem declarou cada intenção."
            )
        if risk >= 0.5:
            actions.append(
                {
                    "tipo": "retencao",
                    "responsavel": "CS",
                    "prazo_horas": 24,
                    "acao": "Confirmar motivo da saída e registrar plano de retenção.",
                    "score": indicators["risco_churn"],
                }
            )
        if opp >= 0.5:
            actions.append(
                {
                    "tipo": "expansao",
                    "responsavel": "Vendedor",
                    "prazo_horas": 48,
                    "acao": "Qualificar necessidade, orçamento e prazo antes de proposta. Priorizar retenção se houver risco.",
                    "score": indicators["oportunidade"],
                }
            )
        if neg >= 0.5:
            actions.append(
                {
                    "tipo": "insatisfacao",
                    "responsavel": "Suporte",
                    "prazo_horas": 24,
                    "acao": "Abrir acompanhamento da insatisfação e combinar retorno.",
                    "score": indicators["sentimento_negativo"],
                }
            )
        if not actions:
            actions.append(
                {
                    "tipo": "acompanhamento",
                    "responsavel": "Executivo da conta",
                    "prazo_horas": 72,
                    "acao": "Registrar próximos passos e confirmar o entendimento com o cliente.",
                    "score": indicators["prioridade"],
                }
            )
        recommendations.extend(
            f"Responsável: {a['responsavel']}. Em até {a['prazo_horas']}h: {a['acao']}"
            for a in actions
        )
        return {
            "produtos": products(text) or ["Não identificado"],
            "classificacoes": predictions,
            "probabilidades_modelo": probabilities,
            "scores": indicators,
            "confianca_minima": round(confidence, 3),
            "cobertura_vocabulario": round(cov, 3),
            "revisao_humana": True,
            "baixa_confianca": low,
            "falantes": speaker_status,
            "termos": terms,
            "recomendacoes": recommendations,
            "acoes": actions,
            "evidencias": evidences[:12],
            "aviso": "Scores de demonstração com dados sintéticos; não são probabilidades calibradas de eventos reais.",
        }
