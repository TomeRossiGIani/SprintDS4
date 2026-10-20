"""Corpus autoral sintético, famílias linguísticas disjuntas e rótulos explícitos."""

import csv, itertools, json, random, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ("sentimento", "churn", "oportunidade")


def main():
    phrases = json.loads((ROOT / "data/frases.json").read_text(encoding="utf-8"))
    rows = []
    rng = random.Random(42)
    assert all(len(v) == 28 for groups in phrases.values() for v in groups.values())
    for family in range(28):
        split = "treino" if family < 20 else ("validacao" if family < 24 else "teste")
        for sentiment, churn, opp, product in itertools.product(
            phrases["sentimento"],
            phrases["churn"],
            phrases["oportunidade"],
            ["Protheus", "RM", "Fluig", "TOTVS"],
        ):
            chunks = [phrases[t][v][family] for t, v in zip(TASKS, (sentiment, churn, opp))]
            rng.shuffle(chunks)
            # A pergunta do vendedor fica fora do texto de intenção do cliente.
            text = f"Vendedor: Como está a operação?\nCliente: Utilizamos {product}. " + " ".join(
                chunks
            )
            rows.append(
                dict(
                    id=f"S{len(rows)+1:04}",
                    familia=f"F{family:02}",
                    particao=split,
                    texto=text,
                    sentimento=sentiment,
                    churn=churn,
                    oportunidade=opp,
                    origem="sintetico_autoral",
                )
            )
    with (ROOT / "data/corpus.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    print({s: sum(r["particao"] == s for r in rows) for s in ["treino", "validacao", "teste"]})


if __name__ == "__main__":
    main()
