"""Candidatos fixados antes da nova avaliação; seleção usa apenas validação."""

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from pipeline import client_text


def make_model(name):
    word = TfidfVectorizer(preprocessor=client_text, ngram_range=(1, 2), sublinear_tf=True)
    if name == "NaiveBayes_palavras":
        return Pipeline([("features", word), ("clf", MultinomialNB(alpha=0.5))])
    features = word
    if name == "Logistica_hibrida":
        features = FeatureUnion(
            [
                ("palavras", word),
                (
                    "caracteres",
                    TfidfVectorizer(
                        preprocessor=client_text,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        sublinear_tf=True,
                        min_df=2,
                    ),
                ),
            ],
            transformer_weights={"palavras": 1.0, "caracteres": 0.8},
        )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(C=4, max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )


CANDIDATES = ("NaiveBayes_palavras", "Logistica_palavras", "Logistica_hibrida")
