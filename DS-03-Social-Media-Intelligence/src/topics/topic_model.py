
import json
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN


INPUT_FILE = Path("data/processed/youtube/cleaned_comments.csv")
OUTPUT_DIR = Path("data/processed/youtube/topics")
MODEL_DIR = Path("models/bertopic")

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def fit_topics(documents, **kwargs):
    """Fit BERTopic and return topic assignments and fitted model."""

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = UMAP(
        n_neighbors=min(15, max(2, len(documents) - 1)),
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=max(3, min(10, len(documents) // 10)),
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=None
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True
    )

    topics, probabilities = topic_model.fit_transform(documents)

    return topic_model, topics, probabilities


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    print("Loading cleaned YouTube comments...")

    df = pd.read_csv(INPUT_FILE)

    if "text" not in df.columns:
        print("The input CSV does not contain a text column.")
        return

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    df.reset_index(drop=True, inplace=True)

    # Remove exact duplicate text for more reliable topic discovery.
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if len(df) < 20:
        print(f"Only {len(df)} unique comments available.")
        print("At least 20 comments are required.")
        return

    documents = df["text"].tolist()

    print(f"Starting BERTopic on {len(documents)} unique comments...")
    print("Loading multilingual embedding model...")

    topic_model, topics, probabilities = fit_topics(documents)

    # Save comment-level topic assignments.
    df["topic_id"] = topics

    df["topic_name"] = [
        topic_model.topic_labels_.get(topic, "Outlier")
        for topic in topics
    ]

    df.to_csv(
        OUTPUT_DIR / "comment_topics.csv",
        index=False
    )

    # Save topic information.
    topic_info = topic_model.get_topic_info()

    topic_info.to_csv(
        OUTPUT_DIR / "topic_summary.csv",
        index=False
    )

    # Save topic keywords.
    keyword_rows = []

    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue

        words = topic_model.get_topic(topic_id) or []

        keyword_rows.append({
            "topic_id": int(topic_id),
            "keywords": ", ".join(
                word for word, score in words[:10]
            )
        })

    pd.DataFrame(keyword_rows).to_csv(
        OUTPUT_DIR / "topic_keywords.csv",
        index=False
    )

    # Generate topic trends over time.
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(
            df["published_at"],
            errors="coerce",
            utc=True
        )

        trend_df = df.dropna(subset=["published_at"]).copy()

        if not trend_df.empty:
            trend_df["date"] = (
                trend_df["published_at"].dt.date.astype(str)
            )

            trends = (
                trend_df.groupby(["date", "topic_id"])
                .size()
                .reset_index(name="comment_count")
            )

            trends.to_csv(
                OUTPUT_DIR / "topic_trends.csv",
                index=False
            )

    # Save the trained BERTopic model.
    topic_model.save(
        str(MODEL_DIR),
        serialization="safetensors",
        save_ctfidf=True
    )

    # Generate execution report.
    topic_counts = pd.Series(topics).value_counts()

    summary = {
        "total_comments_used": len(df),
        "number_of_topics": len(set(topics) - {-1}),
        "outlier_comments": int((pd.Series(topics) == -1).sum()),
        "topic_distribution": {
            str(topic): int(count)
            for topic, count in topic_counts.items()
        },
        "embedding_model": EMBEDDING_MODEL,
        "outputs": [
            "comment_topics.csv",
            "topic_summary.csv",
            "topic_keywords.csv",
            "topic_trends.csv (if dates are available)"
        ]
    }

    with open(
        OUTPUT_DIR / "topic_model_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4, default=str)

    print("\nBERTopic modeling completed!")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
