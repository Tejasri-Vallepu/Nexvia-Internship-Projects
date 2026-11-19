
import pandas as pd
from pathlib import Path
from transformers import pipeline

INPUT_FILE = Path("data/processed/youtube/cleaned_comments.csv")
OUTPUT_FILE = Path("data/processed/youtube/sentiment_results.csv")

def main():
    if not INPUT_FILE.exists():
        print("Cleaned comments file not found!")
        return

    df = pd.read_csv(INPUT_FILE)

    if df.empty or "text" not in df.columns:
        print("No comments available for analysis.")
        return

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].copy()

    if df.empty:
        print("No valid comment text found.")
        return

    print("Loading multilingual sentiment model...")

    sentiment_model = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        top_k=1
    )

    results = []

    for text in df["text"]:
        try:
            prediction = sentiment_model(
                text[:512],
                truncation=True
            )[0][0]

            label = prediction["label"].lower()

            results.append({
                "sentiment": label,
                "confidence": round(float(prediction["score"]), 4)
            })

        except Exception as error:
            print("Error processing comment:", error)
            results.append({
                "sentiment": "error",
                "confidence": 0.0
            })

    result_df = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(results)],
        axis=1
    )

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSentiment analysis completed!")
    print(result_df[["text", "sentiment", "confidence"]])
    print("\nSentiment distribution:")
    print(result_df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
