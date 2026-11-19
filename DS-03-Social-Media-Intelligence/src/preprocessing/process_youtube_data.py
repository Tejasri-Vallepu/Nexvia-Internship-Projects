
import json
import re
import html
from pathlib import Path

import pandas as pd
from langdetect import detect, LangDetectException

RAW_DIR = Path("data/raw/youtube")
OUTPUT_DIR = Path("data/processed/youtube")


def clean_text(text):
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text):
    if len(text) < 15:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def load_json(filename):
    path = RAW_DIR / filename

    if not path.exists():
        print(f"Missing file: {path}")
        return []

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    return data.get("items", [])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = load_json("videos.json")
    comments = load_json("comments.json")

    # Process flat video records
    video_rows = []

    for video in videos:
        video_rows.append({
            "video_id": video.get("video_id"),
            "title": clean_text(video.get("title", "")),
            "description": clean_text(video.get("description", "")),
            "channel_id": video.get("channel_id"),
            "channel_title": video.get("channel_title"),
            "published_at": video.get("published_at"),
        })

    # Process flat comment records
    comment_rows = []

    for comment in comments:
        comment_rows.append({
            "video_id": comment.get("video_id"),
            "comment_id": comment.get("comment_id"),
            "text": clean_text(comment.get("text", "")),
            "published_at": comment.get("published_at"),
            "like_count": comment.get("like_count", 0),
        })

    video_df = pd.DataFrame(video_rows)
    comment_df = pd.DataFrame(comment_rows)

    if not video_df.empty:
        video_df = video_df.drop_duplicates(subset=["video_id"])
        video_df["language"] = video_df["title"].apply(detect_language)

    if not comment_df.empty:
        comment_df = comment_df.drop_duplicates(subset=["comment_id"])
        comment_df["language"] = comment_df["text"].apply(detect_language)

    video_df.to_csv(
        OUTPUT_DIR / "cleaned_videos.csv", index=False
    )

    comment_df.to_csv(
        OUTPUT_DIR / "cleaned_comments.csv", index=False
    )

    report = {
        "videos_loaded": len(videos),
        "comments_loaded": len(comments),
        "videos_after_deduplication": len(video_df),
        "comments_after_deduplication": len(comment_df),
        "comments_with_text": int(
            comment_df["text"].ne("").sum()
        ) if not comment_df.empty else 0,
    }

    with open(
        OUTPUT_DIR / "data_quality_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("Preprocessing completed!")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
