"""Reddit ingestion starter. Implement only with authorized API access."""
import json
from pathlib import Path
from typing import Iterable


def save_raw_records(records: Iterable[dict], output_path: str = "data/raw/reddit_posts.jsonl") -> int:
    """Write records as JSON Lines and return the number saved."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def collect_posts(*args, **kwargs):
    """TODO: Add PRAW client, explicit subreddit/query scope, and rate-limit handling."""
    raise NotImplementedError("Configure authorized Reddit API access before collecting data.")
