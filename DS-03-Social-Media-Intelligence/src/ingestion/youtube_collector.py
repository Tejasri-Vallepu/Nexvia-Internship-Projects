
import os
import json
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
OUTPUT_DIR = Path("data/raw/youtube")

SEARCH_QUERIES = [
    "artificial intelligence",
    "social media misinformation",
    "digital media trends"
]

VIDEOS_PER_QUERY = 20
COMMENT_PAGES_PER_VIDEO = 2


def create_youtube_client():
    """Create YouTube Data API client."""

    if not API_KEY:
        raise ValueError("YOUTUBE_API_KEY missing in .env")

    return build(
        "youtube",
        "v3",
        developerKey=API_KEY
    )


def search_videos(youtube, query, max_results=20):
    """Search YouTube videos for a given topic."""

    videos = []

    try:
        response = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=min(max_results, 50),
            order="relevance"
        ).execute()

        for item in response.get("items", []):
            videos.append({
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "channel_id": item["snippet"]["channelId"],
                "channel_title": item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "search_query": query
            })

    except HttpError as error:
        print(f"Search failed for '{query}': {error}")

    return videos


def collect_comments(youtube, video_id, max_pages=2):
    """
    Collect top-level comments and replies returned by commentThreads.

    Stores author channel IDs and parent comment IDs when available.
    """

    comments = []
    page_token = None

    for page in range(max_pages):

        try:
            response = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=page_token,
                textFormat="plainText"
            ).execute()

            for item in response.get("items", []):

                top_comment = item["snippet"]["topLevelComment"]
                snippet = top_comment["snippet"]

                author_info = snippet.get(
                    "authorChannelId", {}
                )

                # Store top-level comment.
                comments.append({
                    "video_id": video_id,
                    "comment_id": top_comment["id"],
                    "parent_id": None,
                    "author_channel_id": author_info.get("value"),
                    "text": snippet.get("textDisplay", ""),
                    "published_at": snippet.get("publishedAt"),
                    "like_count": snippet.get("likeCount", 0),
                    "comment_type": "top_level"
                })

                # Store replies included in the response.
                replies = item.get(
                    "replies", {}
                ).get("comments", [])

                for reply in replies:

                    reply_snippet = reply.get("snippet", {})

                    reply_author = reply_snippet.get(
                        "authorChannelId", {}
                    )

                    comments.append({
                        "video_id": video_id,
                        "comment_id": reply["id"],
                        "parent_id": top_comment["id"],
                        "author_channel_id": reply_author.get("value"),
                        "text": reply_snippet.get("textDisplay", ""),
                        "published_at": reply_snippet.get("publishedAt"),
                        "like_count": reply_snippet.get("likeCount", 0),
                        "comment_type": "reply"
                    })

            page_token = response.get("nextPageToken")

            if not page_token:
                break

        except HttpError as error:
            print(
                f"Comments unavailable for {video_id}: {error}"
            )
            break

    return comments


def load_existing(filename):
    """Load existing JSON records without deleting old data."""

    path = OUTPUT_DIR / filename

    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except (json.JSONDecodeError, OSError):
        print(f"Could not read existing {filename}.")
        return []


def merge_records(existing_records, new_records, key):
    """Merge old and new records using a unique identifier."""

    record_map = {
        item[key]: item
        for item in existing_records
        if item.get(key)
    }

    for item in new_records:
        identifier = item.get(key)

        if identifier:
            record_map[identifier] = item

    return list(record_map.values())


def save_json(data, filename):
    """Save records as JSON."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / filename

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"Saved {len(data)} records -> {path}")


def collect_youtube_data():
    """Run complete YouTube data collection pipeline."""

    youtube = create_youtube_client()

    all_videos = []
    all_comments = []

    print("Starting multi-topic YouTube collection...")

    # Search videos for all configured topics.
    for query in SEARCH_QUERIES:

        print(f"\nSearching topic: {query}")

        videos = search_videos(
            youtube,
            query,
            VIDEOS_PER_QUERY
        )

        all_videos.extend(videos)

    # Remove duplicate videos.
    video_map = {
        video["video_id"]: video
        for video in all_videos
    }

    videos = list(video_map.values())

    print(f"\nUnique videos found: {len(videos)}")

    # Collect comments and available replies.
    for index, video in enumerate(videos, start=1):

        video_id = video["video_id"]

        print(
            f"Collecting comments {index}/{len(videos)}: "
            f"{video_id}"
        )

        comments = collect_comments(
            youtube,
            video_id,
            COMMENT_PAGES_PER_VIDEO
        )

        all_comments.extend(comments)

        print(f"Comments collected for this video: {len(comments)}")

    # Load previous records.
    existing_videos = load_existing("videos.json")
    existing_comments = load_existing("comments.json")

    # Merge old and new data.
    final_videos = merge_records(
        existing_videos,
        videos,
        "video_id"
    )

    final_comments = merge_records(
        existing_comments,
        all_comments,
        "comment_id"
    )

    # Collection statistics.
    top_level_count = sum(
        1 for item in final_comments
        if item.get("comment_type") == "top_level"
    )

    reply_count = sum(
        1 for item in final_comments
        if item.get("comment_type") == "reply"
    )

    author_id_count = sum(
        1 for item in final_comments
        if item.get("author_channel_id")
    )

    metadata = {
        "source": "YouTube Data API v3",
        "queries": SEARCH_QUERIES,
        "collected_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "videos_in_current_run": len(videos),
        "comments_in_current_run": len(all_comments),

        "total_unique_videos": len(final_videos),
        "total_unique_comments": len(final_comments),

        "top_level_comments": top_level_count,
        "replies": reply_count,
        "comments_with_author_id": author_id_count,

        "comment_pages_per_video": COMMENT_PAGES_PER_VIDEO
    }

    # Save datasets.
    save_json(final_videos, "videos.json")
    save_json(final_comments, "comments.json")
    save_json(metadata, "collection_metadata.json")

    print("\nYOUTUBE DATA COLLECTION COMPLETED!")
    print(json.dumps(metadata, indent=4))


if __name__ == "__main__":

    try:
        collect_youtube_data()

    except Exception as error:
        print("Collection failed:", error)
