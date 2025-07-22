import json
from typing import List

def extract_posts(json_path: str) -> List[dict]:
    """
    Extracts post titles, creation timestamps, and media URIs from a posts_1.json file.
    Returns a list of dictionaries, each containing the post title, creation timestamp, and a list of media URIs.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for entry in data:
        title = entry.get("title", "")
        creation_timestamp = entry.get("creation_timestamp", 0)
        media_uris = [media.get("uri", "") for media in entry.get("media", []) if media.get("uri")]
        results.append({
            "title": title,
            "creation_timestamp": creation_timestamp,
            "media_uris": media_uris
        })

    return results

if __name__ == "__main__":
    json_paths = [
        r"data\instagram-days010601-2025-07-09-GyNYqGsQ\your_instagram_activity\media\posts_1.json",
        r"data\instagram-days010602-2025-07-11-cdZXWvWZ\your_instagram_activity\media\posts_1.json"
    ]

    all_posts = []
    for path in json_paths:
        try:
            posts = extract_posts(path)
            all_posts.extend(posts)
            print(f"{path} - Extracted {len(posts)} posts")
        except FileNotFoundError:
            print(f"File not found: {path}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in file: {path}")

    print("\nSample Output:")
    print(all_posts[:2])  # Print first 2 posts for brevity
