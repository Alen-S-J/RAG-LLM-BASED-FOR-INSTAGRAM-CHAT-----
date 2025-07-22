import json
from typing import List

def extract_reels(json_path: str) -> List[dict]:
    """
    Extracts reels video URIs, creation timestamps, titles, and subtitle URIs from a reels.json file.
    Returns a list of dictionaries, each containing the video URI, creation timestamp, title, source app, and subtitle URI (if present).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for entry in data.get("ig_reels_media", []):
        for media in entry.get("media", []):
            uri = media.get("uri", "")
            creation_timestamp = media.get("creation_timestamp", 0)
            title = media.get("title", "")
            source_app = media.get("cross_post_source", {}).get("source_app", "")
            subtitle_uri = ""
            video_metadata = media.get("media_metadata", {}).get("video_metadata", {})
            if "subtitles" in video_metadata:
                subtitle_uri = video_metadata["subtitles"].get("uri", "")
            results.append({
                "uri": uri,
                "creation_timestamp": creation_timestamp,
                "title": title,
                "source_app": source_app,
                "subtitle_uri": subtitle_uri
            })
    return results

if __name__ == "__main__":
    json_paths = [
        r"data\instagram-days010601-2025-07-09-GyNYqGsQ\your_instagram_activity\media\reels.json",
        r"data\instagram-days010602-2025-07-11-cdZXWvWZ\your_instagram_activity\media\reels.json",
        r"data\instagram-days010603-2025-07-11-5uV1CjIS\your_instagram_activity\media\reels.json"
    ]

    all_reels = []
    for path in json_paths:
        try:
            reels = extract_reels(path)
            all_reels.extend(reels)
            print(f"{path} - Extracted {len(reels)} reels")
        except FileNotFoundError:
            print(f"File not found: {path}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in file: {path}")

    print("\nSample Output:")
    print(all_reels[:2])  # Print first 2 reels for brevity
