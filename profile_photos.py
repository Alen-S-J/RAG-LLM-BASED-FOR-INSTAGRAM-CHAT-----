import json
from typing import List

def extract_profile_photos(json_path: str) -> List[dict]:
    """
    Extracts profile photo URIs, creation timestamps, and source app from a profile_photos.json file.
    Returns a list of dictionaries, each containing the uri, creation_timestamp, and source_app.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for entry in data.get("ig_profile_picture", []):
        uri = entry.get("uri", "")
        creation_timestamp = entry.get("creation_timestamp", 0)
        source_app = entry.get("cross_post_source", {}).get("source_app", "")
        results.append({
            "uri": uri,
            "creation_timestamp": creation_timestamp,
            "source_app": source_app
        })

    return results

if __name__ == "__main__":
    json_paths = [
        r"1-data\instagram-days010601-2025-07-09-GyNYqGsQ\your_instagram_activity\media\profile_photos.json",
        r"2-data\instagram-days010602-2025-07-11-cdZXWvWZ\your_instagram_activity\media\profile_photos.json",
        r"3-data\instagram-days010603-2025-07-11-5uV1CjIS\your_instagram_activity\media\profile_photos.json"
    ]

    all_photos = []
    for path in json_paths:
        try:
            photos = extract_profile_photos(path)
            all_photos.extend(photos)
            print(f"{path} - Extracted {len(photos)} profile photos")
        except FileNotFoundError:
            print(f"File not found: {path}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in file: {path}")

    print("\nSample Output:")
    print(all_photos[:2])  # Print first 2 profile photos for brevity
