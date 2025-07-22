import json
from typing import List, Dict

def extract_post_comments(json_path: str) -> List[Dict[str, any]]:
    """
    Extracts all comments, media owners, and timestamps from the post_comments_1.json file.
    Returns a list of dictionaries, each containing the comment, media owner, and time.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for entry in data:
        string_map = entry.get("string_map_data", {})
        comment = string_map.get("Comment", {}).get("value", "")
        media_owner = string_map.get("Media Owner", {}).get("value", "")
        timestamp = string_map.get("Time", {}).get("timestamp", 0)
        results.append({
            "comment": comment,
            "media_owner": media_owner,
            "timestamp": timestamp
        })
    return results

# Example usage with multiple paths
if __name__ == "__main__":
    paths = [
        r"data\instagram-days010601-2025-07-09-GyNYqGsQ\your_instagram_activity\comments\post_comments_1.json",
        r"data\instagram-days010602-2025-07-11-cdZXWvWZ\your_instagram_activity\comments\post_comments_1.json",
        r"data\instagram-days010603-2025-07-11-5uV1CjIS\your_instagram_activity\comments\post_comments_1.json"
    ]

    for path in paths:
        try:
            comments = extract_post_comments(path)
            print(f"\n📄 {path} — Total Comments: {len(comments)}")
            print(json.dumps(comments[:3], indent=2))  # Show first 3 for preview
        except Exception as e:
            print(f"\n❌ Failed to process {path}: {e}")
