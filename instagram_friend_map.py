import json
from typing import Dict, Any

def extract_friend_map(json_path: str) -> Dict[str, Any]:
    """
    Extracts incoming and outgoing friend requests from the Instagram friend map JSON file.
    Returns a dictionary with 'Incoming requests' and 'Outgoing requests' info.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for entry in data.get("profile_friend_map", []):
        string_map = entry.get("string_map_data", {})
        for key in ["Incoming requests", "Outgoing requests"]:
            if key in string_map:
                results[key] = {
                    "href": string_map[key].get("href", ""),
                    "value": string_map[key].get("value", ""),
                    "timestamp": string_map[key].get("timestamp", 0)
                }
    return results

# Example usage with multiple paths
if __name__ == "__main__":
    paths = [
        r"data\instagram-days010601-2025-07-09-GyNYqGsQ\personal_information\personal_information\instagram_friend_map.json",
        r"data\instagram-days010602-2025-07-11-cdZXWvWZ\personal_information\personal_information\instagram_friend_map.json",
        r"data\instagram-days010603-2025-07-11-5uV1CjIS\personal_information\personal_information\instagram_friend_map.json"
    ]

    for path in paths:
        try:
            friend_map = extract_friend_map(path)
            print(f"\n📄 {path}:\n{json.dumps(friend_map, indent=2)}")
        except Exception as e:
            print(f"\n❌ Failed to read {path}: {e}")
