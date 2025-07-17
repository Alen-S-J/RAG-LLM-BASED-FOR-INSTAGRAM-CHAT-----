import json
from typing import Dict, Any, List
from pathlib import Path

def extract_ads_info(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}

    # Handle "ads_about_meta.json"
    if "label_values" in data:
        for item in data.get("label_values", []):
            label = item.get("label")
            value = item.get("value") or item.get("timestamp_value")
            result[label] = value
        result["fbid"] = data.get("fbid")

    # Handle "videos_watched.json"
    elif isinstance(data, list):
        result["videos_watched"] = []
        for item in data:
            title = item.get("title")
            timestamp = item.get("timestamp")
            if title and timestamp:
                result["videos_watched"].append({"title": title, "timestamp": timestamp})

    else:
        result["error"] = "Unknown JSON structure"

    return result

# Example usage with both files
paths = [
    r"data\instagram-days010601-2025-07-09-GyNYqGsQ\ads_information\instagram_ads_and_businesses\ads_about_meta.json",
    r"data\instagram-days010603-2025-07-11-5uV1CjIS\ads_information\ads_and_topics\videos_watched.json"
]

for path in paths:
    print(f"\nExtracting from: {path}")
    try:
        info = extract_ads_info(path)
        print(json.dumps(info, indent=2))
    except Exception as e:
        print(f"Failed to read {path}: {e}")
