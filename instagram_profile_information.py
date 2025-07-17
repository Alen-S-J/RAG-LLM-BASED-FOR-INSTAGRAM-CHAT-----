import json
from typing import Dict, Any

def extract_profile_account_insights(json_path: str) -> Dict[str, Any]:
    """
    Extracts key profile account insights from the Instagram profile information JSON file.
    Returns a dictionary with all available fields in 'string_map_data'.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for entry in data.get("profile_account_insights", []):
        string_map = entry.get("string_map_data", {})
        for key, value in string_map.items():
            results[key] = {
                "href": value.get("href", ""),
                "value": value.get("value", ""),
                "timestamp": value.get("timestamp", 0)
            }
    return results

# Example usage with multiple paths
if __name__ == "__main__":
    paths = [
        r"data\instagram-days010601-2025-07-09-GyNYqGsQ\personal_information\personal_information\instagram_profile_information.json",
        r"data\instagram-days010602-2025-07-11-cdZXWvWZ\personal_information\personal_information\instagram_profile_information.json",
        r"data\instagram-days010603-2025-07-11-5uV1CjIS\personal_information\personal_information\instagram_profile_information.json"
    ]

    for path in paths:
        try:
            profile_info = extract_profile_account_insights(path)
            print(f"\n📄 {path}:\n{json.dumps(profile_info, indent=2)}")
        except Exception as e:
            print(f"\n❌ Failed to read {path}: {e}")
