import json
from typing import Dict, Any

def extract_personal_information(json_path: str) -> Dict[str, Any]:
    """
    Extracts all key-value pairs from the personal_information.json file's top-level keys.
    Returns a dictionary with all available fields and their values.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for key, value in data.items():
        results[key] = value
    return results

# Example usage with multiple paths
if __name__ == "__main__":
    paths = [
        r"data\instagram-days010601-2025-07-09-GyNYqGsQ\personal_information\personal_information\personal_information.json",
        r"data\instagram-days010602-2025-07-11-cdZXWvWZ\personal_information\personal_information\personal_information.json",
        r"data\instagram-days010603-2025-07-11-5uV1CjIS\personal_information\personal_information\personal_information.json"
    ]

    for path in paths:
        try:
            personal_info = extract_personal_information(path)
            print(f"\n📄 {path}:\n{json.dumps(personal_info, indent=2)}")
        except Exception as e:
            print(f"\n❌ Failed to read {path}: {e}")
