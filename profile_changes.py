import json

def extract_profile_changes(json_path):
    """
    Extracts all profile changes from the profile_changes.json file.
    Returns a list of dictionaries, each containing the change type, previous value, new value, and change date.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = []
    for entry in data.get("profile_profile_change", []):
        string_map = entry.get("string_map_data", {})
        change = {
            "Changed": string_map.get("Changed", {}).get("value", ""),
            "Previous Value": string_map.get("Previous Value", {}).get("value", ""),
            "New Value": string_map.get("New Value", {}).get("value", ""),
            "Change Date": string_map.get("Change Date", {}).get("timestamp", 0)
        }
        results.append(change)
    return results

# Example usage:
if __name__ == "__main__":
    changes = extract_profile_changes("Data/personal_information/personal_information/profile_changes.json")
    print("profile_changes.json:", changes)
