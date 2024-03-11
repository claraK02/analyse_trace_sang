import os
import json

MATCHING_TABLE_PATH = os.path.join('data', 'process_real_data', 'matching_table.json')


def get_matching_table(filepath: str = MATCHING_TABLE_PATH
                       ) -> dict[str, str | None]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    
    with open(file=filepath, encoding='utf8') as f:
        matching_table: dict[str, str | None] = json.load(f)
        f.close()

    for key, value in matching_table.items():
        if value == "":
            matching_table[key] = None

    return matching_table

if __name__ == '__main__':
    from icecream import ic
    matching_table = get_matching_table()
    ic(matching_table)
