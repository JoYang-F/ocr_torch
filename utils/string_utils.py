import json


class CharacterJson(object):
    def __init__(self, character_json_path):
        self._read_character_dict(character_json_path)

    def _read_character_dict(self, character_json_path):
        with open(character_json_path, "r", encoding="utf8") as f:
            char2idx = json.loads(f.read())

        self.char2idx = char2idx
        idx2char = {val: key for key, val in char2idx.items()}
        self.idx2char = idx2char
        self.classes_num = len(char2idx)
