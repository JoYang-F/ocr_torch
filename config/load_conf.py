import os
import yaml
import json


class ReadConfig(object):
    def __init__(self, yml_path):
        self._yml_path = yml_path
        self.base_conf = self._read_yml_conf()
        self._complement_conf()

    def _read_yml_conf(self):
        with open(self._yml_path, "r", encoding="utf8") as f:
            conf = yaml.load(f.read(), Loader=yaml.FullLoader)
        return conf

    def _complement_conf(self):
        character_json_path = self.base_conf["global"].get("character_json_path", "")
        if not character_json_path:
            return
        if not os.path.exists(character_json_path):
            raise Exception("path {} not exists".format(character_json_path))

        try:
            with open(character_json_path, "r", encoding="utf8") as f:
                char2idx = json.loads(f.read())
        except Exception as e:
            raise e

        if "model_det" in self.base_conf.keys():
            self.base_conf["model_det"]["classes_num"] = len(char2idx)
        if "model" in self.base_conf.keys() and "classes_num" not in self.base_conf["model"].keys():
            self.base_conf["model"]["classes_num"] = len(char2idx)
        if "Architecture" in self.base_conf.keys() and "classes_num" not in self.base_conf["Architecture"].keys():
            self.base_conf["Architecture"]["classes_num"] = len(char2idx)
        if "post_process" in self.base_conf.keys():
            self.base_conf["post_process"]["character_json_path"] = character_json_path
