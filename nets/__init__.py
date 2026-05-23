from .det.dbnet import DBNet
from .rec.rnn import CRNN


__all__ = ["build_model"]


def build_model(config):
    # Support both old format (name) and new Architecture format (algorithm)
    module_name = config.pop("algorithm", config.pop("name", None))
    support_dict = ["DBNet", "CRNN"]
    assert module_name in support_dict, "model {} is not supported".format(module_name)

    if module_name == "CRNN":
        # New Architecture format: extract Backbone, Neck, Head sections
        backbone_conf = config.pop("Backbone", {})
        neck_conf = config.pop("Neck", {})
        config.pop("Head", None)
        config.pop("Transform", None)
        config.pop("model_type", None)

        model = CRNN(
            classes_num=config.pop("classes_num"),
            rnn_type=neck_conf.get("rnn_type", "GRU"),
            hidden_size=neck_conf.get("hidden_size", 48),
            num_layers=neck_conf.get("num_layers", 2),
            bidirectional=neck_conf.get("bidirectional", True),
            backbone=backbone_conf,
        )
        return model

    if module_name == "DBNet":
        # Support both new Architecture format and old flat format
        backbone_conf = config.pop("Backbone", None)
        if backbone_conf is not None:
            # New format: parameters from Backbone/Neck/Head sections
            neck_conf = config.pop("Neck", {})
            head_conf = config.pop("Head", {})
            config.pop("Transform", None)
            config.pop("model_type", None)
            model = DBNet(
                inner_channel=neck_conf.get("inner_channel", 96),
                k=head_conf.get("k", 50),
                backbone=backbone_conf,
            )
        else:
            # Old format: parameters are flat in config
            backbone_conf = config.pop("backbone", {})
            model = DBNet(
                inner_channel=config.pop("inner_channel", 96),
                k=config.pop("k", 50),
                backbone=backbone_conf,
            )
        return model

    # Old format: pass remaining config directly
    module_class = eval(module_name)(**config)
    return module_class
