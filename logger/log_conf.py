import os


def logging_conf(log_path, level="INFO"):
    """
    level可选参数为DEBUG, INFO, WARN, ERROR
    """
    return {
        "loggers": {
            "mail": {
                "level": "CRITICAL",
                "propagate": False,
                "handlers": ["mail"]
            },
            "logs": {
                "level": level,
                "propagate": False,
                "handlers": ["logs", "console"]
            },
            "console": {
                "level": level,
                "propagate": False,
                "handlers": ["console"]
            },
        },
        "disable_existing_loggers": False,
        "handlers": {
            "logs": {
                "formatter": "simple",
                "backupCount": 10,
                "class": "logging.handlers.RotatingFileHandler",
                "maxBytes": 10485760,
                "filename": os.path.join(log_path, "log.txt")
            },
            "console": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout"
            },
            "mail": {
                "toaddrs": [""],
                "mailhost": ["smtp.exmail.qq.com", 25],
                "fromaddr": "",
                "level": "CRITICAL",
                "credentials": ["", ""],
                "formatter": "mail",
                "class": "logging.handlers.SMTPHandler",
                "subject": "XXXXX"
            }
        },
        "formatters": {
            "default": {
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "format": "%(asctime)s - %(levelname)s - %(module)s.%(name)s : \n%(message)s"
            },
            "simple": {
                "format": "%(asctime)s - %(levelname)s - %(message)s"
            },
            "mail": {
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "format": "%(asctime)s : %(message)s"
            }
        },
        "version": 1
    }
