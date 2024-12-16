import logging
import colorlog

def setup_logger(name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'):
    formatter = colorlog.ColoredFormatter(
    "%(log_color)s" + format, 
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

