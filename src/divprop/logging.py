import coloredlogs
import logging


def setup(level='VERBOSE'):
    coloredlogs.DEFAULT_FIELD_STYLES["levelname"]["color"] = 8
    coloredlogs.install(
        level=level,
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        msecs=True,
    )


def addFileHandler(filename):
    fh = logging.FileHandler(filename + '.debug.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s')
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

    fh = logging.FileHandler(filename + '.info.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s')
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)


def getLogger(*args, **kwargs):
    return logging.getLogger(*args, **kwargs)
