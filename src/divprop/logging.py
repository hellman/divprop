import coloredlogs
import logging
from datetime import datetime


# https://stackoverflow.com/questions/25194864/python-logging-time-since-start-of-program
class DeltaTimeColoredFormatter(coloredlogs.ColoredFormatter):
    def format(self, record):
        duration = datetime.utcfromtimestamp(record.relativeCreated / 1000)
        msecs = int(record.relativeCreated % 1000)
        record.delta = duration.strftime(f"%H:%M:%S.{msecs:03d}")
        return super().format(record)


class DeltaTimeFormatter(logging.Formatter):
    def format(self, record):
        duration = datetime.utcfromtimestamp(record.relativeCreated / 1000)
        msecs = int(record.relativeCreated % 1000)
        record.delta = duration.strftime(f"%H:%M:%S.{msecs:03d}")
        return super().format(record)


def setup(level='INFO'):
    coloredlogs.DEFAULT_FIELD_STYLES["levelname"]["color"] = 8

    formatter = DeltaTimeColoredFormatter(
        fmt="%(delta)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    now = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
    logger.info(f"starting at {now}")


def addFileHandler(filename):
    fh = logging.FileHandler(filename + '.debug.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(DeltaTimeFormatter(
        '%(delta)s.%(msecs)03d %(levelname)s %(name)s: %(message)s'
    ))
    logging.getLogger().addHandler(fh)

    fh = logging.FileHandler(filename + '.info.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(DeltaTimeFormatter(
        '%(delta)s.%(msecs)03d %(levelname)s %(name)s: %(message)s'
    ))
    logging.getLogger().addHandler(fh)


def getLogger(*args, **kwargs):
    return logging.getLogger(*args, **kwargs)