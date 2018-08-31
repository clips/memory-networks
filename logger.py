import logging


def setup_logging(level, filename):
    """
    :param level: logging.INFO or logging.DEBUG, etc.
    """
    logging.basicConfig(level=level, filename=filename, filemode="a+", format="%(message)s")

    return logging.getLogger(__name__)

def get_logger(filename):
    return setup_logging(logging.DEBUG, filename=filename)