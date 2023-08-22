from typing import Any, Optional, NoReturn 
import loguru


def raise_exception_error(
    message: str, 
    logger: loguru._logger.Logger,
    exception_error: Optional[Any],
    logging_level: str = "error",
) -> NoReturn:
    """Simply wraps logger to raising exceptions.

    Parameters
    -----------
    message: str
    exception: Exception
    logger: Union[logging.Logger, None]
        if None is given, the exception message is not sending to logger.
    logging_level: str
        one of 'debug', 'info', 'warning', 'error', 'critical'
    """

    getattr(logger, logging_level)(exception_error.__name__ + ": " + message)
    raise exception_error(message)
    