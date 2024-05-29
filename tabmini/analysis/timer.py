from math import e
import os
import signal
from typing import Callable


class TimeOutException(Exception):
    pass


def _handler(signum, frame):
    raise TimeOutException("Time limit was reached")


def _os_is_supported():
    return os.name == "posix"


# Throws TimeOutException if the function takes more than time_limit seconds to execute
def execute_with_timelimit(func: Callable[[], dict], time_limit: int = 60) -> dict:
    if not _os_is_supported():
        print("You're running a non-posix complient OS. Time limit can not be enforced for some methods")
        return func()

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(time_limit)
    try:
        result = func()
    except TimeOutException as e:
        signal.alarm(0)
        raise e
    finally:
        signal.alarm(0)

    return result
