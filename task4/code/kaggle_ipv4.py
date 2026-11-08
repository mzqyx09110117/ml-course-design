import socket
import sys

_orig = socket.getaddrinfo


def _gai(host, port, family=0, type=0, proto=0, flags=0):
    return _orig(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _gai

from kaggle.cli import main  # noqa: E402

if __name__ == '__main__':
    sys.argv[0] = 'kaggle'
    main()
