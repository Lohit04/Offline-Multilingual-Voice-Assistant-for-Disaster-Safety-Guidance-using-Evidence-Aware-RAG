"""Offline test harness: Python imports this automatically at startup when its
folder is on PYTHONPATH. It makes the process behave as if there were NO
internet, and records every attempt to reach it:

  * socket.connect / connect_ex to any non-local address  -> blocked + logged
  * getaddrinfo of any dotted hostname (DNS to the internet) -> fails + logged

"Local" = loopback, private ranges (10/8, 172.16/12, 192.168/16), link-local.
That is exactly the traffic a phone-to-laptop hotspot demo needs; nothing else.
Log file: $OFFLINE_GUARD_LOG (default offline_guard.log in the cwd).
Limitation: it only sees traffic made through Python's socket module in THIS
process (not other programs such as Ollama, which is a separate local server).
"""

import ipaddress
import os
import socket
import threading

_LOG = os.environ.get("OFFLINE_GUARD_LOG", "offline_guard.log")
_lock = threading.Lock()
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_getaddrinfo = socket.getaddrinfo


def _record(kind, target):
    with _lock:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(f"{kind}\t{target}\n")


def _is_local_ip(host):
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None  # not an IP literal
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified


def _blocked_address(address):
    if isinstance(address, tuple) and address and isinstance(address[0], str):
        local = _is_local_ip(address[0])
        if local is None:  # a hostname
            return "." in address[0]
        return not local
    return False


def _connect(self, address):
    if _blocked_address(address):
        _record("CONNECT_BLOCKED", address)
        raise OSError(101, "offline guard: network unreachable")
    return _orig_connect(self, address)


def _connect_ex(self, address):
    if _blocked_address(address):
        _record("CONNECT_BLOCKED", address)
        return 101
    return _orig_connect_ex(self, address)


def _getaddrinfo(host, *args, **kwargs):
    if isinstance(host, str) and "." in host and _is_local_ip(host) is None:
        _record("DNS_BLOCKED", host)
        raise socket.gaierror(socket.EAI_NONAME, "offline guard: name resolution disabled")
    return _orig_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
socket.getaddrinfo = _getaddrinfo
