
import ipaddress
import logging
import socket
import urllib.parse
from typing import Optional

import requests

logger = logging.getLogger("FactGuard.Security")

# Blocked IP ranges: loopback, private RFC1918, link-local, cloud metadata, multicast
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # IPv4 loopback
    ipaddress.ip_network("10.0.0.0/8"),        # RFC1918 private
    ipaddress.ip_network("172.16.0.0/12"),     # RFC1918 private
    ipaddress.ip_network("192.168.0.0/16"),    # RFC1918 private
    ipaddress.ip_network("169.254.0.0/16"),    # Link-local / Cloud metadata (AWS/GCP/Azure)
    ipaddress.ip_network("100.64.0.0/10"),     # Carrier-grade NAT
    ipaddress.ip_network("0.0.0.0/8"),         # Current network
    ipaddress.ip_network("224.0.0.0/4"),       # Multicast
    ipaddress.ip_network("240.0.0.0/4"),       # Reserved
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]

# Maximum allowed download size for text/evidence (5 MB)
MAX_DOWNLOAD_SIZE_BYTES = 5 * 1024 * 1024


class SafetyCheckResult(tuple):
    """2-tuple (is_safe, reason) that evaluates truthy only if is_safe is True."""
    def __new__(cls, is_safe: bool, reason: str):
        return super().__new__(cls, (bool(is_safe), str(reason)))

    def __bool__(self):
        return bool(self[0])

    @property
    def is_safe(self) -> bool:
        return bool(self[0])

    @property
    def reason(self) -> str:
        return str(self[1])


def is_safe_url(url: str) -> SafetyCheckResult:
    """
    Validate that a URL is well-formed and does not resolve to loopback,
    private IP ranges, or cloud metadata services (SSRF protection).
    Returns SafetyCheckResult(is_safe, reason).
    """
    if not url or not isinstance(url, str):
        return SafetyCheckResult(False, "URL must be a non-empty string")

    url = url.strip()
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return SafetyCheckResult(False, f"Invalid URL syntax: {e}")

    if parsed.scheme.lower() not in ("http", "https"):
        return SafetyCheckResult(False, f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed.")

    hostname = parsed.hostname
    if not hostname:
        return SafetyCheckResult(False, "URL missing valid hostname")

    # Block common local hostnames
    lower_host = hostname.lower()
    if lower_host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "metadata.google.internal", "instance-data"):
        return SafetyCheckResult(False, f"Access to restricted hostname '{hostname}' is blocked.")

    # Resolve hostname to IP addresses and check all
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        if not addr_info:
            return SafetyCheckResult(False, f"Could not resolve hostname '{hostname}'.")

        for item in addr_info:
            ip_str = item[4][0]
            ip_obj = ipaddress.ip_address(ip_str)

            # Check for IPv6-mapped IPv4 address (e.g. ::ffff:127.0.0.1)
            if ip_obj.version == 6 and getattr(ip_obj, "ipv4_mapped", None):
                ip_obj = ip_obj.ipv4_mapped

            for blocked_net in _BLOCKED_NETWORKS:
                if ip_obj in blocked_net:
                    logger.warning("SSRF blocked: '%s' resolved to restricted IP %s in %s", hostname, ip_str, blocked_net)
                    return SafetyCheckResult(False, f"Resolved IP {ip_str} belongs to a restricted internal network.")
    except socket.gaierror:
        # If DNS cannot be resolved, reject safely
        return SafetyCheckResult(False, f"Hostname '{hostname}' cannot be resolved.")
    except Exception as e:
        return SafetyCheckResult(False, f"Error validating hostname '{hostname}': {e}")

    return SafetyCheckResult(True, "URL is safe")


def sanitize_text(text: str, max_chars: int = 50000) -> str:
    """
    Sanitize untrusted text input against injection attacks:
    - Removes NULL bytes (\x00)
    - Strips non-printable ASCII control characters (keeping \t, \n, \r)
    - Strips zero-width invisible characters (\u200b, \u200c, \u200d, \ufeff)
    - Truncates to max_chars safely
    """
    if not text or not isinstance(text, str):
        return ""
    cleaned = text.replace("\x00", "").replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    cleaned = "".join(ch for ch in cleaned if ch in ("\t", "\n", "\r") or (ord(ch) >= 32 and ord(ch) != 127))
    return cleaned[:max_chars].strip()


def validate_url(url: str) -> str:
    """Validate URL and return sanitized string if safe, else raise ValueError."""
    result = is_safe_url(url)
    if not result:
        raise ValueError(result.reason)
    return url.strip()


def safe_get(
    url: str,
    timeout: int = 10,
    headers: Optional[dict] = None,
    max_bytes: int = MAX_DOWNLOAD_SIZE_BYTES
) -> requests.Response:
    """
    Perform an SSRF-safe HTTP GET request with timeouts and response size limits.
    Raises ValueError if URL is unsafe or exceeds max_bytes.
    """
    is_safe, reason = is_safe_url(url)
    if not is_safe:
        raise ValueError(f"Blocked unsafe URL: {reason}")

    req_headers = {
        "User-Agent": "FactGuardAI/1.3.0 (+https://github.com/factguard-ai; verification-bot)"
    }
    if headers:
        req_headers.update(headers)

    # Stream response to enforce size limit
    with requests.get(url, headers=req_headers, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        content = bytearray()
        for chunk in resp.iter_content(chunk_size=8192):
            content.extend(chunk)
            if len(content) > max_bytes:
                raise ValueError(f"Response size exceeded maximum allowable limit of {max_bytes} bytes.")

        # Reconstruct response content
        resp._content = bytes(content)
        return resp
