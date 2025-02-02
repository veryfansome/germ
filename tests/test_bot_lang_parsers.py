import logging

from bot.lang.parsers import (extract_href_features, fqdn_to_proper_noun, resolve_fqdn)

logger = logging.getLogger(__name__)


def test_extract_href_features():
    artifact = extract_href_features("/")
    assert "scheme" in artifact and artifact["scheme"] == "relative", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "relative", artifact
    assert "port" not in artifact, artifact
    assert "path" in artifact and artifact["path"] == "/", artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("/ui.css")
    assert "scheme" in artifact and artifact["scheme"] == "relative", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "relative", artifact
    assert "port" not in artifact, artifact
    assert "path" in artifact and artifact["path"] == "/ui.css", artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("https://www.google.com")
    assert "scheme" in artifact and artifact["scheme"] == "https", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "www.google.com", artifact
    assert "port" not in artifact, artifact
    assert "path" not in artifact, artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("http://localhost:8080")
    assert "scheme" in artifact and artifact["scheme"] == "http", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "localhost", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" not in artifact, artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("localhost:8080")
    assert "scheme" in artifact and artifact["scheme"] == "relative", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "localhost", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" not in artifact, artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("localhost:8080/index.html")
    assert "scheme" in artifact and artifact["scheme"] == "relative", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "localhost", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" in artifact and artifact["path"] == "/index.html", artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("https://example.com:8080?foo=foo#bar")
    assert "scheme" in artifact and artifact["scheme"] == "https", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "example.com", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" not in artifact, artifact
    assert "query" in artifact and artifact["query"] == "?foo=foo#bar", artifact

    artifact = extract_href_features("https://example.com:8080#bar")
    assert "scheme" in artifact and artifact["scheme"] == "https", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "example.com", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" not in artifact, artifact
    assert "query" in artifact and artifact["query"] == "#bar", artifact

    artifact = extract_href_features("https://example.me:8080/?foo=foo#bar")
    assert "scheme" in artifact and artifact["scheme"] == "https", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "example.me", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" in artifact and artifact["path"] == "/", artifact
    assert "query" in artifact and artifact["query"] == "?foo=foo#bar", artifact

    artifact = extract_href_features("https://example.me:8080/index.php?foo=foo")
    assert "scheme" in artifact and artifact["scheme"] == "https", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "example.me", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" in artifact and artifact["path"] == "/index.php", artifact
    assert "query" in artifact and artifact["query"] == "?foo=foo", artifact

    artifact = extract_href_features("https://example.me:8080/ui.js?foo=foo")
    assert "scheme" in artifact and artifact["scheme"] == "https", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "example.me", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" in artifact and artifact["path"] == "/ui.js", artifact
    assert "query" in artifact and artifact["query"] == "?foo=foo", artifact

    artifact = extract_href_features("https://example.photography:8080/path/to/some/object?foo=foo")
    assert "scheme" in artifact and artifact["scheme"] == "https", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "example.photography", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" in artifact and artifact["path"] == "/path/to/some/object", artifact
    assert "query" in artifact and artifact["query"] == "?foo=foo", artifact

    artifact = extract_href_features("http://127.0.0.1:8080")
    assert "scheme" in artifact and artifact["scheme"] == "http", artifact
    assert "ipv4_address" in artifact and artifact["ipv4_address"] == "127.0.0.1", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" not in artifact, artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("http://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:8080")
    assert "scheme" in artifact and artifact["scheme"] == "http", artifact
    assert "ipv6_address" in artifact and artifact["ipv6_address"] == "2001:0db8:85a3:0000:0000:8a2e:0370:7334", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" not in artifact, artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("http://[2001:db8::1]:8080")
    assert "scheme" in artifact and artifact["scheme"] == "http", artifact
    assert "ipv6_address" in artifact and artifact["ipv6_address"] == "2001:db8::1", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" not in artifact, artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("http://[::1]:8080")
    assert "scheme" in artifact and artifact["scheme"] == "http", artifact
    assert "ipv6_address" in artifact and artifact["ipv6_address"] == "::1", artifact
    assert "port" in artifact and artifact["port"] == "8080", artifact
    assert "path" not in artifact, artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("page.html")
    assert "scheme" in artifact and artifact["scheme"] == "relative", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "relative", artifact
    assert "port" not in artifact, artifact
    assert "path" in artifact and artifact["path"] == "page.html", artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("page.html#section1")
    assert "scheme" in artifact and artifact["scheme"] == "relative", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "relative", artifact
    assert "port" not in artifact, artifact
    assert "path" in artifact and artifact["path"] == "page.html", artifact
    assert "query" in artifact and artifact["query"] == "#section1", artifact

    artifact = extract_href_features("folder/page.html")
    assert "scheme" in artifact and artifact["scheme"] == "relative", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "relative", artifact
    assert "port" not in artifact, artifact
    assert "path" in artifact and artifact["path"] == "folder/page.html", artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("../page.html")
    assert "scheme" in artifact and artifact["scheme"] == "relative", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "relative", artifact
    assert "port" not in artifact, artifact
    assert "path" in artifact and artifact["path"] == "../page.html", artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("//www.google.com")
    assert "scheme" in artifact and artifact["scheme"] == "relative", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "www.google.com", artifact
    assert "port" not in artifact, artifact
    assert "path" not in artifact, artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("www.google.com")
    assert "scheme" in artifact and artifact["scheme"] == "relative", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "www.google.com", artifact
    assert "port" not in artifact, artifact
    assert "path" not in artifact, artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("mailto:someone@example.com?subject=Hello&body=Message")
    assert "skipped" in artifact and artifact["skipped"], artifact
    assert "reason" in artifact and artifact["reason"] == "unsupported_scheme", artifact

    artifact = extract_href_features("tel:+1234567890")
    assert "skipped" in artifact and artifact["skipped"], artifact
    assert "reason" in artifact and artifact["reason"] == "unsupported_scheme", artifact

    artifact = extract_href_features("file:///C:/path/to/file.txt")
    assert "scheme" in artifact and artifact["scheme"] == "file", artifact
    assert "fqdn" not in artifact, artifact
    assert "port" not in artifact, artifact
    assert "path" in artifact and artifact["path"] == "/C:/path/to/file.txt", artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("file:///src/README.md")
    assert "scheme" in artifact and artifact["scheme"] == "file", artifact
    assert "fqdn" not in artifact, artifact
    assert "port" not in artifact, artifact
    assert "path" in artifact and artifact["path"] == "/src/README.md", artifact
    assert "query" not in artifact, artifact

    artifact = extract_href_features("javascript:alert('Hello World!')")
    assert "skipped" in artifact and artifact["skipped"], artifact
    assert "reason" in artifact and artifact["reason"] == "unsupported_scheme", artifact

    artifact = extract_href_features("data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==")
    assert "skipped" in artifact and artifact["skipped"], artifact
    assert "reason" in artifact and artifact["reason"] == "unsupported_scheme", artifact


def test_fqdn_to_proper_noun():
    word = fqdn_to_proper_noun("www.google.com")
    assert word == "GoogleDOTcom", word


def test_resolve_fqdn():
    answer, err, timed_out = resolve_fqdn('www.google.com')
    assert answer is not None, f"err:{err}, timed_out={timed_out}"
    assert err is False, f"err:{err}, timed_out={timed_out}"
    assert timed_out is False, f"err:{err}, timed_out={timed_out}"

    answer, err, timed_out = resolve_fqdn('wwwww.google.com')
    assert answer is None, f"err:{err}, timed_out={timed_out}"
    assert err is True, f"err:{err}, timed_out={timed_out}"
    assert timed_out is False, f"err:{err}, timed_out={timed_out}"

    answer, err, timed_out = resolve_fqdn('www.google.com', nameservers=["127.0.0.1"])
    assert answer is None, f"err:{err}, timed_out={timed_out}"
    assert err is False, f"err:{err}, timed_out={timed_out}"
    assert timed_out is True, f"err:{err}, timed_out={timed_out}"
