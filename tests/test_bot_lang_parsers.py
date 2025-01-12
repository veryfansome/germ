import logging

from bot.lang.parsers import extract_href_features, resolve_fqdn

logger = logging.getLogger(__name__)


def test_extract_href_features():
    artifact = extract_href_features("/")
    assert artifact["scheme"] == "relative"
    assert artifact["fqdn"] == "relative"
    assert "port" not in artifact
    assert artifact["path"] == "/"
    assert "query" not in artifact

    artifact = extract_href_features("/ui.css")
    assert artifact["scheme"] == "relative"
    assert artifact["fqdn"] == "relative"
    assert "port" not in artifact
    assert artifact["path"] == "/ui.css"
    assert "query" not in artifact

    artifact = extract_href_features("https://www.google.com")
    assert artifact["scheme"] == "https"
    assert artifact["fqdn"] == "www.google.com"
    assert "port" not in artifact
    assert "path" not in artifact
    assert "query" not in artifact

    artifact = extract_href_features("http://localhost:8080")
    assert artifact["scheme"] == "http"
    assert artifact["fqdn"] == "localhost"
    assert artifact["port"] == "8080"
    assert "path" not in artifact
    assert "query" not in artifact

    artifact = extract_href_features("localhost:8080")
    assert artifact["scheme"] == "relative"
    assert artifact["fqdn"] == "localhost"
    assert artifact["port"] == "8080"
    assert "path" not in artifact
    assert "query" not in artifact

    artifact = extract_href_features("localhost:8080/index.html")
    assert artifact["scheme"] == "relative"
    assert artifact["fqdn"] == "localhost"
    assert artifact["port"] == "8080"
    assert artifact["path"] == "/index.html"
    assert "query" not in artifact

    artifact = extract_href_features("https://example.com:8080?foo=foo#bar")
    assert artifact["scheme"] == "https"
    assert artifact["fqdn"] == "example.com"
    assert artifact["port"] == "8080"
    assert "path" not in artifact
    assert artifact["query"] == "?foo=foo#bar"

    artifact = extract_href_features("https://example.com:8080#bar")
    assert artifact["scheme"] == "https"
    assert artifact["fqdn"] == "example.com"
    assert artifact["port"] == "8080"
    assert "path" not in artifact
    assert artifact["query"] == "#bar"

    artifact = extract_href_features("https://example.me:8080/?foo=foo#bar")
    assert artifact["scheme"] == "https"
    assert artifact["fqdn"] == "example.me"
    assert artifact["port"] == "8080"
    assert artifact["path"] == "/"
    assert artifact["query"] == "?foo=foo#bar"

    artifact = extract_href_features("https://example.me:8080/index.php?foo=foo")
    assert artifact["scheme"] == "https"
    assert artifact["fqdn"] == "example.me"
    assert artifact["port"] == "8080"
    assert artifact["path"] == "/index.php"
    assert artifact["query"] == "?foo=foo"

    artifact = extract_href_features("https://example.me:8080/ui.js?foo=foo")
    assert artifact["scheme"] == "https"
    assert artifact["fqdn"] == "example.me"
    assert artifact["port"] == "8080"
    assert artifact["path"] == "/ui.js"
    assert artifact["query"] == "?foo=foo"

    artifact = extract_href_features("https://example.photography:8080/path/to/some/object?foo=foo")
    assert artifact["scheme"] == "https"
    assert artifact["fqdn"] == "example.photography"
    assert artifact["port"] == "8080"
    assert artifact["path"] == "/path/to/some/object"
    assert artifact["query"] == "?foo=foo"

    artifact = extract_href_features("http://127.0.0.1:8080")
    assert artifact["scheme"] == "http"
    assert artifact["fqdn"] == "127.0.0.1"
    assert artifact["port"] == "8080"
    assert "path" not in artifact
    assert "query" not in artifact

    artifact = extract_href_features("page.html")
    assert artifact["scheme"] == "relative"
    assert artifact["fqdn"] == "relative"
    assert "port" not in artifact
    assert artifact["path"] == "page.html"
    assert "query" not in artifact

    artifact = extract_href_features("page.html#section1")
    assert artifact["scheme"] == "relative"
    assert artifact["fqdn"] == "relative"
    assert "port" not in artifact
    assert artifact["path"] == "page.html"
    assert artifact["query"] == "#section1"

    artifact = extract_href_features("folder/page.html")
    assert artifact["scheme"] == "relative"
    assert artifact["fqdn"] == "relative"
    assert "port" not in artifact
    assert artifact["path"] == "folder/page.html"
    assert "query" not in artifact

    artifact = extract_href_features("../page.html")
    assert artifact["scheme"] == "relative"
    assert artifact["fqdn"] == "relative"
    assert "port" not in artifact
    assert artifact["path"] == "../page.html"
    assert "query" not in artifact

    artifact = extract_href_features("//www.google.com")
    assert artifact["scheme"] == "relative"
    assert artifact["fqdn"] == "www.google.com"
    assert "port" not in artifact
    assert "path" not in artifact
    assert "query" not in artifact

    artifact = extract_href_features("www.google.com")
    assert artifact["scheme"] == "relative"
    assert artifact["fqdn"] == "www.google.com"
    assert "port" not in artifact
    assert "path" not in artifact
    assert "query" not in artifact

    artifact = extract_href_features("mailto:someone@example.com?subject=Hello&body=Message")
    assert artifact["skipped"]
    assert artifact["reason"] == "unsupported_scheme"

    artifact = extract_href_features("tel:+1234567890")
    assert artifact["skipped"]
    assert artifact["reason"] == "unsupported_scheme"

    artifact = extract_href_features("file:///C:/path/to/file.txt")
    assert artifact["scheme"] == "file"
    assert "fqdn" not in artifact
    assert "port" not in artifact
    assert artifact["path"] == "/C:/path/to/file.txt"
    assert "query" not in artifact

    artifact = extract_href_features("file:///src/README.md")
    assert artifact["scheme"] == "file"
    assert "fqdn" not in artifact
    assert "port" not in artifact
    assert artifact["path"] == "/src/README.md"
    assert "query" not in artifact

    artifact = extract_href_features("javascript:alert('Hello World!')")
    assert artifact["skipped"]
    assert artifact["reason"] == "unsupported_scheme"

    artifact = extract_href_features("data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==")
    assert artifact["skipped"]
    assert artifact["reason"] == "unsupported_scheme"


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
