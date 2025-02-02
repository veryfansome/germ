import logging

from bot.lang.parsers import extract_href_features, iana_data

logger = logging.getLogger(__name__)


def test_extract_href_features():
    iana_data.load_tld_cache()

    artifact = extract_href_features("https://www.google.com")
    assert "scheme" in artifact and artifact["scheme"] == "https", artifact
    assert "fqdn" in artifact and artifact["fqdn"] == "www.google.com", artifact
    assert "port" not in artifact, artifact
    assert "path" not in artifact, artifact
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
