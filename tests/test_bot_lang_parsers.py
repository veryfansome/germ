import logging

from bot.lang.parsers import resolve_fqdn

logger = logging.getLogger(__name__)


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
