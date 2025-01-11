import httpx
import logging
import os
import time

logger = logging.getLogger(__name__)


class IanaTLDCacher:

    data_dir = "/src/data/iana"
    data_file = "/src/data/iana/tlds-alpha-by-domain.txt"
    data_src_url = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"

    def __init__(self):
        self.known_tlds: set[str] = {  # Hardcode a few common TLDs in case we can't fetch from IANA.
            # Infra TLD
            "arpa",

            # Country TLDs
            "ac", "ad", "ae", "af", "ag", "ai", "al", "am", "ao", "aq", "ar", "as", "at", "au", "aw", "ax", "az",
            "ba", "bb", "bd", "be", "bf", "bg", "bh", "bi", "bj", "bm", "bn", "bo", "bq", "br", "bs", "bt", "bw", "by",
            "bz",
            "ca", "cc", "cd", "cf", "cg", "ch", "ci", "ck", "cl", "cm", "cn", "co", "cr", "cu", "cv", "cw", "cx", "cy",
            "cz",
            "de", "dj", "dk", "dm", "do", "dz",
            "ec", "ee", "eg", "eh", "er", "es", "et", "eu",
            "fi", "fj", "fk", "fm", "fo", "fr",
            "ga", "gd", "ge", "gf", "gg", "gh", "gi", "gl", "gm", "gn", "gp", "gq", "gs", "gt", "gu", "gw", "gy",
            "hk", "hm", "hn", "hr", "ht", "hu",
            "id", "ie", "il", "im", "in", "io", "iq", "ir", "is", "it",
            "je", "jm", "jo", "jp",
            "ke", "kg", "kh", "ki", "km", "kn", "kp", "kr", "kw", "ky", "kz",
            "la", "lb", "lc", "li", "lk", "lr", "ls", "lt", "lu", "lv", "ly",
            "ma", "mc", "md", "me", "mg", "mh", "mk", "ml", "mm", "mn", "mo", "mp", "mq", "mr", "ms", "mt", "mu", "mv",
            "mw", "mx", "my", "mz",
            "na", "nc", "ne", "nf", "ng", "ni", "nl", "no", "np", "nr", "nu", "nz",
            "om", "pa", "pe", "pf", "pg", "ph", "pk", "pl", "pm", "pn", "pr", "ps", "pt", "pw", "py",
            "qa",
            "re", "ro", "rs", "ru", "rw",
            "sa", "sb", "sc", "sd", "se", "sg", "sh", "si", "sk", "sl", "sm", "sn", "so", "sr", "ss", "st", "su", "sv",
            "sx", "sy", "sz",
            "tc", "td", "tf", "tg", "th", "tj", "tk", "tl", "tm", "tn", "to", "tr", "tt", "tv", "tw", "tz",
            "ua", "ug", "uk", "us", "uy", "uz",
            "va", "vc", "ve", "vg", "vi", "vn", "vu",
            "wf", "ws",
            "ye", "yt",
            "za", "zm", "zw",

            # Common generic TLDs
            "agency",
            "app",
            "biz",
            "bid",
            "blog",
            "club",
            "co",
            "com",
            "design",
            "dev",
            "info",
            "loan",
            "ltd",
            "me",
            "men",
            "net",
            "online",
            "org",
            "shop",
            "site",
            "store",
            "stream",
            "tech",
            "top",
            "vip",
            "win",
            "work",
            "xyz",

            # Sponsored TLDs
            "aero",
            "asia",
            "bank",
            "cat",
            "coop",
            "edu",
            "gov",
            "int",
            "jobs",
            "mil",
            "museum",
            "post",
            "tel",
            "travel",
            "xxx",
        }
        if not os.path.exists(IanaTLDCacher.data_dir):
            os.mkdir(IanaTLDCacher.data_dir)
        if not os.path.exists(IanaTLDCacher.data_file) or (
                time.time() - os.path.getmtime(IanaTLDCacher.data_file) > 60 * 60 * 24 * 7):
            try:
                resp = httpx.get(IanaTLDCacher.data_src_url)
                with open(IanaTLDCacher.data_file, "wb") as fd:
                    fd.write(resp.content)
            except httpx.RequestError as e:
                logger.error("Failed to download TLD data, try again later")
        if os.path.exists(IanaTLDCacher.data_file):
            with open(IanaTLDCacher.data_file) as fd:
                for line in fd:
                    line = line.strip()
                    if line.startswith("#"):
                        continue
                    self.known_tlds.add(line.lower())
                logger.debug(f"loaded {len(self.known_tlds)} TLDs from {IanaTLDCacher.data_file}, "
                             f"modified at {os.path.getmtime(IanaTLDCacher.data_file)}: {self.known_tlds}")
