from datetime import datetime, timezone
import httpx
import logging

from bot.db.models import SessionLocal, TopLevelDomainName

logger = logging.getLogger(__name__)


class IanaTLDCacher:

    data_src_url = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"

    def __init__(self):
        self.known_tld_names: set[str] = set()

        self.load_from_db()
        self.load_from_iana()

    def is_possible_public_fqdn(self, fqdn: str) -> bool:
        return fqdn.split(".")[-1] in self.known_tld_names

    def load_from_db(self):
        num_before_load = len(self.known_tld_names)
        try:
            with SessionLocal() as session:
                for tld_record in session.query(TopLevelDomainName).all():
                    self.known_tld_names.add(tld_record.top_level_domain_name)
            logger.info(f"loaded {len(self.known_tld_names) - num_before_load} TLD records from PostgreSQL")
        except httpx.RequestError as e:
            logger.error("failed to load TLD records from PostgreSQL", e)

    def load_from_iana(self):
        num_before_load = len(self.known_tld_names)
        try:
            resp = httpx.get(IanaTLDCacher.data_src_url)
            with SessionLocal() as session:
                for line in resp.text.splitlines():
                    line = line.strip()
                    if line.startswith("#"):
                        continue
                    tld_name = line.lower()
                    if tld_name not in self.known_tld_names:
                        self.known_tld_names.add(tld_name)
                        tld_record = TopLevelDomainName(top_level_domain_name=line,
                                                        time_created=datetime.now(timezone.utc))
                        session.add(tld_record)
                    else:
                        tld_record = session.query(TopLevelDomainName).filter_by(
                            top_level_domain_name=tld_name).first()
                        if tld_record:
                            tld_record.time_last_validated = datetime.now(timezone.utc)
                session.commit()
            logger.info(f"added {len(self.known_tld_names) - num_before_load} TLDs from {IanaTLDCacher.data_src_url}")
        except httpx.RequestError as e:
            logger.error("failed to download TLD data from IANA")
