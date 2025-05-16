from datetime import datetime, timezone
from sqlalchemy import MetaData, Table, create_engine as create_pg_engine, update
from sqlalchemy.orm import sessionmaker
import httpx
import logging

from germ.bot.db.pg import DATABASE_URL

logger = logging.getLogger(__name__)

pg_engine = create_pg_engine(f"postgresql+psycopg2://{DATABASE_URL}", echo=False)
pg_session_maker = sessionmaker(autocommit=False, autoflush=False, bind=pg_engine)
top_level_domain_table = Table('top_level_domain', MetaData(), autoload_with=pg_engine)


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
            with pg_session_maker() as session:
                for tld_record in session.execute(top_level_domain_table.select()).fetchall():
                    self.known_tld_names.add(tld_record.domain_name)
            logger.info(f"loaded {len(self.known_tld_names) - num_before_load} TLD records from PostgreSQL")
        except httpx.RequestError as e:
            logger.error("failed to load TLD records from PostgreSQL", e)

    def load_from_iana(self):
        num_before_load = len(self.known_tld_names)
        try:
            resp = httpx.get(IanaTLDCacher.data_src_url)
            with pg_session_maker() as session:
                for line in resp.text.splitlines():
                    line = line.strip()
                    if line.startswith("#"):
                        continue
                    tld_name = line.lower()

                    dt_now = datetime.now(timezone.utc)
                    if tld_name not in self.known_tld_names:
                        self.known_tld_names.add(tld_name)
                        top_level_domain_table.insert().values({"domain_name": tld_name})
                    else:
                        tld_stmt = top_level_domain_table.select().where(top_level_domain_table.c.domain_name == tld_name)
                        tld_record = session.execute(tld_stmt).fetchone()
                        if tld_record:
                            tld_update_stmt = update(top_level_domain_table).where(
                                top_level_domain_table.c.domain_name == tld_name).values(dt_last_verified=dt_now)
                            session.execute(tld_update_stmt)
                session.commit()
            logger.info(f"added {len(self.known_tld_names) - num_before_load} TLDs from {IanaTLDCacher.data_src_url}")
        except httpx.RequestError as e:
            logger.error("failed to download TLD data from IANA")
