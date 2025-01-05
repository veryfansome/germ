
CREATE CONSTRAINT     FOR (entity:Entity)       REQUIRE entity.text              IS UNIQUE
CREATE INDEX          FOR (entity:Entity)       ON entity.plural_forms