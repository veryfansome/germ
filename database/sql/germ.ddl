SET client_min_messages TO WARNING;
-- SET ROLE=:XXXX_admin;
SET search_path TO public;

/**
 *
 *  Extensions, Functions, and Storted Procedures
 *
 **/
--
-- Function to automate updated of `dt_modified` columns so that the appilcation is not responsible for managing that value.
--
CREATE OR REPLACE FUNCTION set_dt_modified_column()
RETURNS TRIGGER AS
$$
	BEGIN
		NEW.dt_modified = NOW();
	RETURN NEW;
END;
$$
LANGUAGE 'plpgsql';

CREATE OR REPLACE FUNCTION update_dt_modified_column(tablename REGCLASS)
RETURNS VOID AS
$$
	BEGIN
		EXECUTE FORMAT('CREATE TRIGGER set_dt_modified_column BEFORE UPDATE ON %s FOR EACH ROW WHEN (OLD IS DISTINCT FROM NEW) EXECUTE FUNCTION set_dt_modified_column();', CONCAT('"', tablename, '"'));
	END;
$$
LANGUAGE 'plpgsql';

/**
 *
 *  Table Definitions
 *
 **/

/* Single type table to avoid many smaller type tables */
DROP TABLE IF EXISTS struct_type CASCADE;
CREATE TABLE struct_type (
      struct_type_id                    						SMALLINT                			NOT NULL GENERATED ALWAYS AS IDENTITY
    , dt_created                      							TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified                     							TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
    , display_order                     						SMALLINT                			NOT NULL DEFAULT 1000   /* Set if needed */
    , group_name                        						TEXT                    			NOT NULL   /* exp: t-shirt size, color - category name*/
    , att_pub_ident                     						TEXT                    			NOT NULL   /* exp: 3XL, 0000FF - identifier */
    , att_value                         						TEXT                    			NOT NULL   /* exp: 3X Large, Blue - display name*/
    , PRIMARY KEY (struct_type_id)
)
;
CREATE UNIQUE INDEX idx_struct_type_att_pub_ident               ON struct_type                      USING btree (group_name, att_pub_ident);
SELECT update_dt_modified_column('struct_type');


CREATE TABLE text_block (
      text_block_id                  						    SMALLINT                			NOT NULL GENERATED ALWAYS AS IDENTITY
    , dt_created                      							TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified                     							TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
	, signature_uuid											UUID								NOT NULL
    , text_block_type_id                    					SMALLINT                			NOT NULL
    , PRIMARY KEY (text_block_id)
    , CONSTRAINT fk_text_block_struct_type FOREIGN KEY (text_block_type_id) REFERENCES struct_type (struct_type_id)
)
;
CREATE UNIQUE INDEX idx_text_block_signature_uuid               ON text_block                       USING btree (signature_uuid);
SELECT update_dt_modified_column('text_block');


DROP TABLE IF EXISTS top_level_domain CASCADE;
CREATE TABLE top_level_domain (
      top_level_domain_id                  						SMALLINT                			NOT NULL GENERATED ALWAYS AS IDENTITY
    , dt_created                      							TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_last_verified                     						TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified                     							TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
	, name														TEXT								NOT NULL
    , PRIMARY KEY (top_level_domain_id)
)
;
CREATE UNIQUE INDEX idx_top_level_domain_name               	ON top_level_domain                 USING btree (name);
SELECT update_dt_modified_column('top_level_domain');
