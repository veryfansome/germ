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


DROP TABLE IF EXISTS text_block CASCADE;
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


DROP TABLE IF EXISTS text_token CASCADE;
CREATE TABLE text_token (
      text_token_id                  						    SMALLINT                			NOT NULL GENERATED ALWAYS AS IDENTITY
    , dt_created                      							TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified                     							TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
    , token_text											    TEXT								NOT NULL
    , PRIMARY KEY (text_token_id)
)
;
CREATE UNIQUE INDEX idx_text_token_token_text                   ON text_token                       USING btree (token_text);
SELECT update_dt_modified_column('text_token');


DROP TABLE IF EXISTS text_token_label CASCADE;
CREATE TABLE text_token_label (
      text_block_id                    					        SMALLINT                			NOT NULL
    , text_token_id                  						    SMALLINT                			NOT NULL
    , token_position                    				        SMALLINT                			NOT NULL
    , dt_created                      							TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified                     							TIMESTAMPTZ             			NOT NULL DEFAULT CURRENT_TIMESTAMP
    , adjgrad_type_id                    				        SMALLINT                			NOT NULL
    , adjpos_type_id                    				        SMALLINT                			NOT NULL
    , adjtype_type_id                    				        SMALLINT                			NOT NULL
    , advtype_type_id                    				        SMALLINT                			NOT NULL
    , case_type_id                    				            SMALLINT                			NOT NULL
    , definite_type_id                    				        SMALLINT                			NOT NULL
    , degree_type_id                    				        SMALLINT                			NOT NULL
    , gender_type_id                    				        SMALLINT                			NOT NULL
    , mood_type_id                    				            SMALLINT                			NOT NULL
    , number_type_id                    				        SMALLINT                			NOT NULL
    , numtype_type_id                    				        SMALLINT                			NOT NULL
    , person_type_id                    				        SMALLINT                			NOT NULL
    , pos_type_id                    				            SMALLINT                			NOT NULL
    , prontype_type_id                    				        SMALLINT                			NOT NULL
    , tense_type_id                    				            SMALLINT                			NOT NULL
    , verbform_type_id                    				        SMALLINT                			NOT NULL
    , xpos_type_id                    				            SMALLINT                			NOT NULL
    , PRIMARY KEY (text_token_id, text_block_id, token_position)
    , CONSTRAINT fk_text_token_label_struct_type_adjgrad    FOREIGN KEY (adjgrad_type_id)   REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_adjpos     FOREIGN KEY (adjpos_type_id)    REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_adjtype    FOREIGN KEY (adjtype_type_id)   REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_advtype    FOREIGN KEY (advtype_type_id)   REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_case       FOREIGN KEY (case_type_id)      REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_definite   FOREIGN KEY (definite_type_id)  REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_degree     FOREIGN KEY (degree_type_id)    REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_gender     FOREIGN KEY (gender_type_id)    REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_mood       FOREIGN KEY (mood_type_id)      REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_number     FOREIGN KEY (number_type_id)    REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_numtype    FOREIGN KEY (numtype_type_id)   REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_person     FOREIGN KEY (person_type_id)    REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_pos        FOREIGN KEY (pos_type_id)       REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_prontype   FOREIGN KEY (prontype_type_id)  REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_tense      FOREIGN KEY (tense_type_id)     REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_verbform   FOREIGN KEY (verbform_type_id)  REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_struct_type_xpos       FOREIGN KEY (xpos_type_id)      REFERENCES struct_type  (struct_type_id)
    , CONSTRAINT fk_text_token_label_text_block             FOREIGN KEY (text_block_id)     REFERENCES text_block   (text_block_id)
    , CONSTRAINT fk_text_token_label_text_token             FOREIGN KEY (text_token_id)     REFERENCES text_token   (text_token_id)
)
;
SELECT update_dt_modified_column('text_token_label');


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
