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

DROP TABLE IF EXISTS chat_user CASCADE;
CREATE TABLE chat_user (
      user_id                                                   SMALLINT                            NOT NULL GENERATED ALWAYS AS IDENTITY
    , dt_created                                                TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified                                               TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , user_name                                                 TEXT                                NOT NULL
    , PRIMARY KEY (user_id)
)
;
CREATE UNIQUE INDEX idx_chat_user_user_name                     ON chat_user                        USING btree (user_name);
SELECT update_dt_modified_column('chat_user');
COMMENT ON TABLE chat_user IS 'This table stores chat user records';


DROP TABLE IF EXISTS user_session CASCADE;
CREATE TABLE user_session (
      session_id                                                SMALLINT                            NOT NULL GENERATED ALWAYS AS IDENTITY
    , dt_created                                                TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified                                               TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , user_id                                                   SMALLINT                            NOT NULL
    , PRIMARY KEY (session_id)
    , CONSTRAINT fk_user_session_chat_user_user_id              FOREIGN KEY (user_id)               REFERENCES chat_user  (user_id)
)
;
SELECT update_dt_modified_column('user_session');
COMMENT ON TABLE user_session IS 'This table stores user session records';


DROP TABLE IF EXISTS chat_message CASCADE;
CREATE TABLE chat_message (
      message_id                                                SMALLINT                            NOT NULL GENERATED ALWAYS AS IDENTITY
    , dt_created                                                TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified                                               TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , from_user                                                 BOOLEAN                             NOT NULL
    , json_sig                                                  UUID                                NOT NULL
    , session_id                                                SMALLINT                            NOT NULL
    , PRIMARY KEY (session_id)
    , CONSTRAINT fk_chat_message_user_session_session_id        FOREIGN KEY (session_id)            REFERENCES user_session  (session_id)
)
;
SELECT update_dt_modified_column('chat_message');
COMMENT ON TABLE chat_message IS 'This table stores chat message records';


DROP TABLE IF EXISTS struct_type CASCADE;
CREATE TABLE struct_type (
      struct_type_id                                            SMALLINT                            NOT NULL GENERATED ALWAYS AS IDENTITY
    , dt_created                                                TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified                                               TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , att_pub_ident                                             TEXT                                NOT NULL   /* exp: 3XL, 0000FF - identifier */
    , att_value                                                 TEXT                                NOT NULL   /* exp: 3X Large, Blue - display name*/
    , display_order                                             SMALLINT                            NOT NULL DEFAULT 1000   /* Set if needed */
    , group_name                                                TEXT                                NOT NULL   /* exp: t-shirt size, color - category name*/
    , PRIMARY KEY (struct_type_id)
)
;
CREATE UNIQUE INDEX idx_struct_type_att_pub_ident               ON struct_type                      USING btree (group_name, att_pub_ident);
SELECT update_dt_modified_column('struct_type');
COMMENT ON TABLE struct_type IS 'This table functions as a single type table to avoid having many smaller type tables';


DROP TABLE IF EXISTS top_level_domain CASCADE;
CREATE TABLE top_level_domain (
      top_level_domain_id                                       SMALLINT                            NOT NULL GENERATED ALWAYS AS IDENTITY
    , dt_created                                                TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_last_verified                                          TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , dt_modified                                               TIMESTAMPTZ                         NOT NULL DEFAULT CURRENT_TIMESTAMP
    , domain_name                                               TEXT                                NOT NULL
    , PRIMARY KEY (top_level_domain_id)
)
;
CREATE UNIQUE INDEX idx_top_level_domain_name                   ON top_level_domain                 USING btree (domain_name);
SELECT update_dt_modified_column('top_level_domain');
