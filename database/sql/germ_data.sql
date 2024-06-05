INSERT INTO struct_type (group_name, att_pub_ident, att_value)
VALUES
      ('text_block_type', 'code', 'Code')
    , ('text_block_type', 'paragraph', 'Paragraph')
ON CONFLICT (group_name, att_pub_ident)
DO NOTHING;
