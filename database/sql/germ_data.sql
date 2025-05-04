INSERT INTO struct_type (group_name, att_pub_ident, att_value)
VALUES
      ('conversation_state', '1', 'Began')
    , ('conversation_state', '2', 'Paused - Idled')  /* on no new user activity */
    , ('conversation_state', '3', 'Paused - Timed Out')  /* on disconnect due to ws timeout */
    , ('conversation_state', '4', 'Resumed')
    , ('conversation_state', '0', 'Ended')
ON CONFLICT (group_name, att_pub_ident)
DO NOTHING;
