import re

ipv4_addr_pattern = re.compile(r"^(?P<ipv4_addr>(\d{1,3}\.){3}\d{1,3})")
ipv6_addr_pattern = re.compile(r"^(?P<ipv6_addr>\[[0-9a-zA-Z:%]{3,39}])")

non_terminal_periods = (
    r"(?<!\sApt)"
    r"(?<!\sBlvd)"
    r"(?<!\sDr)"
    r"(?<!\sJr)"
    r"(?<!\sMr)"
    r"(?<!\sMrs)"
    r"(?<!\sMs)"
    r"(?<!\sPh\.D)"
    r"(?<!\sRd)"
    r"(?<!\sSr)"
    r"(?<!\se\.g)"
    r"(?<!\setc)"
    r"(?<!\si\.e)"
    r"(?<!\s[A-Z])"
    r"(?<!^[a-zA-Z0-9])"
)

naive_tokenize_pattern = re.compile(
    r"("
    r"\s+"
    r"|-+(?=\s|$)"
    r"|(?<=\s)-+"
    r"|-{2,}"
    r"|—+"
    r"|(?<=[a-z])n’t(?=\s|$)"
    r"|(?<=[a-z])n't(?=\s|$)"
    r"|’[a-s,u-z]+(?=\s|$)"
    r"|'[a-s,u-z]+(?=\s|$)"
    r"|’+"
    r"|'+"
    r"|\"+"
    r"|`+"
    r"|,+(?=\"|\s|$)"
    r"|" + non_terminal_periods + r"\.+(?=\"|\s|$)"
    r"|:+"
    r"|;+"
    r"|[?!]+(?=\"|\s|$)"
    r"|\(+"
    r"|\)+"
    r"|\[+"
    r"|]+"
    r"|\{+"
    r"|}+"
    r"|<+"
    r"|>+"
    r")"
)

naive_sentence_end_pattern = re.compile(r"([\n\r]+"
                                        r"|[!?]+\"?(?=\s|$)"
                                        r"|" + non_terminal_periods + r"\.+\"?(?=\s|$))")
# Option 1:
#   [\n\r]+    - Match consecutive newline and carriage returns
# Option 2:
#   [!?]+      - Match ! or ?
#   (?=\s|$)   - Must be followed by \s or end-of-string
# Option 3:
#   non_terminal_periods  - Must not be preceded by non-terminal characters
#   \.+                   - Match .
#   (?=\s|$)              - Must be followed by \s or end-of-string
