import re

ipv4_addr_pattern = re.compile(r"^(?P<ipv4_addr>(\d{1,3}\.){3}\d{1,3})")
ipv6_addr_pattern = re.compile(r"^(?P<ipv6_addr>\[[0-9a-zA-Z:%]{3,39}])")
