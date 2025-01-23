import ipaddress
import random
import re

cidr_mask_pattern = re.compile('/[0-9]+$')

ipv4_masks = {
    "0.0.0.0",
    "128.0.0.0",
    "192.0.0.0",
    "224.0.0.0",
    "240.0.0.0",
    "248.0.0.0",
    "252.0.0.0",
    "254.0.0.0",
    "255.0.0.0",
    "255.128.0.0",
    "255.192.0.0",
    "255.224.0.0",
    "255.240.0.0",
    "255.248.0.0",
    "255.252.0.0",
    "255.254.0.0",
    "255.255.0.0",
    "255.255.128.0",
    "255.255.192.0",
    "255.255.224.0",
    "255.255.240.0",
    "255.255.248.0",
    "255.255.252.0",
    "255.255.254.0",
    "255.255.255.0",
    "255.255.255.128",
    "255.255.255.192",
    "255.255.255.224",
    "255.255.255.240",
    "255.255.255.248",
    "255.255.255.252",
    "255.255.255.254",
    "255.255.255.255",
}

# TODO: same pattern works for selection of random public blocks
ipv4_private_ip_blocks = [
    # Octet1    Octet2      Octet3      Octet4      Mask
    (10,        0,          0,          0,          8),
    (172,       16,         0,          0,          12),
    (192,       168,        0,          0,          16),
]

ipv4_private_networks: list[ipaddress.IPv4Network] = [
    ipaddress.IPv4Network(f"{block[0]}.{block[1]}.{block[2]}.{block[3]}/{block[4]}")
    for block in ipv4_private_ip_blocks]

ipv4_private_subnet_cache = {}

ipv4_public_ip_blocks = [
    # Octet1    Octet2      Octet3      Octet4      Mask
    (8,         8,          8,          0,          24),    # Google DNS
    (12,        0,          0,          0,          8),     # At&t
    (13,        32,         0,          0,          15),    # AWS
    (13,        64,         0,          0,          11),    # Microsoft Azure
    (17,        0,          0,          0,          8),     # Apple
    (17,        172,        0,          0,          16),    # Apple
    (17,        178,        0,          0,          16),    # Apple
    (31,        13,         24,         0,          21),    # Facebook
    (35,        191,        0,          0,          16),    # Google Cloud Platform
    (40,        0,          0,          0,          8),     # Microsoft
    (40,        76,         0,          0,          14),    # Microsoft
    (51,        144,        0,          0,          12),    # Microsoft
    (52,        0,          0,          0,          8),     # AWS and Microsoft
    (54,        0,          0,          0,          8),     # AWS
    (64,        233,        160,        0,          19),    # Google
    (66,        220,        144,        0,          20),    # Facebook
    (69,        63,         176,        0,          20),    # Facebook
    (69,        171,        224,        0,          20),    # Facebook
    (75,        154,        0,          0,          16),    # Netflix
    (99,        0,          0,          0,          8),     # At&t
    (108,       175,        32,         0,          20),    # Netflix
    (104,       16,         0,          0,          12),    # Cloudflare
    (129,       35,         0,          0,          16),    # IBM
    (137,       84,         0,          0,          16),    # Netflix
    (137,       154,        0,          0,          16),    # Oracle
    (137,       254,        0,          0,          16),    # Oracle
    (157,       240,        0,          0,          16),    # Facebook
    (162,       158,        0,          0,          15),    # Cloudflare
    (169,       45,         0,          0,          16),    # IBM
    (172,       217,        0,          0,          16),    # Google
    (173,       245,        48,         0,          20),    # Cloudflare
    (192,       35,         51,         0,          24),    # IBM
    (192,       70,         0,          0,          16),    # Verizon
    (192,       80,         0,          0,          16),    # Verizon
    (192,       234,        0,          0,          16),    # Salesforce
    (205,       251,        192,        0,          19),    # AWS
    (204,       14,         0,          0,          16),    # Salesforce
    (204,       127,        0,          0,          16),    # At&t
    (216,       232,        0,          0,          16),    # Verizon
]

ipv4_public_networks: list[ipaddress.IPv4Network] = [
    ipaddress.IPv4Network(f"{block[0]}.{block[1]}.{block[2]}.{block[3]}/{block[4]}")
    for block in ipv4_public_ip_blocks]

ipv4_public_subnet_cache = {}

ipv4_special_addresses = {
    "1.0.0.1",    # Cloudflare DNS
    "1.1.1.1",    # Cloudflare DNS
    "1.1.1.2",    # Cloudflare DNS
    "1.1.1.3",    # Cloudflare DNS
    "4.4.4.4",    # Level3 Communications DNS
    "8.8.8.8",    # Google DNS
    "8.8.4.4",    # Google DNS
    "9.9.9.9",    # Quad9 DNS
    "64.6.64.6",  # Verisign DNS
}

# Special IPv4 ranges
ipv4_special_cidr_ranges = [
    "0.0.0.0/8",           # "This" Network
    "10.0.0.0/8",          # Private-Use
    "100.64.0.0/10",       # Shared Address Space (Carrier-Grade NAT)
    "127.0.0.0/8",         # Loopback
    "169.254.0.0/16",      # Link Local
    "172.16.0.0/12",       # Private-Use
    "192.0.0.0/24",        # IETF Protocol Assignments
    "192.0.2.0/24",        # Documentation (TEST-NET-1)
    "192.88.99.0/24",      # 6to4 Relay Anycast
    "192.168.0.0/16",      # Private-Use
    "198.18.0.0/15",       # Network Interconnect Device Benchmark Testing
    "198.51.100.0/24",     # Documentation (TEST-NET-2)
    "203.0.113.0/24",      # Documentation (TEST-NET-3)
    "224.0.0.0/4",         # Multicast
    "240.0.0.0/4",         # Reserved for Future Use
    "255.255.255.255/32",  # Limited Broadcast
]

# Special IPv6 ranges
ipv6_special_cidr_ranges = [
    "::1/128",        # Loopback
    "::/128",         # Unspecified
    "::ffff:0:0/96",  # IPv4-mapped IPv6 addresses
    "100::/64",       # Discard Prefix
    "2001::/32",      # Teredo
    "fc00::/7",       # Unique Local Addresses
    "fe80::/10",      # Link-Local
    "ff00::/8",       # Multicast
]

octet_value_choices = list(range(1, 255))

# Matches for trailing or consecutive 0s.
zero_octet_pattern = re.compile(r"(\.0){1}(?=(\.0|$))")


def augment_ipv4_zeros(addr_blob: str):
    augmented_examples = [addr_blob]
    addr_sans_mask, _ = split_cidr_mask(addr_blob)
    if addr_sans_mask not in ipv4_masks:
        augmented_examples.append(addr_sans_mask)

    if ".0" in addr_blob:
        # Examples with wildcards or placeholders
        for repl in [".*", ".x", ".X"]:
            for pattern in [
                r"(\.0)+(?=/)",        # Replace and truncate
                r"(\.0){1}(?=(/|.0))"  # Replace in-place
            ]:
                repl_addr, _ = re.subn(pattern, repl, addr_blob)
                augmented_examples.append(repl_addr)
                repl_addr_sans_mask, _ = split_cidr_mask(repl_addr)
                augmented_examples.append(repl_addr_sans_mask)

        # Replace trailing or consecutive .0s with new randomly selected octet.
        randomized_addr = addr_sans_mask
        zero_octet_match = zero_octet_pattern.search(addr_sans_mask)
        while zero_octet_match:
            # Replace first occurrence
            randomized_addr = (randomized_addr[:zero_octet_match.start()]
                               + "." + str(random.choice(octet_value_choices))
                               + randomized_addr[zero_octet_match.end():])
            augmented_examples.append(randomized_addr)
            zero_octet_match = zero_octet_pattern.search(randomized_addr)  # In case of more

    return augmented_examples


def generate_corpus(num_examples=60_000):
    corpus = {f"/{i}": [f"/{i}"] for i in range(1, 129)}  # Covers masks for IPv4 and IPv6
    for mask in ipv4_masks:
        corpus[mask] = [mask]

    ##
    # Special

    corpus[ipv4_special_cidr_ranges[:1].pop()] = ipv4_special_cidr_ranges[:1]  # Skip for 0.0.0.0
    for cidr_range in ipv4_special_cidr_ranges[1:-1]:
        for exp in augment_ipv4_zeros(cidr_range):
            if exp not in corpus:
                corpus[exp] = [exp]
    corpus[ipv4_special_cidr_ranges[-1:].pop()] = ipv4_special_cidr_ranges[-1:]  # Skip for 255.255.255.255.

    for cidr_range in ipv6_special_cidr_ranges:
        corpus[cidr_range] = [cidr_range]

    # Augmentation can return several variations of each example, depending on the CIDR range, so `num_examples` is
    # more of a loose hint than a precise target. Expect ~4X `num_examples` since 80% are IPv4 ranges that are
    # augmented.
    for _ in range(num_examples - len(corpus)):
        r = random.random()
        # Split: 80% IPv4 and 20% IPv6
        if r < 0.4:
            # IPv4 private
            new_cidr = generate_random_private_cidr(ipv4_private_ip_blocks,
                                                    ipv4_private_networks, ipv4_private_subnet_cache)
            for exp in augment_ipv4_zeros(new_cidr):
                if exp not in corpus:
                    corpus[exp] = [exp]
        elif r < 0.8:
            # IPv4 public
            new_cidr = generate_random_private_cidr(ipv4_public_ip_blocks,
                                                    ipv4_public_networks, ipv4_public_subnet_cache)
            for exp in augment_ipv4_zeros(new_cidr):
                if exp not in corpus:
                    corpus[exp] = [exp]
        else:
            # IPv6
            pass

    return list(corpus.values())


def generate_random_private_cidr(ip_blocks, networks, cache):

    # Randomly select a private IP block
    block_idx = random.choice(range(len(ip_blocks)))
    block = ip_blocks[block_idx]

    # Generate a list of valid subnet sizes for the selected block
    valid_subnet_sizes = list(range(block[4], 25))  # Down to /24

    # Randomly select a subnet size from the valid options
    subnet_size = random.choice(valid_subnet_sizes)

    # Generate a random subnet within the selected block
    # NOTE: needs to be cached or performance will suffer significantly
    cache_key = f"{block_idx}_{subnet_size}"
    if cache_key not in cache:
        cache[cache_key] = list(networks[block_idx].subnets(new_prefix=subnet_size))
    random_subnet = random.choice(cache[cache_key])

    return str(random_subnet)


def split_cidr_mask(range_blob: str):
    cidr_mask_match = cidr_mask_pattern.search(range_blob)
    if cidr_mask_match:
        return range_blob[:cidr_mask_match.start()], range_blob[cidr_mask_match.start():]
    return range_blob, None


if __name__ == '__main__':
    ipaddr_data = generate_corpus()
    print(ipaddr_data[:300])  # Inspect the first few
