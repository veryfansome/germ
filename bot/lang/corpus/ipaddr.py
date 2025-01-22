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
    (10, 0, 0, 0, 8),  # 10.0.0.0/8
    (172, 16, 0, 0, 12),  # 172.16.0.0/12
    (192, 168, 0, 0, 16)  # 192.168.0.0/16
]

ipv4_private_networks: list[ipaddress.IPv4Network] = [
    ipaddress.IPv4Network(f"{block[0]}.{block[1]}.{block[2]}.{block[3]}/{block[4]}")
    for block in ipv4_private_ip_blocks]

# Special IPv4 Addresses
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

ipv4_subnet_cache = {}

# Special IPv6 Addresses
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


def augment_ipv4_zeros(addr_blob: str):
    augmented_examples = [addr_blob]
    addr_sans_mask, _ = split_cidr_mask(addr_blob)
    if addr_sans_mask not in ipv4_masks:
        augmented_examples.append(addr_sans_mask)

    if ".0" in addr_blob:
        # Examples with wildcards or placeholders
        #for repl in [".*", ".x", ".X"]:
        #    for pattern in [
        #        r"(\.0)+(?=/)",        # Replace with truncate
        #        r"(\.0){1}(?=(/|.0))"  # Replace each
        #    ]:
        #        repl_addr, _ = re.subn(pattern, repl, addr_blob)
        #        augmented_examples.append(repl_addr)
        #        repl_addr_sans_mask, _ = split_cidr_mask(repl_addr)
        #        augmented_examples.append(repl_addr_sans_mask)

        # Examples with placeholders
        # TODO: this doing this and it's not desired.
        # >>> re.sub(r"(\.0){1}(?=(\.0|$))", f".{random.choice(list(range(1, 255)))}", "10.0.0.0")
        # '10.211.211.211'
        randomized_addr = re.sub(r"(\.0){1}(?=(\.0|$))", f".{random.choice(list(range(1, 255)))}", addr_sans_mask)
        augmented_examples.append(randomized_addr)

    return augmented_examples


def generate_corpus(num_examples=100_000):
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

    for _ in range(num_examples - len(corpus)):
        r = random.random()
        if r < 0.4:
            # IPv4 private
            new_cidr = generate_random_private_cidr()
            for exp in augment_ipv4_zeros(new_cidr):
                if exp not in corpus:
                    corpus[exp] = [exp]
        elif r < 0.8:
            # IPv4 public
            pass
        else:
            # IPv6
            pass

    return list(corpus.values())


def generate_random_private_cidr():

    # Randomly select a private IP block
    block_idx = random.choice(range(len(ipv4_private_ip_blocks)))
    block = ipv4_private_ip_blocks[block_idx]

    # Generate a list of valid subnet sizes for the selected block
    if block[4] == 8:
        valid_subnet_sizes = list(range(8, 25))  # /8 to /24
    elif block[4] == 12:
        valid_subnet_sizes = list(range(12, 25))  # /12 to /24
    else:
        valid_subnet_sizes = list(range(16, 25))  # /16 to /24

    # Randomly select a subnet size from the valid options
    subnet_size = random.choice(valid_subnet_sizes)

    # Generate a random subnet within the selected block
    # NOTE: needs to be cached for performance will suffer significantly
    cache_key = f"{block_idx}_{subnet_size}"
    if cache_key not in ipv4_subnet_cache:
        ipv4_subnet_cache[cache_key] = list(ipv4_private_networks[block_idx].subnets(new_prefix=subnet_size))
    random_subnet = random.choice(ipv4_subnet_cache[cache_key])

    return str(random_subnet)


def split_cidr_mask(range_blob: str):
    cidr_mask_match = cidr_mask_pattern.search(range_blob)
    if cidr_mask_match:
        return range_blob[:cidr_mask_match.start()], range_blob[cidr_mask_match.start():]
    return range_blob, None


if __name__ == '__main__':
    ipaddr_data = generate_corpus()
    print(ipaddr_data[:1000])  # Inspect the first few
