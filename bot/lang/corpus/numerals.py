import random
import re

prefixed_symbols = [
    "$", "€", "£", "¥", "₽", "₹", "₩"
]
prefixed_symbol_pattern = re.compile("^(" + ("|".join(prefixed_symbols)) + ")")

relative_symbols = [
    "±", "~",
]
relative_symbol_pattern = re.compile("^(" + ("|".join(relative_symbols + ["-"])) + ")")

suffixed_symbols = [
    "%", "°",
    # Area
    "mm²", "mm^2", "cm²", "cm^2", "m²", "m^2", "km²", "km^2", "in²", "in^2", "ft²", "ft^2", "sqft", "sq. ft.", "mi²",
    "ha", "ac",
    # Bits and Bytes
    "TiB", "TB", "Tb", "GiB", "GB", "Gb", "MB", "Mb", "KB", "Kb", "B", "b",
    # Bit and Byte rate
    "GB/s", "Gb/s", "Gbps", "MB/s", "Mb/s", "Mbps", "KB/s", "Kb/s", "B/s", "b/s",
    # Distance
    "mm", "cm", "m", "km", "in", "ft", "yd", "mi"
    # Energy
    "J", "kJ", "cal", "Wh",
    # Force
    "N",
    # Mass
    "mg", "g", "kg", "lb", "oz",
    # Power
    "W", "kW", "hp",
    # Pressure
    "Pa", "kPa", "atm", "bar",
    # Speed
    "m/s", "km/h", "kph", "kmh", "mph",
    # Temperature
    "°C", "°F", "K",
    # Time
    "yr", "d", "h", "hr", "m", "min", "s", "sec", "ms",
    # Volume
    "mL", "L", "gal", "qt", "fl oz",
]
suffixed_symbol_pattern = re.compile("(" + ("|".join([re.escape(symbol) for symbol in suffixed_symbols])) + ")$")


def tokenize_example(example: str):
    tokens = []
    relative_symbol_match = relative_symbol_pattern.search(example)
    if relative_symbol_match:
        tokens.append(example[:relative_symbol_match.end()])
        example = example[relative_symbol_match.end():]
    prefixed_symbol_match = prefixed_symbol_pattern.search(example)
    if prefixed_symbol_match:
        tokens.append(example[:prefixed_symbol_match.end()])
        example = example[prefixed_symbol_match.end():]
    suffixed_symbol_match = suffixed_symbol_pattern.search(example)
    if suffixed_symbol_match:
        tokens.append(example[:suffixed_symbol_match.start()])
        example = example[suffixed_symbol_match.start():]
    if example:
        tokens.append(example)
    return tokens


def generate_numeric_corpus(num_examples=236_736):  # Same as number of english words in nltk word corpus
    """
    Generates a list of num_examples random numeric strings representative
    of typical digit-based data (int, decimal, fraction) with optional attached symbols/units.

    Returns:
      A list of strings (each is a single "example").
    """
    corpus = set()
    for n in range(101):  # Add all first 100
        corpus.add(str(n))
    for _ in range(num_examples - 101):
        corpus.add(random_numeric_example())

    return [tokenize_example(exp) for exp in corpus]


def random_decimal(min_val=0, max_val=9999):
    """
    Returns a random decimal string, up to 3 decimal places.
    E.g. '3.14', '999.999', '10.0'.
    """
    whole = random.randint(min_val, max_val)
    fractional_places = random.randint(0, 3)
    if fractional_places == 0:
        return str(whole)
    fraction_part = random.randint(0, 10**fractional_places - 1)
    fraction_str = str(fraction_part).zfill(fractional_places)  # pad with zeros if needed
    return f"{whole}.{fraction_str}"


def random_fraction():
    """
    Returns a fraction string like '1/2', '3/4', or '10 1/2'.
    We'll keep numerators/denominators up to 12 for variety.
    Sometimes produce a mixed fraction (like '10 1/2'), sometimes just a simple fraction.
    """
    whole_part = random.randint(0, 20)      # smallish whole part
    numerator = random.randint(1, 12)
    denominator = random.randint(2, 12)     # no zero or 1 for denominator
    if whole_part == 0:
        # just fraction
        return f"{numerator}/{denominator}"
    else:
        # e.g., "10 1/2"
        return f"{whole_part} {numerator}/{denominator}"


def random_integer(min_val=0, max_val=999999, use_commas_prob=0.3):
    """
    Returns a string of a random integer within [min_val, max_val].
    Sometimes includes commas based on 'use_commas_prob'.
    """
    num = random.randint(min_val, max_val)
    plain_str = str(num)

    # Decide whether to add commas
    if random.random() < use_commas_prob and num >= 1000:
        # Use Python built-in formatting for thousands separators
        return f"{num:,}"
    else:
        return plain_str


def random_numeric_example():
    """
    Returns a single random numeric 'token/string' that might appear in text, focusing on:
      - integer (0..999999)
      - decimal
      - fraction
      - scientific
      with optional attached symbol or unit.
    """
    # Weighted random choice among numeric formats
    # e.g. int: 40%, decimal: 30%, fraction: 15%, scientific: 15%
    r = random.random()
    if r < 0.40:
        base_str = random_integer(0, 999999)
    elif r < 0.70:
        base_str = random_decimal(0, 9999)
    elif r < 0.85:
        base_str = random_fraction()
    else:
        base_str = random_scientific_notation(0, 999, 10)

    # ~11% chance to add a negative symbol
    if r < 0.11:
        base_str = "-" + base_str
    else:
        # ~33% chance to add a unit or symbol
        r = random.random()
        if r < 0.33:
            base_str = random_unit_or_symbol(base_str)
        # ~11% chance to add a relative symbol
        if r < 0.11:
            base_str = random.choice(relative_symbols) + base_str

    return base_str


def random_scientific_notation(min_val=0, max_val=999, max_exponent=10):
    """
    Returns a random number in scientific notation, e.g., '3.14e+5' or '12e-3'.

    Args:
      min_val (int): minimum integer for the base's whole part
      max_val (int): maximum integer for the base's whole part
      max_exponent (int): max absolute value for the exponent

    The base is generated by a small decimal function,
    and the exponent is an integer in [-max_exponent, max_exponent].
    """
    # 1) Generate a "base" as a decimal up to 3 fractional digits
    whole = random.randint(min_val, max_val)
    fractional_places = random.randint(0, 3)
    if fractional_places == 0:
        base_str = str(whole)
    else:
        fraction_part = random.randint(0, 10 ** fractional_places - 1)
        fraction_str = str(fraction_part).zfill(fractional_places)  # pad with zeros
        base_str = f"{whole}.{fraction_str}"

    # 2) Random exponent
    exponent = random.randint(-max_exponent, max_exponent)
    # skip exponent == 0 occasionally to ensure variety?
    # (you can, but it's optional; 0 just means 1.23e+0, which is effectively 1.23)

    # 3) Format the exponent sign
    # Typically: e+3 or e-3
    if exponent >= 0:
        sci_str = f"{base_str}e+{exponent}"
    else:
        sci_str = f"{base_str}e{exponent}"  # automatically includes the '-' sign

    return sci_str


def random_unit_or_symbol(number_str):
    """
    Sometimes prepend or append a unit or symbol (like '$10', '10°', etc.).
    """
    threshold_rate = len(prefixed_symbols) / len(suffixed_symbols)
    r = random.random()
    if r < threshold_rate:
        symbol = random.choice(prefixed_symbols)
        return symbol + number_str
    else:
        symbol = random.choice(suffixed_symbols)
        return number_str + symbol


if __name__ == '__main__':
    numeric_data = generate_numeric_corpus(10_000)  # e.g., 10k for testing
    print(numeric_data[:100])  # Inspect the first 50
