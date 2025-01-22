import random
import re

prefixed_symbols = {
    # NOTE: Different maximums with rounded up conversion rates
    #      Minimum  Maximum                   Decimal,    Fraction,   Scientific Notation
    "$": ((0,       1_000_000_000_000),       True,       False,      None),
    "€": ((0,       1_000_000_000_000),       True,       False,      None),
    "£": ((0,       1_000_000_000_000),       True,       False,      None),
    "¥": ((0,       10_000_000_000_000),      True,       False,      None),
    "₽": ((0,       100_000_000_000_000),     True,       False,      None),
    "₹": ((0,       100_000_000_000_000),     True,       False,      None),
    "₩": ((0,       1_000_000_000_000_000),   True,       False,      None),
}
prefixed_symbol_pattern = re.compile("^(" + ("|".join(prefixed_symbols.keys())) + ")")

relative_symbols = [
    "±", "~",
]
relative_symbol_pattern = re.compile("^(" + ("|".join(relative_symbols + ["-"])) + ")")

suffixed_blobs = {
    #             Minimum   Maximum             Decimal,    Fraction,   Scientific notation
    "hundred":  ((1,        999),               True,       False,      None),
    "k":        ((1,        999),               True,       False,      None),  # Thousands
    "thousand": ((1,        999),               True,       False,      None),
    "M":        ((1,        999),               True,       False,      None),  # Millions
    "mil":      ((1,        999),               True,       False,      None),
    "million":  ((1,        999),               True,       False,      None),
    "bil":      ((1,        999),               True,       False,      None),
    "billion":  ((1,        999),               True,       False,      None),
    "trillion": ((1,        999),               True,       False,      None),
    "x":        ((-100,     100),               True,       False,      None),  # Times
    "X":        ((-100,     100),               True,       False,      None),
    "times":    ((-100,     100),               True,       False,      None),
    "%":        ((-1000,    1000),              True,       False,      None),
    "°":        ((-459.67,  10000),             True,       False,      None),  # Temperature or rotation, skimp on -
    # Area
    "µm²":      ((0,        1000),              True,       False,      "-"),
    "µm^2":     ((0,        1000),              True,       False,      "-"),
    "nm²":      ((0,        1000),              True,       False,      None),
    "nm^2":     ((0,        1000),              True,       False,      None),
    "mm²":      ((1,        100),               True,       False,      None),
    "mm^2":     ((1,        100),               True,       False,      None),
    "cm²":      ((1,        100),               True,       False,      None),
    "cm^2":     ((1,        100),               True,       False,      None),
    "m²":       ((1,        1000),              True,       False,      None),
    "m^2":      ((1,        1000),              True,       False,      None),
    "km²":      ((1,        10000),             True,       False,      None),
    "km^2":     ((1,        10000),             True,       False,      None),
    "in²":      ((1,        100),               True,       False,      None),
    "in^2":     ((1,        100),               True,       False,      None),
    "ft²":      ((1,        10000),             True,       False,      None),
    "ft^2":     ((1,        10000),             True,       False,      None),
    "sqft":     ((1,        10000),             True,       False,      None),
    "yd²":      ((1,        10000),             True,       False,      None),
    "yd^2":     ((1,        10000),             True,       False,      None),
    "mi²":      ((1,        10000),             True,       False,      None),
    "mi^2":     ((1,        10000),             True,       False,      None),
    "ha":       ((1,        10000),             True,       False,      None),
    "ac":       ((1,        10000),             True,       False,      None),
    # Bits and Bytes
    "EiB":      ((1,        1024),              True,       False,      None),
    "EB":       ((1,        1024),              True,       False,      None),
    "Eb":       ((1,        1024),              True,       False,      None),
    "PiB":      ((1,        1024),              True,       False,      None),
    "PB":       ((1,        1024),              True,       False,      None),
    "Pb":       ((1,        1024),              True,       False,      None),
    "TiB":      ((1,        1024),              True,       False,      None),
    "TB":       ((1,        1024),              True,       False,      None),
    "Tb":       ((1,        1024),              True,       False,      None),
    "GiB":      ((1,        1024),              True,       False,      None),
    "GB":       ((1,        1024),              True,       False,      None),
    "Gb":       ((1,        1024),              True,       False,      None),
    "MB":       ((1,        1024),              True,       False,      None),
    "Mb":       ((1,        1024),              True,       False,      None),
    "KB":       ((1,        1024),              True,       False,      None),
    "Kb":       ((1,        1024),              True,       False,      None),
    "B":        ((1,        1024),              True,       False,      None),  # Also covers billions
    "b":        ((1,        1024),              True,       False,      None),
    # Bit and Byte rate
    "EB/s":     ((1,        1024),              True,       False,      None),
    "Eb/s":     ((1,        1024),              True,       False,      None),
    "Ebps":     ((1,        1024),              True,       False,      None),
    "PB/s":     ((1,        1024),              True,       False,      None),
    "Pb/s":     ((1,        1024),              True,       False,      None),
    "Pbps":     ((1,        1024),              True,       False,      None),
    "TB/s":     ((1,        1024),              True,       False,      None),
    "Tb/s":     ((1,        1024),              True,       False,      None),
    "Tbps":     ((1,        1024),              True,       False,      None),
    "GB/s":     ((1,        1024),              True,       False,      None),
    "Gb/s":     ((1,        1024),              True,       False,      None),
    "Gbps":     ((1,        1024),              True,       False,      None),
    "MB/s":     ((1,        1024),              True,       False,      None),
    "Mb/s":     ((1,        1024),              True,       False,      None),
    "Mbps":     ((1,        1024),              True,       False,      None),
    "KB/s":     ((1,        1024),              True,       False,      None),
    "Kb/s":     ((1,        1024),              True,       False,      None),
    "B/s":      ((1,        1024),              True,       False,      None),
    "b/s":      ((1,        1024),              True,       False,      None),
    # Distance
    "µm":       ((0,        1000),              True,       False,      "-"),
    "nm":       ((0,        1000),              True,       False,      None),
    "mm":       ((1,        100),               True,       False,      None),
    "cm":       ((0,        100),               True,       True,       None),
    "m":        ((0,        1000),              True,       True,       None),  # May be `minute` in some cases.
    "km":       ((0,        10000),             True,       True,       None),
    "″":        ((0,        100),               True,       True,       None),
    "in":       ((0,        100),               True,       True,       None),
    "′":        ((0,        10000),             True,       True,       None),
    "ft":       ((0,        10000),             True,       True,       None),
    "yd":       ((0,        10000),             True,       True,       None),
    "mi":       ((0,        10000),             True,       True,       None),
    "ly":       ((0,        100_000_000_000),   True,       False,      "+"),  # Light Year
    "AU":       ((0,        100000),            True,       False,      None),
    # Energy
    "µJ":       ((1,        1000),              True,       False,      "-"),
    "nJ":       ((1,        1000),              True,       False,      None),
    "mJ":       ((1,        1000),              True,       False,      None),
    "J":        ((1,        1000),              True,       False,      None),
    "kJ":       ((1,        1000),              True,       False,      None),
    "MJ":       ((1,        1000),              True,       False,      None),
    "GJ":       ((1,        1000),              True,       False,      "+"),
    "TJ":       ((1,        1000),              True,       False,      "+"),
    "PJ":       ((1,        1000),              True,       False,      "+"),
    "EJ":       ((1,        1000),              True,       False,      "+"),
    "kcal":     ((1,        10000),             True,       False,      None),  # Calorie or kilocalorie
    "Wh":       ((1,        1000),              True,       False,      None),
    "kWh":      ((2,        1000),              True,       False,      None),
    "MWh":      ((1,        1000),              True,       False,      None),
    "GWh":      ((1,        1000),              True,       False,      None),
    "TWh":      ((1,        1000),              True,       False,      None),
    "PWh":      ((1,        1000),              True,       False,      None),
    "EWh":      ((1,        1000),              True,       False,      None),
    # Force
    "N":        ((0,        10_000_000),        True,       False,      None),
    # Mass
    "mg":       ((1,        1000),              True,       True,       None),
    "g":        ((1,        1000),              True,       True,       None),
    "kg":       ((1,        1000),              True,       True,       None),
    "lb":       ((1,        10000),             True,       True,       None),
    "oz":       ((1,        64),                True,       True,       None),
    # Power
    "W":        ((1,        1000),              True,       False,      None),
    "kW":       ((1,        1000),              True,       False,      None),
    "MW":       ((1,        1000),              True,       False,      None),
    "GW":       ((1,        1000),              True,       False,      None),
    "TW":       ((1,        1000),              True,       False,      None),
    "PW":       ((1,        1000),              True,       False,      None),
    "EW":       ((1,        1000),              True,       False,      None),
    "hp":       ((1,        1_000_000),         True,       False,      None),  # Horsepower
    "HP":       ((1,        1_000_000),         True,       False,      None),  # Horsepower
    # Pressure
    "Pa":       ((1,        1000),              True,       False,      None),
    "kPa":      ((1,        1000),              True,       False,      None),
    "bar":      ((1,        100),               True,       False,      None),
    "MPa":      ((1,        1000),              True,       False,      None),
    "GPa":      ((1,        1000),              True,       False,      None),
    "atm":      ((0,        100_000),           True,       False,      None),
    # Speed
    "m/s":      ((0,        1000),              True,       False,      None),
    "km/h":     ((0,        10000),             True,       False,      None),
    "kph":      ((0,        10000),             True,       False,      None),
    "kmh":      ((0,        10000),             True,       False,      None),
    "mph":      ((0,        10000),             True,       False,      None),
    # Temperature
    "°C":       ((-273.15,  1000),              True,       False,      None),
    "°F":       ((-459.67,  1000),              True,       False,      None),
    "K":        ((0,        10000),             True,       False,      None),  # Also covers thousands
    # Time
    "µs":       ((0,        1000),              True,       False,      "-"),
    "ns":       ((1,        1000),              True,       False,      None),
    "ms":       ((1,        1000),              True,       False,      None),
    "s":        ((1,        1000),              True,       False,      None),
    "sec":      ((1,        1000),              True,       False,      None),
    "min":      ((1,        1000),              True,       True,       None),
    "h":        ((1,        1000),              True,       True,       None),
    "hr":       ((1,        1000),              True,       True,       None),
    "d":        ((1,        1000),              True,       True,       None),
    "D":        ((1,        1000),              True,       True,       None),
    "yr":       ((1,        10000),             True,       True,       None),
    "Y":        ((1,        10000),             True,       True,       None),
    "Yr":       ((1,        10000),             True,       True,       None),
    # Volume
    "µm³":      ((0,        1000),              True,       False,      "-"),
    "µm^3":     ((0,        1000),              True,       False,      "-"),
    "nm³":      ((1,        1000),              True,       False,      None),
    "nm^3":     ((1,        1000),              True,       False,      None),
    "mm³":      ((1,        100),               True,       False,      None),
    "mm^3":     ((1,        100),               True,       False,      None),
    "cc":       ((1,        100),               True,       False,      None),
    "cm³":      ((1,        100),               True,       False,      None),
    "cm^3":     ((1,        100),               True,       False,      None),
    "m³":       ((1,        1000),              True,       False,      None),
    "m^3":      ((1,        1000),              True,       False,      None),
    "µL":       ((0,        1000),              True,       False,      "-"),
    "nL":       ((1,        1000),              True,       False,      None),
    "mL":       ((1,        1000),              True,       True,       None),
    "L":        ((1,        1000),              True,       True,       None),
    "hL":       ((1,        1000),              True,       False,      None),  # Hectoliter
    "tsp":      ((1,        30),                True,       True,       None),  # Teaspoon
    "tbsp":     ((1,        15),                True,       True,       None),  # Tablespoon
    "c":        ((1,        12),                True,       True,       None),  # cups, can be calorie but rare
    "fl oz":    ((1,        64),                True,       True,       None),
    "pt":       ((1,        8),                 True,       True,       None),  # Pint
    "qt":       ((1,        8),                 True,       True,       None),  # Quart
    "gal":      ((1,        1000),              True,       True,       None),
}
suffixed_symbol_pattern = re.compile("(" + ("|".join([re.escape(symbol) for symbol in suffixed_blobs.keys()])) + ")$")


def generate_corpus(num_examples=236_736):  # Same as number of english words in nltk word corpus
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

    return [tokenize_example_text(exp) for exp in corpus]


def random_decimal(min_val=0, max_val=9999):
    """
    Returns a random decimal string, up to 3 decimal places.
    E.g. '3.14', '999.999', '10.0'.
    """
    whole = random.randint(int(min_val), max_val)
    fractional_places = random.randint(0, 3)
    if fractional_places == 0:
        return str(whole)
    fraction_part = random.randint(0, 10**fractional_places - 1)
    fraction_str = str(fraction_part).zfill(fractional_places)  # pad with zeros if needed
    return f"{whole}.{fraction_str}"


def random_fraction(min_whole_val=0, max_whole_val=20):
    """
    Returns a fraction string like '1/2', '3/4', or '10 1/2'.
    We'll keep numerators/denominators up to 12 for variety.
    Sometimes produce a mixed fraction (like '10 1/2'), sometimes just a simple fraction.
    """
    whole_part = random.randint(min_whole_val, max_whole_val)
    numerator = random.randint(1, 12)
    denominator = random.randint(2, 12) # no zero or 1 for denominator
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
    r = random.random()
    if r < 0.50:
        # random unit
        base_str = random_numeric_example_with_unit()
    else:
        # Weighted random choice among numeric formats
        # e.g. int: 40%, decimal: 30%, fraction: 15%, scientific: 15%
        # Should be same as random_number_for_unit() below.
        r = random.random()
        if r < 0.40:
            base_str = random_integer(min_val=0, max_val=999999)
        elif r < 0.70:
            base_str = random_decimal(min_val=0, max_val=9999)
        elif r < 0.85:
            base_str = random_fraction(min_whole_val=0, max_whole_val=20)
        else:
            base_str = random_scientific_notation(min_val=0, max_val=999)

        r = random.random()
        # ~50% chance to add a negative symbol
        if r < 0.50:
            base_str = "-" + base_str

    r = random.random()
    # ~11% chance to add a relative symbol
    if r < 0.11:
        base_str = random.choice(relative_symbols) + base_str

    return base_str


def random_numeric_example_with_unit():
    """
    Sometimes prepend or append a unit or symbol (like '$10', '10°', etc.).
    """
    threshold_rate = len(prefixed_symbols) / len(suffixed_blobs)
    r = random.random()
    if r < threshold_rate:
        prefix = random.choice(list(prefixed_symbols.keys()))
        num_str = random_number_for_unit(prefixed_symbols, prefix)
        return prefix + num_str
    else:
        suffix = random.choice(list(suffixed_blobs.keys()))
        num_str = random_number_for_unit(suffixed_blobs, suffix)
        return num_str + suffix


def random_number_for_unit(unit_range_map, unit):
    """
    Return a random numeric string plausible for the given unit.
    If not in UNIT_RANGES, fallback to (0, 9999).
    We'll just do an integer for brevity; you can also do decimals or e-not.
    """
    likely_range, decimals_possible, fractions_possible, sci_notation_possible = unit_range_map.get(
        unit, ((0, 9999), False, False, False))
    min_val, max_val = likely_range
    # Same ratios and random_numeric_example().
    r = random.random()
    if r < 0.40:
        pass  # Return integer
    elif decimals_possible and r < 0.70:
        return random_decimal(min_val=min_val, max_val=max_val)
    elif fractions_possible and r < 0.85:
        return random_fraction(min_whole_val=min_val, max_whole_val=max_val)
    elif sci_notation_possible is not None:
        return random_scientific_notation(min_val=min_val, max_val=max_val,
                                          positive_exponent=True if sci_notation_possible == "+" else False,
                                          negative_exponent=True if sci_notation_possible == "-" else False)
    return random_integer(min_val=int(min_val), max_val=int(max_val))


def random_scientific_notation(min_val=0, max_val=999, max_exponent=10, positive_exponent=True, negative_exponent=True):
    """
    Returns a random number in scientific notation, e.g., '3.14e+5' or '12e-3'.

    Args:
      min_val (int): minimum integer for the base's whole part
      max_val (int): maximum integer for the base's whole part
      max_exponent (int): max absolute value for the exponent
      positive_exponent (bool): enables positive exponent
      negative_exponent (bool): enables negative exponent

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
    exponent = random.randint(-max_exponent if negative_exponent else 0, max_exponent if positive_exponent else 0)
    # skip exponent == 0 occasionally to ensure variety?
    # (you can, but it's optional; 0 just means 1.23e+0, which is effectively 1.23)

    # 3) Format the exponent sign
    # Typically: e+3 or e-3
    if exponent >= 0:
        sci_str = f"{base_str}e+{exponent}"
    else:
        sci_str = f"{base_str}e{exponent}"  # automatically includes the '-' sign

    return sci_str


def tokenize_example_text(example: str):
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


if __name__ == '__main__':
    numeric_data = generate_corpus(10_000)  # e.g., 10k for testing
    print(numeric_data[:100])  # Inspect the first few
