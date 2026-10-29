"""Isotope-aware wrapper around chemparse.

chemparse only understands element symbols, so 'LiF' -> {'Li': 1, 'F': 1}
with no idea which Li isotope is meant. Two things are needed:

1. A way for the user to SAY which isotope they mean, in the formula string.
2. A rule for what to assume when they don't (natural abundance).

Isotope notation accepted in the formula (all give Li-6 here):
    [6Li]F     ^6LiF     Li-6F     (a bare "6Li" is NOT accepted: "H2O" would
                                    read as isotope 2 of O)
"""
import re
from chemparse import parse_formula

# natural isotopic abundances (atom fractions, IUPAC/NIST), only for elements
# in the muon-capture table that have isotope-specific entries. Extend as needed
# (or pull from the `periodictable` / `mendeleev` packages).
ABUNDANCE = {
    "Li": {6: 0.0759, 7: 0.9241},
    "B": {10: 0.199, 11: 0.801},
    "C": {12: 0.9893, 13: 0.0107},
    "O": {16: 0.99757, 17: 0.00038, 18: 0.00205},
    "Cl": {35: 0.7576, 37: 0.2424},
    "Cr": {50: 0.04345, 52: 0.83789, 53: 0.09501, 54: 0.02365},
    "Ni": {58: 0.680769, 60: 0.262231, 61: 0.011399, 62: 0.036345, 64: 0.009256},
    "Cu": {63: 0.6915, 65: 0.3085},
    "Br": {79: 0.5069, 81: 0.4931},
}

_ISO = re.compile(
    r"\[(\d+)([A-Z][a-z]?)\]"  # [6Li]
    r"|\^(\d+)([A-Z][a-z]?)"  # ^6Li
    r"|([A-Z][a-z]?)-(\d+)"  # Li-6
)
_PLACEHOLDERS = [f"X{c}" for c in "abcdefghijklmnopqrstuvwxyz"]


def parse_formula_isotopes(formula, split_natural=True):
    """Return {'Li-6': n, 'F': n, ...} of atom counts.

    Isotopes written in the formula keep their label. With split_natural=True,
    plain elements that appear in ABUNDANCE are split into isotopes by natural
    abundance (e.g. 'Li': 1 -> 'Li-6': 0.0759, 'Li-7': 0.9241).
    """
    mapping = {}  # placeholder -> 'Li-6'

    def repl(m):
        g = m.groups()
        if g[0]:
            sym, a = g[1], g[0]
        elif g[2]:
            sym, a = g[3], g[2]
        else:
            sym, a = g[4], g[5]
        label = f"{sym}-{a}"
        for ph, lab in mapping.items():
            if lab == label:
                return ph
        ph = _PLACEHOLDERS[len(mapping)]  # chemparse sees an unknown 'element'
        mapping[ph] = label
        return ph

    cleaned = _ISO.sub(repl, formula)
    counts = parse_formula(cleaned)

    out = {}
    for sym, n in counts.items():
        if sym in mapping:
            out[mapping[sym]] = out.get(mapping[sym], 0) + n
        elif split_natural and sym in ABUNDANCE:
            for a, frac in ABUNDANCE[sym].items():
                key = f"{sym}-{a}"
                out[key] = out.get(key, 0) + n * frac
        else:
            out[sym] = out.get(sym, 0) + n
    return out


def lookup_name(key, available):
    """Map an isotope key to an entry that exists in the data.

    'Li-6' -> 'Li-6' if tabulated; otherwise fall back to the plain element
    ('C-12' -> 'C', whose entries are ~99% 12C anyway).
    """
    if key in available:
        return key
    sym = key.split("-")[0]
    return sym if sym in available else None


if __name__ == "__main__":
    print(parse_formula_isotopes("LiF"))        # natural Li, split by abundance
    print(parse_formula_isotopes("[6Li]F"))     # pure Li-6
    print(parse_formula_isotopes("Li-7(OH)"))   # pure Li-7
    print(parse_formula_isotopes("H2O"))        # no ambiguity with bare digits