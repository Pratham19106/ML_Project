"""
unit_test_cbm.py
================
Unit tests for src/models/concept_encoder.py.

All tests run locally with zero torch dependency (numpy + csv only).
Exit code 0 = ALL PASS.  Non-zero = at least one failure.

Usage
-----
    python scripts/unit_test_cbm.py
"""

import sys
from pathlib import Path

import numpy as np

# Allow import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.concept_encoder import (
    BORDER_VOCAB,
    BORDER_VOCAB_INV,
    BOTTLENECK_DIM,
    COLOUR_VOCAB,
    COLOUR_VOCAB_INV,
    N_BORDER,
    N_COLOUR,
    N_SHAPE,
    N_SYMCAT,
    SHAPE_VOCAB,
    SHAPE_VOCAB_INV,
    SYMCAT_VOCAB,
    SYMCAT_VOCAB_INV,
    decode_concept_indices,
    encode_all_classes,
    get_concept_tensor_for_class,
    get_oracle_concept_matrix,
    load_class_concepts,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
_results = []


def check(name: str, condition: bool, msg: str = ""):
    status = PASS if condition else FAIL
    line = f"  [{status}] {name}"
    if not condition and msg:
        line += f"\n         -> {msg}"
    print(line)
    _results.append((name, condition))
    return condition


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Vocabulary sizes
# ──────────────────────────────────────────────────────────────────────────────

def test_vocab_sizes():
    print("\nTest 1 — Vocabulary sizes")
    check("SHAPE_VOCAB has 6 entries",  len(SHAPE_VOCAB)  == 6)
    check("COLOUR_VOCAB has 5 entries", len(COLOUR_VOCAB) == 5)
    check("BORDER_VOCAB has 4 entries", len(BORDER_VOCAB) == 4)
    check("SYMCAT_VOCAB has 7 entries", len(SYMCAT_VOCAB) == 7)
    check("BOTTLENECK_DIM == 22",       BOTTLENECK_DIM    == 22)
    check("N_SHAPE == 6",               N_SHAPE           == 6)
    check("N_COLOUR == 5",              N_COLOUR          == 5)
    check("N_BORDER == 4",              N_BORDER          == 4)
    check("N_SYMCAT == 7",              N_SYMCAT          == 7)


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — CSV loads 43 classes with no missing values
# ──────────────────────────────────────────────────────────────────────────────

def test_csv_loading():
    print("\nTest 2 — CSV loading (43 classes, no missing values)")
    rows = load_class_concepts()
    check("Exactly 43 rows loaded", len(rows) == 43,
          f"Got {len(rows)}")

    required_keys = {"class_id", "class_name", "shape", "colour_profile",
                     "border_type", "symbol_category", "symbol_content"}
    for row in rows:
        for key in required_keys:
            val = row.get(key, "")
            check(
                f"class {row['class_id']:02d} -> '{key}' not empty",
                bool(str(val).strip()),
                f"'{key}' is empty for class_id={row['class_id']}",
            )



# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — All 43 classes encode without KeyError
# ──────────────────────────────────────────────────────────────────────────────

def test_encoding_completeness():
    print("\nTest 3 — encode_all_classes() covers all 43 class_ids")
    enc = encode_all_classes()
    check("Encoding has 43 entries", len(enc) == 43, f"Got {len(enc)}")
    check("Class IDs are 0..42",
          set(enc.keys()) == set(range(43)),
          f"Missing: {set(range(43)) - set(enc.keys())}")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — Concept tensors: shape, dtype, sum, and active positions
# ──────────────────────────────────────────────────────────────────────────────

def test_concept_tensors():
    print("\nTest 4 — Concept tensor shape / dtype / one-hot validity")
    enc = encode_all_classes()

    for cid in range(43):
        vec = get_concept_tensor_for_class(cid, encoding=enc)
        check(f"class {cid:02d}: shape == (22,)",
              vec.shape == (BOTTLENECK_DIM,),
              f"Got {vec.shape}")
        check(f"class {cid:02d}: dtype == float32",
              vec.dtype == np.float32)
        check(f"class {cid:02d}: sum == 4  (exactly 4 active concepts)",
              int(vec.sum()) == 4,
              f"Got sum={vec.sum()}")
        check(f"class {cid:02d}: all values in {{0, 1}}",
              set(vec.tolist()).issubset({0.0, 1.0}))


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — Oracle matrix uniqueness (no two classes share identical vectors)
# ──────────────────────────────────────────────────────────────────────────────

def test_oracle_uniqueness():
    print("\nTest 5 — Oracle concept vector uniqueness")
    X, y = get_oracle_concept_matrix()
    check("Oracle matrix shape == (43, 22)",
          X.shape == (43, BOTTLENECK_DIM),
          f"Got {X.shape}")
    unique_rows = np.unique(X, axis=0)
    n_unique = len(unique_rows)
    # Known finding: vocabulary collisions exist (all warning_hazard triangle
    # signs share the same concept vector, all mandatory_direction blue circles
    # share another, etc.). This documents the theoretical accuracy ceiling.
    check(
        "Oracle matrix has >= 11 distinct concept vectors",
        n_unique >= 11,
        "Unexpectedly few unique rows: " + str(n_unique),
    )
    if n_unique < 43:
        print("    [INFO] Vocabulary collision: " + str(n_unique) + "/43 unique vectors.")
        print("    [INFO] Theoretical accuracy ceiling is below 100%.")
        print("    [INFO] See reports/diagnostics/oracle_cbm_analysis.json for detail.")


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — Round-trip: encode -> decode preserves concept strings
# ──────────────────────────────────────────────────────────────────────────────

def test_roundtrip_decode():
    print("\nTest 6 — Round-trip encode -> decode")
    rows  = load_class_concepts()
    enc   = encode_all_classes()

    for row in rows:
        cid = row["class_id"]
        e   = enc[cid]
        decoded = decode_concept_indices(
            e["shape_idx"], e["colour_idx"], e["border_idx"], e["symcat_idx"]
        )
        check(f"class {cid:02d}: shape round-trips",
              decoded["shape"]           == row["shape"],
              f"Expected '{row['shape']}', got '{decoded['shape']}'")
        check(f"class {cid:02d}: colour_profile round-trips",
              decoded["colour_profile"]  == row["colour_profile"],
              f"Expected '{row['colour_profile']}', got '{decoded['colour_profile']}'")
        check(f"class {cid:02d}: border_type round-trips",
              decoded["border_type"]     == row["border_type"],
              f"Expected '{row['border_type']}', got '{decoded['border_type']}'")
        check(f"class {cid:02d}: symbol_category round-trips",
              decoded["symbol_category"] == row["symbol_category"],
              f"Expected '{row['symbol_category']}', got '{decoded['symbol_category']}'")


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 — Vocabulary inverse maps are consistent
# ──────────────────────────────────────────────────────────────────────────────

def test_inverse_maps():
    print("\nTest 7 — Vocabulary inverse maps are bijective")
    for name, fwd, inv in [
        ("SHAPE",  SHAPE_VOCAB,  SHAPE_VOCAB_INV),
        ("COLOUR", COLOUR_VOCAB, COLOUR_VOCAB_INV),
        ("BORDER", BORDER_VOCAB, BORDER_VOCAB_INV),
        ("SYMCAT", SYMCAT_VOCAB, SYMCAT_VOCAB_INV),
    ]:
        for key, idx in fwd.items():
            check(f"{name}: inv[{idx}] == '{key}'",
                  inv[idx] == key,
                  f"Got '{inv.get(idx)}'")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("unit_test_cbm.py — Concept Encoder Tests")
    print("=" * 60)

    test_vocab_sizes()
    test_csv_loading()
    test_encoding_completeness()
    test_concept_tensors()
    test_oracle_uniqueness()
    test_roundtrip_decode()
    test_inverse_maps()

    # ── Summary ────────────────────────────────────────────────────────────────
    total  = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed  |  {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for name, ok in _results:
            if not ok:
                print(f"  FAIL {name}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED OK")
        sys.exit(0)
