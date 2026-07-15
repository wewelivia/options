"""
Verify the xbbg return-shape helpers handle the shapes xbbg actually produces,
so the live path does not raise the 'DataFrame has no attribute iloc' /
tuple-column class of errors. Runs without a Terminal by faking a `blp` module.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import datetime as dt
import numpy as np
import pandas as pd

from app.data.bloomberg import (_first_scalar, _flat_columns, _flatten_bdp,
                                 BloombergProvider)
from app.data.chain_builder import _build_live


def test_first_scalar_shapes():
    # flat single-row frame
    assert _first_scalar(pd.DataFrame({"px_last": [5500.0]}, index=["SPX Index"])) == 5500.0
    # MultiIndex column frame
    mi = pd.DataFrame([[5500.0]], index=["SPX Index"],
                      columns=pd.MultiIndex.from_tuples([("SPX Index", "px_last")]))
    assert _first_scalar(mi) == 5500.0
    # Series
    assert _first_scalar(pd.Series([1.08], index=["px_last"])) == 1.08
    # empty
    assert _first_scalar(pd.DataFrame()) is None
    print("first_scalar: OK")


def test_flat_columns_tuples():
    mi = pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples(
        [("AAPL", "Security Description"), ("AAPL", "Strike")]))
    flat = _flat_columns(mi)
    assert any("security" in f for f in flat), flat
    print("flat_columns:", flat)


def test_flatten_bdp_multiindex():
    mi = pd.DataFrame(
        [[0.20, 5500.0]], index=["OPT1"],
        columns=pd.MultiIndex.from_tuples([("OPT1", "IVOL_MID"), ("OPT1", "OPT_STRIKE_PX")]))
    out = _flatten_bdp(mi)
    assert list(out.columns) == ["ivol_mid", "opt_strike_px"], list(out.columns)
    print("flatten_bdp cols:", list(out.columns))


class FakeBlp:
    """Mimics xbbg.blp with MultiIndex columns (the awkward case)."""
    def __init__(self):
        self.as_of = dt.date.today()

    def bdp(self, tickers, flds):
        if isinstance(tickers, str):  # spot()
            return pd.DataFrame([[5500.0]], index=[tickers],
                                columns=pd.MultiIndex.from_tuples([(tickers, "PX_LAST")]))
        # option_fields bulk
        rows, idx = [], []
        exp = self.as_of + dt.timedelta(days=160)
        F = 5500.0
        for i, tk in enumerate(tickers):
            K = 4500 + i * 100
            iv = 20.0 + 0.002 * (K - F)  # percent, downward skew-ish
            cp = "Call" if K >= F else "Put"
            rows.append([np.nan, np.nan, 10.0, iv, float(K), exp, cp, 1000, 50, F])
            idx.append(tk)
        cols = ["PX_BID", "PX_ASK", "PX_LAST", "IVOL_MID", "OPT_STRIKE_PX",
                "OPT_EXPIRE_DT", "OPT_PUT_CALL", "OPEN_INT", "PX_VOLUME", "OPT_UNDL_PX"]
        mi = pd.MultiIndex.from_tuples([(tk, c) for c in cols for tk in [idx[0]]]) if False else \
             pd.MultiIndex.from_product([["X"], cols])
        return pd.DataFrame(rows, index=idx, columns=mi)

    def bds(self, tickers, flds):
        # OPT_CHAIN members with a tuple/MultiIndex column
        secs = [f"SPX {tickers} C{k}" for k in range(4500, 6600, 100)]
        return pd.DataFrame({("SPX Index", "Security Description"): secs})


def test_build_live_with_fake_blp():
    prov = BloombergProvider.__new__(BloombergProvider)  # bypass __init__ imports
    prov._xbbg = FakeBlp()
    prov._blpapi_ok = False

    # spot
    assert prov.spot("SPX Index") == 5500.0
    # chain members
    members = prov.chain_tickers("SPX Index")
    assert len(members) > 5, members
    # full live build
    chain = _build_live(prov, "SPX Index", n_expiries=6, max_options=1500)
    assert chain.source == "bloomberg"
    assert chain.expiries, "no expiries assembled"
    sl = chain.expiries[0]
    ks, vs = sl.smile()
    print(f"build_live: {len(members)} members, expiry {sl.expiry}, {len(ks)} smile strikes, "
          f"vols {vs.min():.3f}-{vs.max():.3f}")
    assert len(ks) >= 5
    assert 0.05 < vs.mean() < 1.0, "IVs should be normalised to decimals"


if __name__ == "__main__":
    test_first_scalar_shapes()
    test_flat_columns_tuples()
    test_flatten_bdp_multiindex()
    test_build_live_with_fake_blp()
    print("\nALL XBBG-SHAPE TESTS PASSED")
