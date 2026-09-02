"""CFTC market code -> a price series, hand-audited one entry at a time.

WHY THIS IS AN EXPLICIT MAP AND NEVER A DERIVED ONE

CFTC's own grouping columns cannot do this job. `cftc_commodity_code` is 133 for
BOTH CME Bitcoin and Bitcoin Cash, so folding on it silently merges two different
assets; market names get renamed wholesale (2022-02-08) and 26-30% of codes have
carried more than one name. Every entry below was resolved against the live
endpoint before it was written down, and anything that did not resolve is absent
rather than guessed.

EVERY PRICE HERE IS A PROXY, AND THE PANEL SAYS SO

CFTC reports positioning in a specific futures contract. The series below are the
underlying index, a front-month continuous future, or an ETF -- never the exact
contract. The gap is largest where it is least obvious:

  - VIX FUTURES positioning against the VIX SPOT index. VIX futures trade at their
    own term structure and routinely disagree with spot in both level and
    direction. This is the one entry most likely to be misread, so `proxy` names
    what it actually is.
  - Front-month continuous futures (ZN=F, 6E=F, ...) splice across expiries, so
    their level has roll effects the underlying does not.
  - MSCI EM / EAFE / DJ Real Estate use liquid ETFs, which carry fees and tracking
    error and are quoted in share terms, not index terms.

`kind` drives the caption. `label` is what the reader sees.

NOT MAPPED, deliberately: 3-month SOFR (134741, the largest market in the file by
open interest). Its Yahoo symbol SR3=F resolves but returns a single observation,
which is worse than nothing. A market with no entry simply gets no price panel.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceRef:
    symbol: str
    label: str
    #: index | future | etf -- what the series actually is, for the caption.
    kind: str
    #: One line naming the gap between this series and the reported contract.
    proxy: str = ""

    @property
    def is_exact(self) -> bool:
        return False  # nothing here is the reported contract itself


_IDX = "index"
_FUT = "future"
_ETF = "etf"

#: Verified 2026-09-02: every symbol below returned daily history with
#: dataGranularity="1d". The count is the observations available from 1990.
MAP: dict[str, PriceRef] = {
    # ---- equity indices. The Consolidated code and both its legs share a series,
    # since they are the same underlying at different contract sizes.
    "13874+": PriceRef("^GSPC", "S&P 500", _IDX),
    "13874A": PriceRef("^GSPC", "S&P 500", _IDX),
    "13874U": PriceRef("^GSPC", "S&P 500", _IDX),
    "20974+": PriceRef("^NDX", "NASDAQ-100", _IDX),
    "209742": PriceRef("^NDX", "NASDAQ-100", _IDX),
    "209747": PriceRef("^NDX", "NASDAQ-100", _IDX),
    "12460+": PriceRef("^DJI", "Dow Jones Industrial Average", _IDX),
    "124603": PriceRef("^DJI", "Dow Jones Industrial Average", _IDX),
    "124608": PriceRef("^DJI", "Dow Jones Industrial Average", _IDX),
    "239742": PriceRef("^RUT", "Russell 2000", _IDX),
    "239747": PriceRef("^RUT", "Russell 2000", _IDX),
    "33874A": PriceRef("^SP400", "S&P MidCap 400", _IDX),
    "244042": PriceRef(
        "EEM", "MSCI Emerging Markets (EEM)", _ETF,
        "an ETF tracking the index, so it carries fees and tracking error",
    ),
    "244041": PriceRef(
        "EFA", "MSCI EAFE (EFA)", _ETF,
        "an ETF tracking the index, so it carries fees and tracking error",
    ),
    "124606": PriceRef(
        "IYR", "DJ US Real Estate (IYR)", _ETF,
        "an ETF tracking the index, so it carries fees and tracking error",
    ),
    # ---- volatility. The proxy note here matters more than anywhere else.
    "1170E1": PriceRef(
        "^VIX", "VIX index (spot)", _IDX,
        "SPOT VIX, not the futures CFTC reports on. VIX futures carry their own "
        "term structure and can move against spot in both level and direction",
    ),
    # ---- US rates. Front-month continuous futures.
    "020601": PriceRef("ZB=F", "30Y T-bond future", _FUT, "front-month continuous, so it splices across expiries"),
    "020604": PriceRef("UB=F", "Ultra T-bond future", _FUT, "front-month continuous, so it splices across expiries"),
    "043602": PriceRef("ZN=F", "10Y T-note future", _FUT, "front-month continuous, so it splices across expiries"),
    "043607": PriceRef("TN=F", "Ultra 10Y future", _FUT, "front-month continuous, so it splices across expiries"),
    "044601": PriceRef("ZF=F", "5Y T-note future", _FUT, "front-month continuous, so it splices across expiries"),
    "042601": PriceRef("ZT=F", "2Y T-note future", _FUT, "front-month continuous, so it splices across expiries"),
    "045601": PriceRef("ZQ=F", "30-day Fed funds future", _FUT, "front-month continuous, so it splices across expiries"),
    # ---- FX. CME futures, quoted in USD per unit.
    "099741": PriceRef("6E=F", "Euro FX future", _FUT, "front-month continuous"),
    "097741": PriceRef("6J=F", "Japanese yen future", _FUT, "front-month continuous"),
    "096742": PriceRef("6B=F", "British pound future", _FUT, "front-month continuous"),
    "090741": PriceRef("6C=F", "Canadian dollar future", _FUT, "front-month continuous"),
    "232741": PriceRef("6A=F", "Australian dollar future", _FUT, "front-month continuous"),
    "092741": PriceRef("6S=F", "Swiss franc future", _FUT, "front-month continuous"),
    "112741": PriceRef("6N=F", "NZ dollar future", _FUT, "front-month continuous"),
    "095741": PriceRef("6M=F", "Mexican peso future", _FUT, "front-month continuous"),
    "102741": PriceRef("6L=F", "Brazilian real future", _FUT, "front-month continuous"),
    "098662": PriceRef("DX-Y.NYB", "US dollar index", _IDX),
    # ---- broad commodity
    "221602": PriceRef("^BCOM", "Bloomberg Commodity index", _IDX),
    # ---- digital assets. Spot, against futures positioning.
    "133741": PriceRef("BTC-USD", "Bitcoin (spot)", _IDX, "spot, not the CME future"),
    "133742": PriceRef("BTC-USD", "Bitcoin (spot)", _IDX, "spot, not the CME future"),
    "133LM1": PriceRef("BTC-USD", "Bitcoin (spot)", _IDX, "spot, not the listed future"),
    "133LM4": PriceRef("BTC-USD", "Bitcoin (spot)", _IDX, "spot, not the listed future"),
    "146021": PriceRef("ETH-USD", "Ether (spot)", _IDX, "spot, not the CME future"),
    "146022": PriceRef("ETH-USD", "Ether (spot)", _IDX, "spot, not the CME future"),
    "146LM1": PriceRef("ETH-USD", "Ether (spot)", _IDX, "spot, not the listed future"),
    "146LM3": PriceRef("ETH-USD", "Ether (spot)", _IDX, "spot, not the listed future"),
    "176LM1": PriceRef("XRP-USD", "XRP (spot)", _IDX, "spot, not the listed future"),
    "176LM3": PriceRef("XRP-USD", "XRP (spot)", _IDX, "spot, not the listed future"),
    "177LM3": PriceRef("SOL-USD", "Solana (spot)", _IDX, "spot, not the listed future"),
}

#: The distinct symbols to ingest. Several CFTC codes share one series, so this is
#: shorter than MAP.
SYMBOLS: tuple[str, ...] = tuple(sorted({r.symbol for r in MAP.values()}))


def for_code(code: str) -> PriceRef | None:
    """The price series for a CFTC market code, or None if none is mapped."""
    return MAP.get(str(code).strip())
