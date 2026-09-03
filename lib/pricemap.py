"""CFTC market code -> a price series, hand-audited one entry at a time.

WHY THIS IS AN EXPLICIT MAP AND NEVER A DERIVED ONE

CFTC's own grouping columns cannot do this job. `cftc_commodity_code` is 133 for
BOTH CME Bitcoin and Bitcoin Cash -- and the file really does carry both, `133LM4
NANO BITCOIN PERP STYLE` and `133LM5 BITCOIN CASH PERP STYLE`, so folding on that
column silently merges two different assets. Look for them below: adjacent entries,
different symbols, one line each. Market names get renamed wholesale (2022-02-08)
and 26-30% of codes have carried more than one name, so the name is not a key
either. Every entry below was resolved against the live endpoint before it was
written down, and anything that did not resolve is absent rather than guessed.

WHAT EARNS AN ENTRY

Four conditions, all of them:

  1. The code is an OUTRIGHT contract. A basis, a location differential, a
     calendar or crack spread, a monthly-index settlement and a dividend index all
     fail this: their value is a difference or an average, not the level of any
     tradeable series, so no price series can stand in for them. This is what
     excludes most of the file -- 900-odd natural-gas basis, electricity-hub and
     REC/carbon codes have no underlying quote anywhere, not just none on Yahoo.
  2. A symbol resolving to that underlying returned DAILY bars from the live
     endpoint (`dataGranularity == "1d"`, explicit period bounds).
  3. That symbol's history covers the MAJORITY of the code's reporting window.
     `lib.metrics.asof` joins strictly backward and carries the last close forward
     with no staleness cap, so a series that stops early draws a flat line that
     reads as "the price stopped moving." A series that starts late is milder --
     early reports simply get NaN -- but a code whose price is missing for most of
     its life gets no panel rather than a mostly-empty one.
  4. Peak open interest reached 10,000 contracts somewhere in its own history.
     Below that a panel is decoration.

Two documented exceptions to (4), and they are the only two -- an undocumented third
would turn the floor into a suggestion:

  `133LM5 BITCOIN CASH PERP STYLE` peaked at 3,243. It is here because it is the code
      the first paragraph warns about, and a map that demonstrates the hazard beats
      one that only describes it.
  `1330E1 BITCOIN-USD - CBOE FUTURES EXCHANGE` peaked at 6,932 over 145 report weeks
      (2017-12-19 to 2019-05-07). It is the Cboe XBT contract: the historical start of
      listed bitcoin positioning, and the only record of it. Small because the product
      failed, not because the market was uninteresting.

EVERY PRICE HERE IS A PROXY, AND THE PANEL SAYS SO

CFTC reports positioning in a specific futures contract. The series below are the
underlying index, a front-month continuous future, or an ETF -- never the exact
contract. The gap is largest where it is least obvious:

  - VIX FUTURES positioning against the VIX SPOT index. VIX futures trade at their
    own term structure and routinely disagree with spot in both level and
    direction. This is the one entry most likely to be misread, so `proxy` names
    what it actually is.
  - Front-month continuous futures (ZN=F, 6E=F, GC=F, CL=F, ...) splice across
    expiries. Every roll moves the level by the calendar spread, so a multi-year
    move in one of these mixes price change with accumulated roll yield. In
    contango-heavy markets -- natural gas, crude in 2015-2020 -- that term is
    large enough to reverse the sign of a long-horizon "return".
  - MSCI EM / EAFE / DJ Real Estate and the eleven S&P Select Sector legs use
    liquid ETFs, which carry fees and tracking error and are quoted in share terms,
    not index terms.
  - Grains, softs and livestock quote in CENTS (the endpoint returns currency
    "USX"), so their panel numbers are two orders of magnitude off the dollar
    figure a reader may expect. Named per entry where it applies.

`kind` drives the caption. `label` is what the reader sees.

NOT MAPPED, and why -- each of these was probed, not assumed:

  134741 / 134742 / 134FM1  SOFR-3M and SOFR-1M. 134741 is the largest market in
      the file by open interest. SR3=F and SR1=F both resolve and both return a
      SINGLE observation, which is worse than nothing.
  132741's short-tenor sibling 032741 ONE-MONTH EURODOLLAR. GE=F is the THREE-month
      contract; a different tenor is a different instrument, not a proxy.
  041741  13-WEEK U.S. TREASURY BILLS. ^IRX resolves with 9,203 daily points, but
      it is a YIELD. Plotting it under a futures position inverts the direction.
  001626  WHEAT-HRSpring. MW=F returns HTTP 404.
  135731  CANOLA (366k contracts). RS=F resolves with the right granularity and
      then yields zero non-null closes.
  025651  ETHANOL. EH=F holds 5,015 daily bars but its last close is 2025-04-03
      while positioning runs to 2026-08-25 -- condition (3): a carried-forward
      close would draw seventeen flat months.
  189691  LITHIUM HYDROXIDE. LTH=F resolves with one observation.
  191693 / 191691 / 191696  ALUMINUM and ALUMINUM MWP. CME lists both a full
      delivered-Midwest price (ALI) and a Midwest-premium-only contract (AUP), both
      at 25 metric tons, and "ALUMINUM MWP" does not decide between them. Their
      levels differ by the whole LME base. Only ALI has a symbol, so a guess here
      would be an order-of-magnitude error, not a small one.
  180LM2 / 180LM3  SUI. SUI-USD RESOLVES -- and returns "Salmonation USD", a
      different token, 810 points ending 2024-06-04 at $0.0003. The clearest case
      in this file for probing rather than pattern-matching a ticker.
  089741  RUSSIAN RUBLE (peak 133k). 6R=F resolves with 5,654 points whose last
      close is 0.0 and whose currency is null; RUB=X is quoted rubles-per-dollar
      against a contract quoted dollars-per-ruble, i.e. inverted.
  094741 DEUTSCHE MARK (DEM=X: one observation), 096741 POUND STERLING-OLD,
      085691 COPPER, 054641 LIVE HOGS, 002601/005601/001601 the "in 1000 Bushels"
      grains, 148771 NYSE CMP: all retired between 1987 and 2000, before their
      symbol's first observation. Zero overlap fails condition (3).
  111652  UNLEADED GASOLINE- N.Y. HARBOR (peak 180k, 901 reports 1986-2006). RB=F
      begins 2000-11 and covers 6 of its 21 reporting years, and RBOB is a
      different blendstock spec besides.
  43874A / 43874Q / 239750  the S&P 500 and Russell 2000 DIVIDEND index futures.
      Their underlying is a realized-dividend index; ^GSPC and ^RUT are price
      indices. Same name, unrelated series.
  343603 / 342603 / 344605 / 344606 / 344607  ERIS SOFR SWAP futures, 047745 EURO
      SHORT TERM RATE, 188691 COBALT, 052644 CME MILK IV, and the Mt Belvieu
      propane/butane/ethane/natural-gasoline OPIS strip (B0=F: one observation) --
      no listed daily series for the underlying.

A market with no entry simply gets no price panel; the board is built to degrade
that way.
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

#: Verified 2026-09-02 (financials) and 2026-09-03 (everything added since): every
#: symbol below returned daily history with dataGranularity="1d" from explicit
#: period bounds. Observation counts quoted in the comments are from 1990.
MAP: dict[str, PriceRef] = {
    # ---- equity indices. The Consolidated code and both its legs share a series,
    # since they are the same underlying at different contract sizes.
    "13874+": PriceRef("^GSPC", "S&P 500", _IDX),
    "13874A": PriceRef("^GSPC", "S&P 500", _IDX),
    "13874U": PriceRef("^GSPC", "S&P 500", _IDX),
    "138741": PriceRef(
        "^GSPC", "S&P 500", _IDX,
        "the full-size $500-multiplier pit contract, delisted 2021-09; the index "
        "series starts 1990 and its first four reporting years have no price",
    ),
    "20974+": PriceRef("^NDX", "NASDAQ-100", _IDX),
    "209742": PriceRef("^NDX", "NASDAQ-100", _IDX),
    "209747": PriceRef("^NDX", "NASDAQ-100", _IDX),
    "209741": PriceRef(
        "^NDX", "NASDAQ-100", _IDX,
        "the full-size contract, delisted 2015-06; same index, 100x the E-mini",
    ),
    "12460+": PriceRef("^DJI", "Dow Jones Industrial Average", _IDX),
    "124603": PriceRef("^DJI", "Dow Jones Industrial Average", _IDX),
    "124608": PriceRef("^DJI", "Dow Jones Industrial Average", _IDX),
    "124601": PriceRef(
        "^DJI", "Dow Jones Industrial Average", _IDX,
        "the CBOT $10-multiplier contract, delisted 2014-12",
    ),
    "239742": PriceRef("^RUT", "Russell 2000", _IDX),
    "239747": PriceRef("^RUT", "Russell 2000", _IDX),
    "239741": PriceRef(
        "^RUT", "Russell 2000", _IDX,
        "the full-size CME contract, delisted 2008-09",
    ),
    "23977A": PriceRef(
        "^RUT", "Russell 2000", _IDX,
        "the ICE mini listing that held the franchise 2008-2018, peaking at 760k "
        "contracts, before it moved back to CME",
    ),
    "33874A": PriceRef("^SP400", "S&P MidCap 400", _IDX),
    "338741": PriceRef(
        "^SP400", "S&P MidCap 400", _IDX,
        "the full-size midcap contract, delisted 2008-03",
    ),
    "13874W": PriceRef(
        "^SP500TR", "S&P 500 Total Return", _IDX,
        "the TOTAL-RETURN index, which is the right underlying here and is NOT "
        "^GSPC: it reinvests dividends, so it compounds above the price index",
    ),
    "13874N": PriceRef(
        "^SP500TR", "S&P 500 Total Return", _IDX,
        "the predecessor total-return contract, delisted 2021-03; same index",
    ),
    "244042": PriceRef(
        "EEM", "MSCI Emerging Markets (EEM)", _ETF,
        "an ETF tracking the index, so it carries fees and tracking error",
    ),
    "244742": PriceRef(
        "EEM", "MSCI Emerging Markets (EEM)", _ETF,
        "the 2009-2011 E-mini listing; an ETF proxy, so fees and tracking error",
    ),
    "244041": PriceRef(
        "EFA", "MSCI EAFE (EFA)", _ETF,
        "an ETF tracking the index, so it carries fees and tracking error",
    ),
    "244741": PriceRef(
        "EFA", "MSCI EAFE (EFA)", _ETF,
        "the 2008-2011 E-mini listing; an ETF proxy, so fees and tracking error",
    ),
    "24477M": PriceRef(
        "ACWI", "MSCI ACWI (ACWI)", _ETF,
        "the contract settles on ACWI NET TOTAL RETURN; this ETF is a price series "
        "paying its dividends out, so the two diverge by the distribution stream",
    ),
    "124606": PriceRef(
        "IYR", "DJ US Real Estate (IYR)", _ETF,
        "an ETF tracking the index, so it carries fees and tracking error",
    ),
    "239744": PriceRef(
        "IWD", "Russell 1000 Value (IWD)", _ETF,
        "an ETF tracking the index, so it carries fees and tracking error",
    ),
    "240743": PriceRef(
        "^N225", "Nikkei 225", _IDX,
        "the index in yen, matching this yen-denominated contract; CME's own "
        "settlement is the Osaka print, a few hours later than this close",
    ),
    "240741": PriceRef(
        "^N225", "Nikkei 225", _IDX,
        "the index in YEN against a contract that settles in DOLLARS, so this "
        "series omits the USDJPY leg the position is actually exposed to",
    ),
    # ---- S&P Select Sector futures, against the eleven Select Sector SPDRs. The
    # futures settle on the index; the ETF is the index minus fees plus tracking
    # error, and is quoted per share rather than in index points.
    "13874C": PriceRef("XLF", "Financial Select Sector (XLF)", _ETF, "an ETF on the index the contract settles against, in share terms not index points"),
    "138748": PriceRef("XLP", "Consumer Staples Select Sector (XLP)", _ETF, "an ETF on the index the contract settles against, in share terms not index points"),
    "138749": PriceRef("XLE", "Energy Select Sector (XLE)", _ETF, "an ETF on the index the contract settles against, in share terms not index points"),
    "13874I": PriceRef("XLK", "Technology Select Sector (XLK)", _ETF, "an ETF on the index the contract settles against, in share terms not index points"),
    "13874E": PriceRef("XLV", "Health Care Select Sector (XLV)", _ETF, "an ETF on the index the contract settles against, in share terms not index points"),
    "13874J": PriceRef("XLU", "Utilities Select Sector (XLU)", _ETF, "an ETF on the index the contract settles against, in share terms not index points"),
    "13874F": PriceRef("XLI", "Industrial Select Sector (XLI)", _ETF, "an ETF on the index the contract settles against, in share terms not index points"),
    "13874H": PriceRef("XLB", "Materials Select Sector (XLB)", _ETF, "an ETF on the index the contract settles against, in share terms not index points"),
    "13874P": PriceRef(
        "XLC", "Communication Services Select Sector (XLC)", _ETF,
        "an ETF proxy, and the youngest of the eleven: it only starts 2018-06, when "
        "the sector was carved out of technology and telecoms",
    ),
    "13874R": PriceRef(
        "XLRE", "Real Estate Select Sector (XLRE)", _ETF,
        "an ETF proxy starting 2015-10, when real estate left the financials sector",
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
    "132741": PriceRef(
        "GE=F", "3-month Eurodollar future", _FUT,
        "the contract itself, front-month continuous. Both series end together at "
        "the June 2023 SOFR transition; the symbol begins 2000-09, so the first "
        "fourteen of this market's thirty-seven reporting years have no price",
    ),
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
    "122741": PriceRef(
        "6Z=F", "South African rand future", _FUT,
        "front-month continuous, and it begins 2001-04 against reports from "
        "1998-06, so the first three years have no price",
    ),
    "299741": PriceRef(
        "EURGBP=X", "EUR/GBP spot", _IDX,
        "the SPOT cross, not the CME futures leg; the two differ by the interest "
        "differential priced into the contract's forward date",
    ),
    "399741": PriceRef(
        "EURJPY=X", "EUR/JPY spot", _IDX,
        "the SPOT cross, not the CME futures leg; the two differ by the interest "
        "differential priced into the contract's forward date",
    ),
    # 299661 EURO FX/JAPANESE YEN - SMALL is deliberately ABSENT. EURJPY=X resolves,
    # but its first bar is 2003-01-23 against reports from 1999-01-19, so only
    # 56 of 125 report weeks (44.8%) are priced -- the lowest of any candidate and
    # the only one under half. That fails condition (3), the same test that rejects
    # RB=F and the pre-2000 grains, and a rule with a silent exception is not a rule.
    "098662": PriceRef("DX-Y.NYB", "US dollar index", _IDX),
    # ---- broad commodity indices
    "221602": PriceRef("^BCOM", "Bloomberg Commodity index", _IDX),
    "256741": PriceRef(
        "^SPGSCI", "S&P GSCI", _IDX,
        "the index itself; the contract was delisted 2017-09 while the index runs on",
    ),
    # ---- energy: natural gas. Every code here settles on Henry Hub one way or
    # another, so they share NG=F -- the NYMEX front-month continuous. What differs
    # between them is contract size, settlement date and venue, and the proxy line
    # says which.
    "023651": PriceRef(
        "NG=F", "Henry Hub natural gas future", _FUT,
        "front-month continuous of THIS contract. Gas is the worst market in the "
        "file for roll effects: seasonal contango moves the splice by a large "
        "fraction of the level every year",
    ),
    "023391": PriceRef(
        "NG=F", "Henry Hub natural gas future", _FUT,
        "the ICE last-day financial swap settles on the NYMEX price this series "
        "follows, at 2,500 MMBtu against NYMEX's 10,000; plus the roll splice",
    ),
    "023392": PriceRef(
        "NG=F", "Henry Hub natural gas future", _FUT,
        "the ICE PENULTIMATE swap settles a business day earlier in the month than "
        "the NYMEX contract this series follows; plus the roll splice",
    ),
    "023A55": PriceRef(
        "NG=F", "Henry Hub natural gas future", _FUT,
        "financially settled against the same NYMEX last-day price this series is "
        "taken from, at 10,000 MMBtu; plus the roll splice",
    ),
    "023A56": PriceRef(
        "NG=F", "Henry Hub natural gas future", _FUT,
        "penultimate settlement, one business day before the NYMEX front month "
        "this series follows; plus the roll splice",
    ),
    "03565B": PriceRef(
        "NG=F", "Henry Hub natural gas future", _FUT,
        "the 2,500 MMBtu financial version of the contract this series follows, a "
        "quarter its size; plus the roll splice",
    ),
    "03565C": PriceRef(
        "NG=F", "Henry Hub natural gas future", _FUT,
        "penultimate-settled and a quarter the size of the NYMEX contract this "
        "series follows; plus the roll splice",
    ),
    "023P01": PriceRef(
        "NG=F", "Henry Hub natural gas future", _FUT,
        "the Nasdaq Futures Henry Hub financial, delisted 2018 with the rest of "
        "that venue's energy complex; same hub, different exchange",
    ),
    "023P03": PriceRef(
        "NG=F", "Henry Hub natural gas future", _FUT,
        "the Nasdaq Futures penultimate financial, delisted 2019; same hub, "
        "different exchange",
    ),
    # ---- energy: crude and refined products.
    "067651": PriceRef(
        "CL=F", "WTI crude future", _FUT,
        "front-month continuous of THIS contract -- the physically-delivered NYMEX "
        "WTI. Only the expiry splice separates them, and April 2020's negative "
        "settlement is in both",
    ),
    "067411": PriceRef(
        "CL=F", "WTI crude future", _FUT,
        "ICE's cash-settled WTI settles on the NYMEX price this series is taken "
        "from; the roll splice is what remains",
    ),
    "06765A": PriceRef(
        "CL=F", "WTI crude future", _FUT,
        "financially settled to the NYMEX WTI settlement this series follows; plus "
        "the roll splice",
    ),
    "06765I": PriceRef(
        "CL=F", "WTI crude future", _FUT,
        "the earlier NYMEX WTI financial, delisted 2013-09; same settlement price",
    ),
    "06741Q": PriceRef(
        "CL=F", "WTI crude future", _FUT,
        "an ICE first-line swap on the same NYMEX settlement; plus the roll splice",
    ),
    "067655": PriceRef(
        "CL=F", "WTI crude future", _FUT,
        "the half-size E-mini, delisted 2007-01; same barrel, 500 of them",
    ),
    "06765T": PriceRef(
        "BZ=F", "Brent crude future", _FUT,
        "front-month continuous of THIS contract -- the endpoint names it 'Brent "
        "Crude Oil Last Day Financial'. It begins 2007-07, four years BEFORE this "
        "code is first reported, so every report week is priced; what remains is "
        "the roll splice",
    ),
    "06765J": PriceRef(
        "BZ=F", "Brent crude future", _FUT,
        "the NYMEX Brent financial, delisted 2013-05; the same Brent settlement",
    ),
    "111659": PriceRef(
        "RB=F", "RBOB gasoline future", _FUT,
        "front-month continuous of this contract, quoted per gallon. Gasoline "
        "carries a hard seasonal spec change at each spring/autumn roll",
    ),
    "022651": PriceRef(
        "HO=F", "NY Harbor ULSD future", _FUT,
        "this IS the contract -- the endpoint still labels it 'Heating Oil', the "
        "name it carried before the 2013 sulphur re-spec. Front-month continuous",
    ),
    # ---- metals. Front-month continuous COMEX/NYMEX, so each roll shifts the
    # level by the carry the reported contract does not pay.
    "088691": PriceRef(
        "GC=F", "Gold future", _FUT,
        "front-month continuous COMEX gold; every roll adds the cost-of-carry "
        "spread, which in a high-rate year is visible at this level",
    ),
    "088695": PriceRef(
        "MGC=F", "Micro gold future", _FUT,
        "the micro contract itself, front-month continuous -- 10 troy oz against "
        "COMEX's 100, at the same price per ounce",
    ),
    "088606": PriceRef(
        "GC=F", "Gold future", _FUT,
        "the CBOT 100-oz gold contract, delisted 2008-07 after the CME/CBOT "
        "merger; GC=F is COMEX's listing of the same metal at the same size",
    ),
    "084691": PriceRef(
        "SI=F", "Silver future", _FUT,
        "front-month continuous COMEX silver, plus the carry in each roll",
    ),
    "084694": PriceRef(
        "SIL=F", "Micro silver future", _FUT,
        "the micro contract itself -- 1,000 troy oz against COMEX's 5,000",
    ),
    "085692": PriceRef(
        "HG=F", "Copper future", _FUT,
        "front-month continuous COMEX copper, quoted per pound. 2025's US tariff "
        "episode blew the COMEX-LME spread out, so this level is a US price, not a "
        "world one",
    ),
    "085699": PriceRef(
        "HG=F", "Copper future", _FUT,
        "the full-size COMEX series; the micro is a tenth the pounds at the same "
        "price per pound",
    ),
    "076651": PriceRef(
        "PL=F", "Platinum future", _FUT,
        "front-month continuous NYMEX platinum, plus the carry in each roll",
    ),
    "075651": PriceRef(
        "PA=F", "Palladium future", _FUT,
        "front-month continuous NYMEX palladium. Repeated physical squeezes have "
        "put this contract in backwardation for years at a stretch, so its rolls "
        "subtract from the level rather than add",
    ),
    "192651": PriceRef(
        "HRC=F", "Hot-rolled coil steel future", _FUT,
        "front-month continuous of this contract -- the endpoint names it 'U.S. "
        "Midwest Domestic Hot-Rolled Coil'. Cash-settled to a monthly index, so "
        "the daily series is itself an expectation of an average",
    ),
    # ---- grains and oilseeds. CBOT quotes these in CENTS: the endpoint returns
    # currency "USX" for every one marked below, so 538.75 is $5.3875 a bushel.
    "002602": PriceRef(
        "ZC=F", "Corn future", _FUT,
        "front-month continuous CBOT corn, quoted in CENTS per bushel. The "
        "Dec-to-Mar new-crop roll is the largest splice in the grain complex",
    ),
    "001602": PriceRef(
        "ZW=F", "Chicago SRW wheat future", _FUT,
        "front-month continuous, in CENTS per bushel. SRW specifically -- a "
        "different wheat class from the KC contract below",
    ),
    "001612": PriceRef(
        "KE=F", "KC HRW wheat future", _FUT,
        "front-month continuous, in CENTS per bushel. Hard red winter, whose basis "
        "against Chicago SRW has swung by more than a dollar a bushel",
    ),
    "004603": PriceRef(
        "ZO=F", "Oats future", _FUT,
        "front-month continuous, in CENTS per bushel. The thinnest CBOT grain, so "
        "its daily closes are noisier than the rest of this block",
    ),
    "039601": PriceRef(
        "ZR=F", "Rough rice future", _FUT,
        "front-month continuous, quoted in dollars per hundredweight -- not cents, "
        "unlike the other CBOT grains here",
    ),
    "005602": PriceRef(
        "ZS=F", "Soybean future", _FUT,
        "front-month continuous, in CENTS per bushel",
    ),
    "005603": PriceRef(
        "ZS=F", "Soybean future", _FUT,
        "the full-size series; the mini is 1,000 bushels against 5,000 at the same "
        "price per bushel, in CENTS",
    ),
    "007601": PriceRef(
        "ZL=F", "Soybean oil future", _FUT,
        "front-month continuous, in CENTS per pound. Biodiesel policy has driven "
        "this contract's level more than the crush since 2021",
    ),
    "026603": PriceRef(
        "ZM=F", "Soybean meal future", _FUT,
        "front-month continuous, quoted in dollars per short ton",
    ),
    "037021": PriceRef(
        "CPO=F", "USD Malaysian crude palm oil future", _FUT,
        "front-month continuous of this contract -- CME's dollar-denominated "
        "calendar swap on the Bursa Malaysia settlement, not the ringgit FCPO",
    ),
    # ---- softs and fibre. ICE contracts, all quoted in CENTS ("USX") except cocoa.
    "080732": PriceRef(
        "SB=F", "Sugar No. 11 future", _FUT,
        "front-month continuous, in CENTS per pound. No. 11 is the world raw "
        "contract, not the US domestic No. 16",
    ),
    "083731": PriceRef(
        "KC=F", "Coffee C future", _FUT,
        "front-month continuous, in CENTS per pound. Arabica only -- robusta "
        "trades in London and is not in this series",
    ),
    "073732": PriceRef(
        "CC=F", "Cocoa future", _FUT,
        "front-month continuous, in DOLLARS per metric ton, unlike its ICE "
        "neighbours here which quote in cents",
    ),
    "033661": PriceRef(
        "CT=F", "Cotton No. 2 future", _FUT,
        "front-month continuous, in CENTS per pound",
    ),
    "040701": PriceRef(
        "OJ=F", "FCOJ future", _FUT,
        "front-month continuous, in CENTS per pound of solids. The series begins "
        "2001-09 against reports from 1986, so the first fifteen years are blank",
    ),
    # ---- livestock. CME, quoted in CENTS per pound. These are the one block where
    # the front-month splice is structural rather than incidental: cattle and hogs
    # are seasonal and non-storable, so adjacent expiries are not arbitrage-linked
    # the way a metal's are, and the roll gap is a real price difference.
    "054642": PriceRef(
        "HE=F", "Lean hog future", _FUT,
        "front-month continuous, in CENTS per pound. Non-storable, so the roll gap "
        "is a genuine seasonal price step, not a carry",
    ),
    "057642": PriceRef(
        "LE=F", "Live cattle future", _FUT,
        "front-month continuous, in CENTS per pound. Non-storable, so the roll gap "
        "is a genuine seasonal price step, not a carry",
    ),
    "061641": PriceRef(
        "GF=F", "Feeder cattle future", _FUT,
        "front-month continuous, in CENTS per pound, cash-settled to the CME "
        "Feeder Cattle Index -- a seven-day average, so the daily print already "
        "smooths",
    ),
    # ---- dairy and lumber. All cash-settled to a monthly USDA or transaction
    # index, so the futures price is an expectation of an average that has not been
    # published yet -- a bigger gap than a roll.
    "052641": PriceRef(
        "DC=F", "Class III milk future", _FUT,
        "front-month continuous, cash-settled to the monthly USDA Class III price; "
        "the series begins 2006-04 against reports from 1997",
    ),
    "063642": PriceRef(
        "CSC=F", "Cash-settled cheese future", _FUT,
        "front-month continuous, settled to the monthly CME cheese price -- an "
        "expectation of an average, not a spot quote",
    ),
    "050642": PriceRef(
        "CB=F", "Cash-settled butter future", _FUT,
        "front-month continuous, in CENTS per pound, settled to the monthly CME "
        "butter price",
    ),
    "052642": PriceRef(
        "GNF=F", "Nonfat dry milk future", _FUT,
        "front-month continuous, in CENTS per pound, settled to the monthly USDA "
        "NFDM price",
    ),
    "058644": PriceRef(
        "LBR=F", "Lumber future", _FUT,
        "front-month continuous of the 27,500-board-foot contract that replaced "
        "random-length lumber in 2023; the series starts 2022-08, so it covers "
        "this code's whole reporting window and none of the old contract's",
    ),
    # ---- digital assets. Spot coin prices, against futures positioning: the CME
    # and Coinbase contracts settle to a reference rate fixed once a day, while
    # these series are 24/7 closes, so the two disagree by up to a session's move.
    "133741": PriceRef("BTC-USD", "Bitcoin (spot)", _IDX, "spot, not the CME future"),
    "133742": PriceRef("BTC-USD", "Bitcoin (spot)", _IDX, "spot, not the CME future"),
    "133LM1": PriceRef("BTC-USD", "Bitcoin (spot)", _IDX, "spot, not the listed future"),
    "133LM4": PriceRef("BTC-USD", "Bitcoin (spot)", _IDX, "spot, not the listed future"),
    # 133LM5 sits here on purpose. It carries cftc_commodity_code 133, the SAME
    # code as the four Bitcoin lines above, and it is a different asset. This is
    # the entry that makes the module docstring's first paragraph checkable.
    "133LM5": PriceRef(
        "BCH-USD", "Bitcoin Cash (spot)", _IDX,
        "spot BITCOIN CASH -- a different asset from the Bitcoin lines above, "
        "which share its cftc_commodity_code of 133. Kept below this file's "
        "10,000-contract floor (it peaked at 3,243) precisely to keep that "
        "collision visible",
    ),
    "1330E1": PriceRef(
        "BTC-USD", "Bitcoin (spot)", _IDX,
        "spot, against the Cboe XBT future that was delisted 2019-04",
    ),
    "146021": PriceRef("ETH-USD", "Ether (spot)", _IDX, "spot, not the CME future"),
    "146022": PriceRef("ETH-USD", "Ether (spot)", _IDX, "spot, not the CME future"),
    "146LM1": PriceRef("ETH-USD", "Ether (spot)", _IDX, "spot, not the listed future"),
    "146LM3": PriceRef("ETH-USD", "Ether (spot)", _IDX, "spot, not the listed future"),
    "176LM1": PriceRef("XRP-USD", "XRP (spot)", _IDX, "spot, not the listed future"),
    "176LM3": PriceRef("XRP-USD", "XRP (spot)", _IDX, "spot, not the listed future"),
    "176740": PriceRef("XRP-USD", "XRP (spot)", _IDX, "spot, not the CME future"),
    "176LM2": PriceRef("XRP-USD", "XRP (spot)", _IDX, "spot, not the listed future"),
    "177LM3": PriceRef("SOL-USD", "Solana (spot)", _IDX, "spot, not the listed future"),
    "177741": PriceRef("SOL-USD", "Solana (spot)", _IDX, "spot, not the CME future"),
    "177LM1": PriceRef("SOL-USD", "Solana (spot)", _IDX, "spot, not the listed future"),
    "168DC1": PriceRef(
        "DOGE-USD", "Dogecoin (spot)", _IDX,
        "spot, not the Coinbase Derivatives future; the coin series begins "
        "2017-11, well before positioning in it was reported",
    ),
}

#: The distinct symbols to ingest. Several CFTC codes share one series, so this is
#: shorter than MAP.
SYMBOLS: tuple[str, ...] = tuple(sorted({r.symbol for r in MAP.values()}))


def for_code(code: str) -> PriceRef | None:
    """The price series for a CFTC market code, or None if none is mapped."""
    return MAP.get(str(code).strip())
