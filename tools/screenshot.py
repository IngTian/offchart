#!/usr/bin/env python3
"""Regenerate the README's board screenshots from a running Grafana.

    .venv/bin/python tools/screenshot.py          # all shots
    .venv/bin/python tools/screenshot.py cftc     # just one

Needs `make grafana` up and a backfilled store. Drives headless Chrome rather than
Grafana's image-renderer plugin, which is a 400 MB sidecar container for something
Chrome already does -- and Chrome is already on the machine of anyone who is looking
at a Grafana board.

WHAT IS AND IS NOT CAPTURED, AND WHY IT IS A LICENCE QUESTION

Only CFTC boards. A committed PNG of a board is a published RENDERING of that board's
data, which puts it in the same category as committing the data itself -- and this
repo commits none. CFTC Commitments of Traders is US Government work in the public
domain, so a picture of it is free to publish. The other two boards are not:

  ProShares  "No permission is granted to sell, copy, publish, reproduce, distribute,
             modify, post, 'mirror' on any server ... any of the contents"
  SSGA       "You may not ... disseminate or otherwise transfer, in whole or in part,
             any of the information contained in this Site, except for the purpose of
             accessing the Site and producing print-outs for your own personal use"
  UMich      grants use, prohibits redistribution without written consent -- which is
             already why tests/fixtures' UMich slice is fabricated rather than sampled

A PNG on a public repo is neither a personal print-out nor non-redistribution. So the
flows and inflation boards are described in prose and not photographed, for the same
reason FINRA's margin debt is absent rather than fetched.

NOT BYTE-REPRODUCIBLE, UNLIKE THE REST OF THIS REPO

The store is gitignored and grows weekly, so two runs a week apart differ and a fresh
clone cannot reproduce these files at all. That is a real step down from
`make readme-table`, which regenerates exactly. The mitigation is that the time range
is PINNED to absolute epochs below rather than `now-5y`, so the window does not slide
under the image -- only the newest few weeks of data change. Re-run after anything
that moves a panel, and treat the result as a dated snapshot.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
BASE = "http://localhost:3000"

#: Absolute epoch ms, not `now-5y`. A relative range would make every regeneration
#: shift the x-axis, so a diff could never distinguish "the UI changed" from "time
#: passed". 2006-06-13 (the first TFF report) .. 2026-11-08.
FROM, TO = 1150156800000, 1794096000000

#: E-mini S&P 500 in the financial-futures family: 1,065 weekly reports from the first
#: TFF report, ONE segment, and 4.5M contracts of open interest at peak. The segment
#: count is why it rather than NASDAQ-100 Consolidated -- that code carries the
#: 2023-05-02 contract re-basing, and a 5.17x step in the middle of a hero image reads
#: as a broken chart rather than as the unit change it is. Picked by query, not taste:
#: segments = 1 AND weeks > 900, ordered by open interest.
MARKET = "var-dataset=cftc_tff_fut&var-market=13874A&var-group=asset_mgr%2Blev_money"

#: Device pixels per CSS pixel. Retina, because GitHub downscales and a 1x capture
#: of a chart with 1,065 weekly points loses the line.
SCALE = 2

#: name -> (query, viewport, panel, keep). Viewport is CSS pixels, chosen WIDE AND
#: SHORT because GitHub renders README images at about 870 px and a nine-panel board
#: scaled to that is illegible.
#:
#: `keep` is the number of DEVICE pixels of height to retain, measured from the top.
#: It exists because both routes leave dead background below the content: a kiosk
#: dashboard ends on whatever panel boundary the viewport lands on, and a `d-solo`
#: panel renders at about 76% of the viewport with the remainder as page background.
#: Neither is croppable with `sips`, which ignores --cropOffset and always crops
#: CENTRED -- verified, and it silently eats the panel title. So ffmpeg does it.
#:
#: These are MEASURED, not derived. Re-measure after anything that moves the layout:
#: run with --no-crop, open the PNG, and read off where the content ends.
SHOTS = {
    # The hero: headline tile, every cohort for the week, and twenty years of history.
    # Height lands exactly on the boundary below "Net position as % of open interest".
    "board": (MARKET, (2100, 1078), None, None),
    # The distinction a net alone cannot make, which is why panel 8 exists. Panel 9
    # (directional purity) is deliberately NOT shot: on this market it is a thin band
    # near zero, and the hero's table already carries the purity column.
    "both-sides": (MARKET, (1900, 420), "panel-8", 680),
}
UID = "offchart-cftc"


def _chrome(url: str, out: Path, width: int, height: int) -> None:
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [
                str(CHROME),
                "--headless=new",
                f"--user-data-dir={profile}",  # never touch the real profile
                "--hide-scrollbars",
                f"--force-device-scale-factor={SCALE}",
                f"--window-size={width},{height}",
                # Grafana is a SPA: the panels fetch after first paint, so a plain
                # screenshot catches empty frames. This waits for the fetches.
                "--virtual-time-budget=25000",
                f"--screenshot={out}",
                url,
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )


def _crop_top(path: Path, width_px: int, keep_px: int) -> bool:
    """Keep the top `keep_px` rows. Returns False if ffmpeg is unavailable."""
    if not shutil.which("ffmpeg"):
        return False
    tmp = path.with_suffix(".crop.png")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-vf", f"crop={width_px}:{keep_px}:0:0", str(tmp)],
        check=True, capture_output=True, timeout=120,
    )
    tmp.replace(path)
    return True


def capture(name: str, crop: bool = True) -> Path:
    query, (width, height), panel, keep = SHOTS[name]
    out = ASSETS / f"{name}.png"
    ASSETS.mkdir(exist_ok=True)

    if panel:
        # d-solo renders the panel alone: no nav, no variable bar, no other panels.
        url = (f"{BASE}/d-solo/{UID}/x?{query}&from={FROM}&to={TO}"
               f"&panelId={panel}&theme=dark")
    else:
        url = f"{BASE}/d/{UID}?{query}&from={FROM}&to={TO}&kiosk&theme=dark"

    _chrome(url, out, width, height)
    if crop and keep and not _crop_top(out, width * SCALE, keep):
        print("    ! ffmpeg not found -- left uncropped, expect dead space below")
    return out


def main() -> int:
    if not CHROME.exists():
        print(f"no Chrome at {CHROME}")
        return 1
    args = [a for a in sys.argv[1:] if a != "--no-crop"]
    crop = "--no-crop" not in sys.argv[1:]
    wanted = args or list(SHOTS)
    unknown = [w for w in wanted if w not in SHOTS]
    if unknown:
        print(f"unknown shot(s) {unknown}; known: {list(SHOTS)}")
        return 1
    for name in wanted:
        path = capture(name, crop=crop)
        size = path.stat().st_size
        print(f"  {path.relative_to(ROOT)}  {size / 1024:,.0f} KB")
        if size < 20_000:
            print("    ! suspiciously small -- is Grafana up and the store backfilled?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
