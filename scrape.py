# ============================================================
# Motifiti scraper: incremental tracker for GitHub Actions
# Storage: data/cars.parquet (master) + data/checkpoint.parquet (crash recovery)
#
# Env vars (all optional):
#   MAX_PAGES=2       limit listing pages (testing)
#   RECHECK_DAYS=3    re-fetch listed cars not checked for N days (0 = all, every run)
# ============================================================

import os
import re
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import asyncio
import aiohttp
import duckdb
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://motifiti.co.ke"

DATA_DIR = "data"
MASTER = f"{DATA_DIR}/cars.parquet"
CHECKPOINT = f"{DATA_DIR}/checkpoint.parquet"

MAX_PAGES = int(os.getenv("MAX_PAGES", "0")) or None
RECHECK_DAYS = int(os.getenv("RECHECK_DAYS", "3"))

MULTI_WORD_MAKES = [
    "Land Rover", "Alfa Romeo", "Aston Martin", "Rolls Royce", "Great Wall",
    "Mercedes Benz", "Mercedes-Benz",
]

# Statuses that mean "this car is gone": last_seen gets stamped, rechecks stop
FINAL_STATUSES = {"sold", "removed"}


def today_str() -> str:
    return datetime.now(ZoneInfo("Africa/Nairobi")).date().isoformat()


def is_final(status) -> bool:
    return (status or "").strip().lower() in FINAL_STATUSES


# ============================================================
# Parsing helpers
# ============================================================

def parse_title(title: str):
    title = title.strip()
    m = re.match(r"^(\d{4})\s+(.*)$", title)
    if m:
        year = int(m.group(1))
        rest = m.group(2).strip()
    else:
        year = None
        rest = title
    if not rest:
        return year, None, None
    for mw in MULTI_WORD_MAKES:
        if rest.lower().startswith(mw.lower() + " ") or rest.lower() == mw.lower():
            make = rest[:len(mw)]
            model = rest[len(mw):].strip() or None
            return year, make, model
    tokens = rest.split(" ", 1)
    make = tokens[0]
    model = tokens[1].strip() if len(tokens) > 1 else None
    return year, make, model


def _num(text):
    if text is None:
        return None
    m = re.search(r"[\d,]+(?:\.\d+)?", text)
    if not m:
        return None
    return float(m.group(0).replace(",", "")) if "." in m.group(0) else int(m.group(0).replace(",", ""))


def _clean(text):
    return re.sub(r"\s+", " ", text or "").strip() or None


# ============================================================
# Listing page parsing
# ============================================================

@dataclass
class ListingRow:
    url: str
    ref_slug: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    make: Optional[str] = None
    model: Optional[str] = None
    mileage_km: Optional[int] = None
    engine_cc: Optional[int] = None
    fuel_type: Optional[str] = None
    transmission: Optional[str] = None
    condition: Optional[str] = None
    price_kes: Optional[int] = None
    listing_type: Optional[str] = None
    listed_by: Optional[str] = None


def parse_listing_card(card) -> ListingRow:
    href = card.get("href", "")
    row = ListingRow(url=BASE_URL + href if href.startswith("/") else href)

    m = re.search(r"/vehicle/([^/]+)/([^/?#]+)", href)
    if m:
        row.ref_slug = f"{m.group(1)}/{m.group(2)}"

    heading = card.select_one(".public-listing-heading")
    row.title = _clean(heading.get_text()) if heading else None
    row.year, row.make, row.model = parse_title(row.title) if row.title else (None, None, None)

    icon_items = [_clean(s.get_text()) for s in card.select(".public-listing-icon-item")]
    for val in icon_items:
        if val is None:
            continue
        if "km" in val.lower():
            row.mileage_km = _num(val)
        elif "cc" in val.lower():
            row.engine_cc = _num(val)
        elif val.lower() in ("petrol", "diesel", "hybrid", "electric"):
            row.fuel_type = val
        elif val.lower() in ("auto", "manual"):
            row.transmission = val

    cond = card.select_one(".public-listing-condition")
    row.condition = _clean(cond.get_text()) if cond else None

    price = card.select_one(".public-listing-price")
    row.price_kes = _num(price.get_text()) if price else None

    dealer_logo = card.select_one("img.public-listing-dealer-logo")
    private_label = card.select_one(".public-listing-user-label")
    if dealer_logo is not None:
        row.listing_type = "Dealer"
        row.listed_by = _clean(dealer_logo.get("alt"))
    elif private_label is not None:
        row.listing_type = "Private"
        row.listed_by = "Private"
    else:
        row.listing_type = None
        row.listed_by = None

    return row


def parse_listing_page(html: str):
    soup = BeautifulSoup(html, "lxml")
    cards = soup.select("a.public-listing-row")
    return [parse_listing_card(c) for c in cards]


# ============================================================
# Detail page parsing
# ============================================================

@dataclass
class DetailRecord:
    ref: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    make: Optional[str] = None
    model: Optional[str] = None
    listing_type: Optional[str] = None
    listed_by: Optional[str] = None
    mini_site_url: Optional[str] = None
    mileage_km: Optional[int] = None
    engine_cc: Optional[int] = None
    fuel_type: Optional[str] = None
    transmission: Optional[str] = None
    condition: Optional[str] = None
    trade_in_accepted: bool = False
    asset_financing_available: bool = False
    price_kes: Optional[int] = None
    negotiable: bool = False
    availability: str = "Available"
    specs: dict = field(default_factory=dict)
    town_kml: Optional[float] = None
    highway_kml: Optional[float] = None
    combined_kml: Optional[float] = None
    fuel_consumption_kml: Optional[str] = None
    insurance_kes_per_year: Optional[int] = None


def _running_cost_value(grid, label):
    for item in grid.select(".vd-running-item"):
        lab = item.select_one(".vd-running-label")
        val = item.select_one(".vd-running-value")
        if lab and _clean(lab.get_text()) == label:
            return _num(val.get_text()) if val else None
    return None


def parse_detail_page(html: str, url: Optional[str] = None) -> DetailRecord:
    soup = BeautifulSoup(html, "lxml")
    rec = DetailRecord(url=url)

    ref_span = soup.select_one(".vd-stock-id")
    if ref_span:
        rec.ref = _clean(ref_span.get_text()).replace("Ref:", "").strip()

    h1 = soup.select_one("h1")
    rec.title = _clean(h1.get_text()) if h1 else None
    rec.year, rec.make, rec.model = parse_title(rec.title) if rec.title else (None, None, None)

    for s in soup.select(".vd-icon-row .vd-icon-item"):
        val = _clean(s.get_text())
        if val is None:
            continue
        if "km" in val.lower():
            rec.mileage_km = _num(val)
        elif "cc" in val.lower():
            rec.engine_cc = _num(val)
        elif val.lower() in ("petrol", "diesel", "hybrid", "electric"):
            rec.fuel_type = val
        elif val.lower() in ("auto", "manual"):
            rec.transmission = val

    dealer_row = soup.select_one(".mv-dealer-row")
    if dealer_row:
        link = dealer_row.select_one("a")
        if link is not None:
            rec.listing_type = "Dealer"
            rec.listed_by = _clean(link.get_text())
            rec.mini_site_url = link.get("href")
        else:
            span = dealer_row.select_one("span")
            text = _clean(span.get_text()) if span else _clean(dealer_row.get_text())
            rec.listing_type = "Private"
            rec.listed_by = re.sub(r"^Listed by\s*", "", text or "").strip() or "Private"

    badge_row = soup.select_one(".badge-row")
    if badge_row:
        badges = [_clean(b.get_text()) for b in badge_row.select(".badge")]
        for b in badges:
            if b is None:
                continue
            bl = b.lower()
            if bl == "trade-in accepted":
                rec.trade_in_accepted = True
            elif bl == "asset financing available":
                rec.asset_financing_available = True
            elif b not in ("Negotiable",):
                rec.condition = b

    price_div = soup.select_one(".vd-price")
    if price_div:
        rec.price_kes = _num(price_div.get_text())
        rec.negotiable = price_div.select_one(".badge.negotiable") is not None

    # Availability: banner only appears when the car is not available (e.g. "Sold")
    avail = soup.select_one(".vd-availability")
    if avail:
        strong = avail.select_one("strong")
        rec.availability = _clean(strong.get_text() if strong else avail.get_text()) or "Available"
    else:
        rec.availability = "Available"

    specs_section = None
    for det in soup.select("details.vd-more"):
        summ = det.select_one("summary")
        if summ and _clean(summ.get_text()) == "Specs":
            specs_section = det
            break
    if specs_section:
        for p in specs_section.select("p.field"):
            label_el = p.select_one(".label")
            if not label_el:
                continue
            label = _clean(label_el.get_text())
            value = _clean(p.get_text().replace(label_el.get_text(), "", 1))
            if label:
                rec.specs[label] = value

    running_section = None
    for det in soup.select("details.vd-more"):
        summ = det.select_one("summary")
        if summ and _clean(summ.get_text()) == "Running costs (estimated)":
            running_section = det
            break
    if running_section:
        grid = running_section.select_one(".vd-running-grid")
        if grid:
            rec.town_kml = _running_cost_value(grid, "Town driving")
            rec.highway_kml = _running_cost_value(grid, "Highway")
            rec.combined_kml = _running_cost_value(grid, "Combined")

        if rec.combined_kml is not None:
            rec.fuel_consumption_kml = str(rec.combined_kml)
        elif rec.town_kml is not None and rec.highway_kml is not None:
            rec.fuel_consumption_kml = f"{rec.town_kml}-{rec.highway_kml}"
        elif rec.town_kml is not None:
            rec.fuel_consumption_kml = str(rec.town_kml)
        elif rec.highway_kml is not None:
            rec.fuel_consumption_kml = str(rec.highway_kml)
        else:
            rec.fuel_consumption_kml = None

        ins_val = running_section.select_one(".vd-running-insurance .vd-running-value")
        if ins_val:
            rec.insurance_kes_per_year = _num(ins_val.get_text())

    return rec


# ============================================================
# Fetching
# ============================================================

LISTING_PAGE_URL = BASE_URL + "/?page={page}"
CONCURRENCY = 8
DELAY_SECONDS = 0.25
RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"}
SAVE_EVERY = 250         # write checkpoint.parquet every N detail pages

GONE = object()          # sentinel: page returned 404/410 (listing taken down)


async def fetch(session, sem, url):
    """HTML string, GONE for 404/410, or None if it fails (retries 429/5xx/timeouts)."""
    for attempt in range(RETRIES):
        try:
            async with sem:
                async with session.get(url) as resp:
                    if resp.status in RETRY_STATUSES:
                        raise aiohttp.ClientError(f"HTTP {resp.status}")
                    if resp.status in (404, 410):
                        return GONE
                    if resp.status >= 400:
                        return None
                    text = await resp.text()
                await asyncio.sleep(DELAY_SECONDS)
                return text
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(2 ** attempt)
    return None


async def detect_last_page(session, sem):
    html = await fetch(session, sem, LISTING_PAGE_URL.format(page=1))
    if not isinstance(html, str):
        raise RuntimeError("Could not fetch listing page 1")
    soup = BeautifulSoup(html, "lxml")
    candidates = []

    m = re.search(r"page\s+\d+\s+of\s+(\d+)", soup.get_text(" ", strip=True), re.I)
    if m:
        candidates.append(int(m.group(1)))

    for a in soup.select("a[href*='page=']"):
        m = re.search(r"[?&]page=(\d+)", a.get("href", ""))
        if m:
            candidates.append(int(m.group(1)))

    return max(candidates) if candidates else None


async def probe_last_page(session, sem, cap=1000):
    async def has_cards(page):
        html = await fetch(session, sem, LISTING_PAGE_URL.format(page=page))
        return isinstance(html, str) and bool(parse_listing_page(html))

    lo, hi = 1, 2
    while hi <= cap and await has_cards(hi):
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if await has_cards(mid):
            lo = mid
        else:
            hi = mid
    return lo


async def listing_page_rows(session, sem, page):
    html = await fetch(session, sem, LISTING_PAGE_URL.format(page=page))
    if not isinstance(html, str):
        return page, None
    return page, parse_listing_page(html)


async def detail_record(session, sem, url):
    """Returns (url, DetailRecord) | (url, GONE) | (url, None on failure)."""
    html = await fetch(session, sem, url)
    if html is GONE:
        return url, GONE
    if html is None:
        return url, None
    return url, parse_detail_page(html, url=url)


# ============================================================
# Parquet storage (DuckDB reads/writes, pandas holds the typed frame)
# ============================================================

FIELDNAMES = [
    "ref", "url", "title", "year", "make", "model",
    "listing_type", "listed_by", "mini_site_url",
    "mileage_km", "engine_cc", "fuel_type", "transmission",
    "condition", "trade_in_accepted", "asset_financing_available",
    "price_kes", "negotiable", "availability",
    "first_seen", "last_seen", "last_checked",     # dates stored as YYYY-MM-DD strings
    "spec_performance", "spec_interior_comfort", "spec_safety",
    "town_kml", "highway_kml", "combined_kml", "fuel_consumption_kml",
    "insurance_kes_per_year",
]
INT_COLS = ["year", "mileage_km", "engine_cc", "price_kes", "insurance_kes_per_year"]
FLOAT_COLS = ["town_kml", "highway_kml", "combined_kml"]
BOOL_COLS = ["trade_in_accepted", "asset_financing_available", "negotiable"]
STR_COLS = [c for c in FIELDNAMES if c not in INT_COLS + FLOAT_COLS + BOOL_COLS]


def record_to_row(rec) -> dict:
    d = dataclasses.asdict(rec)
    specs = d.pop("specs", {})
    d["spec_performance"] = specs.get("Performance")
    d["spec_interior_comfort"] = specs.get("Interior & Comfort")
    d["spec_safety"] = specs.get("Safety")
    return {k: d.get(k) for k in FIELDNAMES}


def to_frame(rows) -> pd.DataFrame:
    """List of row dicts -> DataFrame with stable dtypes (so parquet types never drift)."""
    df = pd.DataFrame(rows, columns=FIELDNAMES)
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64").round().astype("Int64")
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in BOOL_COLS:
        df[c] = df[c].astype("boolean")
    for c in STR_COLS:
        df[c] = df[c].astype("string")
    return df


def load_rows(path) -> list:
    """Parquet -> list of dicts with None for missing values."""
    if not os.path.exists(path):
        return []
    df = duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()
    df = df.astype(object).where(df.notna(), None)
    return df.to_dict("records")


def write_parquet(df: pd.DataFrame, path: str):
    """Atomic write: a crash mid-save never corrupts the file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    con = duckdb.connect()
    con.register("df", df)
    con.execute(f"COPY (SELECT * FROM df ORDER BY make, model, year) "
                f"TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()
    os.replace(tmp, path)


def pick_jobs(master_df: pd.DataFrame, live_df: pd.DataFrame, today: str) -> pd.DataFrame:
    """Decide which detail pages to fetch, in SQL. Returns (reason, url) rows."""
    con = duckdb.connect()
    con.register("master", master_df)
    con.register("live", live_df)
    final = ", ".join(f"'{s}'" for s in sorted(FINAL_STATUSES))
    sql = f"""
    WITH active AS (SELECT * FROM master WHERE lower(availability) NOT IN ({final}))
    -- not seen before
    SELECT 'new' AS reason, l.url FROM live l
      WHERE l.url NOT IN (SELECT url FROM master)
    UNION ALL
    -- was active, no longer on the listing pages: probably sold/removed
    SELECT 'vanished', a.url FROM active a
      WHERE a.url NOT IN (SELECT url FROM live)
    UNION ALL
    -- listed price differs from what we stored
    SELECT 'price_changed', a.url FROM active a JOIN live l ON l.url = a.url
      WHERE l.price_kes IS NOT NULL AND a.price_kes IS DISTINCT FROM l.price_kes
    UNION ALL
    -- safety net: still listed but not checked for RECHECK_DAYS days
    SELECT 'stale', a.url FROM active a JOIN live l ON l.url = a.url
      WHERE a.last_checked IS NULL
         OR date_diff('day', CAST(a.last_checked AS DATE), DATE '{today}') >= {RECHECK_DAYS}
    UNION ALL
    -- marked Removed (maybe a transient 404) but back on the listing pages
    SELECT 'relisted', m.url FROM master m JOIN live l ON l.url = m.url
      WHERE lower(m.availability) = 'removed'
    """
    out = con.sql(sql).df()
    con.close()
    return out


# ============================================================
# Main run
# ============================================================

async def run_async(max_pages=MAX_PAGES):
    today = today_str()
    os.makedirs(DATA_DIR, exist_ok=True)

    master = {r["url"]: r for r in load_rows(MASTER)}
    changes = {r["url"]: r for r in load_rows(CHECKPOINT)}   # leftover from a crashed run
    if changes:
        print(f"Recovered {len(changes)} rows from {CHECKPOINT}")
        master.update(changes)
    print(f"Today: {today} | {len(master)} cars in {MASTER}")

    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)

    async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout, connector=connector) as session:

        # ---- Phase 0: how many pages? ----
        if max_pages is None:
            max_pages = await detect_last_page(session, sem)
            if max_pages is None:
                print("No pagination info on page 1, probing for the last page...")
                max_pages = await probe_last_page(session, sem)
        print(f"Listing pages to read: {max_pages}")

        # ---- Phase 1: listing pages ----
        page_tasks = [asyncio.create_task(listing_page_rows(session, sem, p))
                      for p in range(1, max_pages + 1)]
        results = []
        bar = tqdm(total=len(page_tasks), desc="Listing pages", unit="page", mininterval=10)
        for coro in asyncio.as_completed(page_tasks):
            results.append(await coro)
            bar.update(1)
        bar.close()

        live_price, failed_pages, empty_pages = {}, [], []
        for page, rows in sorted(results, key=lambda r: r[0]):
            if rows is None:
                failed_pages.append(page)
                continue
            if not rows:
                empty_pages.append(page)
            for r in rows:
                live_price.setdefault(r.url, r.price_kes)

        if not live_price:
            raise RuntimeError("No listings found on any page; aborting so nothing is mis-marked.")

        live_df = pd.DataFrame({
            "url": list(live_price),
            "price_kes": pd.array(list(live_price.values()), dtype="Int64"),
        })

        # ---- Decide what to fetch (DuckDB SQL) ----
        jobs_all = pick_jobs(to_frame(list(master.values())), live_df, today)
        reasons = jobs_all.groupby("reason").size().to_dict()
        jobs = jobs_all.drop_duplicates("url")["url"].tolist()

        print(f"{len(live_price)} on listing pages | {len(master)} in master | "
              f"{len(jobs)} detail fetches {reasons}")
        if failed_pages:
            print(f"  {len(failed_pages)} listing pages failed: {failed_pages}")
        if empty_pages:
            print(f"  {len(empty_pages)} listing pages had 0 cards: {empty_pages[:10]}")

        # ---- Phase 2: detail pages ----
        failed_urls = []
        n_new = n_sold = n_removed = 0
        tasks = [asyncio.create_task(detail_record(session, sem, u)) for u in jobs]
        bar = tqdm(total=len(tasks), desc="Detail pages", unit="car", mininterval=10)
        done = 0
        try:
            for coro in asyncio.as_completed(tasks):
                url, rec = await coro
                old = master.get(url)

                if rec is None:                                   # network failure: retry next run
                    failed_urls.append(url)

                elif rec is GONE:                                 # 404: listing taken down
                    if old is None:
                        failed_urls.append(url)
                    elif not is_final(old["availability"]):
                        row = dict(old)
                        row.update(availability="Removed", last_seen=today, last_checked=today)
                        master[url] = changes[url] = row
                        n_removed += 1

                else:
                    row = record_to_row(rec)
                    final = is_final(row["availability"])
                    row["last_checked"] = today
                    row["last_seen"] = today if final else None
                    if old is None:
                        row["first_seen"] = today
                        n_new += 1
                    else:
                        row["first_seen"] = old.get("first_seen") or today
                    if final:
                        n_sold += 1
                    master[url] = changes[url] = row

                done += 1
                if done % SAVE_EVERY == 0 and changes:
                    write_parquet(to_frame(list(changes.values())), CHECKPOINT)
                    bar.write(f"Checkpoint saved: {done}/{len(tasks)} ({len(changes)} rows changed)")
                bar.set_postfix(failed=len(failed_urls), sold=n_sold, removed=n_removed)
                bar.update(1)
        finally:
            bar.close()
            write_parquet(to_frame(list(master.values())), MASTER)
            if os.path.exists(CHECKPOINT):
                os.remove(CHECKPOINT)

    print(f"Done -> {MASTER}  ({len(master)} total | {n_new} new | "
          f"{n_sold} now sold | {n_removed} removed | {len(failed_urls)} failed)")
    return failed_urls


if __name__ == "__main__":
    asyncio.run(run_async())
