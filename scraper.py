# ============================================================
# Motifiti scraper: daily tracker for GitHub Actions
# Storage: data/cars.parquet (master) + data/checkpoint.parquet (crash recovery)
#          data/state.json (what the site said last run + tracking start)
#
# Each run:
#   1. read page-1 header + all listing pages -> find new cars
#   2. fully scrape new cars (first_seen = today)
#   3. re-check EVERY active car's page for Sold/Removed (last_seen = that day)
#   4. fix make/model/year from titles (titles.py), save parquet + state
#
# Env vars (optional):  MAX_PAGES=2  limit listing pages (testing)
# ============================================================

import os
import re
import json
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

from titles import fix_titles          # self-learning title fixer (titles.py, same folder)

BASE_URL = "https://motifiti.co.ke"

DATA_DIR = "data"
MASTER = f"{DATA_DIR}/cars.parquet"
CHECKPOINT = f"{DATA_DIR}/checkpoint.parquet"
STATE = f"{DATA_DIR}/state.json"

MAX_PAGES = int(os.getenv("MAX_PAGES") or 0) or None     # empty/unset -> all pages

MULTI_WORD_MAKES = [
    "Land Rover", "Alfa Romeo", "Aston Martin", "Rolls Royce", "Great Wall",
    "Mercedes Benz", "Mercedes-Benz",
]

# Statuses that mean "this car is gone": last_seen gets stamped, checks stop
FINAL_STATUSES = {"sold", "removed"}


def now_nairobi() -> datetime:
    return datetime.now(ZoneInfo("Africa/Nairobi"))


def today_str() -> str:
    return now_nairobi().date().isoformat()


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


def parse_header(html: str):
    """'3399 cars listed · page 1 of 142' -> (3399, 142). Either may be None."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    total = last = None

    m = re.search(r"([\d,]+)\s+cars?\s+listed", text, re.I)
    if m:
        total = int(m.group(1).replace(",", ""))

    m = re.search(r"page\s+\d+\s+of\s+(\d+)", text, re.I)
    if m:
        last = int(m.group(1))
    else:                                        # fallback: highest ?page=N link
        nums = [int(x.group(1)) for a in soup.select("a[href*='page=']")
                if (x := re.search(r"[?&]page=(\d+)", a.get("href", "")))]
        last = max(nums) if nums else None

    return total, last


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


def parse_availability(soup) -> str:
    """The banner only appears when the car is not available (e.g. 'Sold')."""
    avail = soup.select_one(".vd-availability")
    if not avail:
        return "Available"
    strong = avail.select_one("strong")
    return _clean(strong.get_text() if strong else avail.get_text()) or "Available"


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

    rec.availability = parse_availability(soup)

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
SAVE_EVERY = 250         # write checkpoint.parquet every N fetched pages

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


async def read_header(session, sem):
    """Page-1 header -> (total_listed, last_page); (None, None) if page 1 can't be read."""
    html = await fetch(session, sem, LISTING_PAGE_URL.format(page=1))
    if not isinstance(html, str):
        return None, None
    return parse_header(html)


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


async def listing_page_urls(session, sem, page):
    html = await fetch(session, sem, LISTING_PAGE_URL.format(page=page))
    if not isinstance(html, str):
        return page, None
    return page, [r.url for r in parse_listing_page(html)]


async def detail_record(session, sem, url):
    """Full scrape. Returns (url, DetailRecord) | (url, GONE) | (url, None on failure)."""
    html = await fetch(session, sem, url)
    if html is GONE:
        return url, GONE
    if html is None:
        return url, None
    return url, parse_detail_page(html, url=url)


async def status_record(session, sem, url):
    """Status check only. Returns (url, availability str) | (url, GONE) | (url, None on failure)."""
    html = await fetch(session, sem, url)
    if html is GONE:
        return url, GONE
    if html is None:
        return url, None
    return url, parse_availability(BeautifulSoup(html, "lxml"))


# ============================================================
# State (what the site said last run + when tracking began)
# ============================================================

def load_state() -> dict:
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE)


# ============================================================
# Parquet storage (DuckDB reads/writes, pandas holds the typed frame)
# ============================================================

FIELDNAMES = [
    "ref", "url", "title", "year", "make", "model", "make_source",
    "listing_type", "listed_by", "mini_site_url",
    "mileage_km", "engine_cc", "fuel_type", "transmission",
    "condition", "trade_in_accepted", "asset_financing_available",
    "price_kes", "negotiable", "availability",
    "first_seen", "last_seen", "last_checked", "baseline",   # dates are YYYY-MM-DD strings
    "spec_performance", "spec_interior_comfort", "spec_safety",
    "town_kml", "highway_kml", "combined_kml", "fuel_consumption_kml",
    "insurance_kes_per_year",
]
INT_COLS = ["year", "mileage_km", "engine_cc", "price_kes", "insurance_kes_per_year"]
FLOAT_COLS = ["town_kml", "highway_kml", "combined_kml"]
BOOL_COLS = ["trade_in_accepted", "asset_financing_available", "negotiable", "baseline"]
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


def finalize(master: dict):
    """Fix titles over the whole master, write it, then drop the checkpoint."""
    if not master:
        return
    df = to_frame(list(master.values()))
    try:
        fixed = fix_titles(df)
        df = to_frame(fixed.astype(object).where(fixed.notna(), None).to_dict("records"))
    except Exception as e:
        print(f"Title fixing failed ({e!r}); saving titles as scraped")
    write_parquet(df, MASTER)
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)


# ============================================================
# Main run
# ============================================================

async def run_async(max_pages=MAX_PAGES):
    today = today_str()
    os.makedirs(DATA_DIR, exist_ok=True)
    limited = max_pages is not None                      # test run: skip the count comparison

    state = load_state()
    master = {r["url"]: r for r in load_rows(MASTER)}
    changes = {r["url"]: r for r in load_rows(CHECKPOINT)}   # leftover from a crashed run
    if changes:
        print(f"Recovered {len(changes)} rows from {CHECKPOINT}")
        master.update(changes)
    for r in master.values():
        if r.get("baseline") is None:        # rows from before this column existed
            r["baseline"] = True

    # Date counting began. Cars first seen on this date have an unknown real listing date ("baseline").
    if not state.get("tracking_start"):
        firsts = [r["first_seen"] for r in master.values() if r.get("first_seen")]
        state["tracking_start"] = min(firsts) if firsts else today
    is_baseline_day = (today == state["tracking_start"])
    print(f"Today: {today} | {len(master)} cars in {MASTER} | tracking since {state['tracking_start']}"
          + (" | baseline day" if is_baseline_day else ""))

    failed_urls = []
    n_new = n_sold = n_removed = 0
    ok = False

    def save_checkpoint(bar, done, total):
        if done % SAVE_EVERY == 0 and changes:
            write_parquet(to_frame(list(changes.values())), CHECKPOINT)
            bar.write(f"Checkpoint saved: {done}/{total} ({len(changes)} rows changed)")

    try:
        sem = asyncio.Semaphore(CONCURRENCY)
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=CONCURRENCY)

        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout, connector=connector) as session:

            # ---- Phase 0: read the site header, compare with last run ----
            total_listed, header_last = await read_header(session, sem)
            prev_total, prev_last = state.get("last_total_listed"), state.get("last_page")
            if total_listed is not None:
                print(f"Site says: {total_listed} cars listed, {header_last} pages"
                      + (f" (last run: {prev_total} cars, {prev_last} pages)" if prev_total else ""))
            else:
                print("Could not read the page-1 header")

            if max_pages is None:
                max_pages = header_last
                if max_pages is None and prev_last:
                    max_pages = prev_last
                    print(f"Using last run's page count: {max_pages}")
                if max_pages is None:
                    print("No pagination info, probing for the last page...")
                    max_pages = await probe_last_page(session, sem)
            print(f"Listing pages to read: {max_pages}")

            # ---- Phase 1: listing pages ----
            page_tasks = [asyncio.create_task(listing_page_urls(session, sem, p))
                          for p in range(1, max_pages + 1)]
            results = []
            bar = tqdm(total=len(page_tasks), desc="Listing pages", unit="page", mininterval=10)
            for coro in asyncio.as_completed(page_tasks):
                results.append(await coro)
                bar.update(1)
            bar.close()

            live, seen, failed_pages, empty_pages = [], set(), [], []
            for page, page_urls in sorted(results, key=lambda r: r[0]):
                if page_urls is None:
                    failed_pages.append(page)
                    continue
                if not page_urls:
                    empty_pages.append(page)
                for u in page_urls:
                    if u not in seen:
                        seen.add(u)
                        live.append(u)

            if not live:
                raise RuntimeError("No listings found on any page; aborting.")

            # Remember what the site said (real header values, safe even on test runs)
            if total_listed is not None:
                state["last_total_listed"] = total_listed
            if header_last is not None:
                state["last_page"] = header_last
            state["last_live_found"] = len(live)

            # Sanity check: did we actually get (nearly) everything the site claims?
            if total_listed and not limited:
                ratio = len(live) / total_listed
                if ratio < 0.95:
                    print(f"WARNING: found {len(live)} cars but site claims {total_listed} "
                          f"({ratio:.0%}). Some listing pages may have failed or shifted.")

            # ---- Decide what to fetch ----
            new_urls = [u for u in live if u not in master]
            relisted = [u for u in live if u in master
                        and (master[u]["availability"] or "").lower() == "removed"]
            full_jobs = new_urls + relisted
            full_set = set(full_jobs)
            # every active car gets a status check (skip ones already checked today, e.g. after a crash)
            check_jobs = [u for u, r in master.items()
                          if not is_final(r["availability"])
                          and r.get("last_checked") != today
                          and u not in full_set]

            print(f"{len(live)} on listing pages | {len(new_urls)} new | {len(relisted)} relisted | "
                  f"{len(check_jobs)} to status-check")
            if failed_pages:
                print(f"  {len(failed_pages)} listing pages failed: {failed_pages}")
            if empty_pages:
                print(f"  {len(empty_pages)} listing pages had 0 cards: {empty_pages[:10]}")

            # ---- Phase 2: full scrape of new / relisted cars ----
            tasks = [asyncio.create_task(detail_record(session, sem, u)) for u in full_jobs]
            bar = tqdm(total=len(tasks), desc="New cars", unit="car", mininterval=10)
            done = 0
            for coro in asyncio.as_completed(tasks):
                url, rec = await coro
                old = master.get(url)

                if rec is None:                                   # network failure: retry next run
                    failed_urls.append(url)
                elif rec is GONE:
                    if old is None:
                        failed_urls.append(url)
                else:
                    row = record_to_row(rec)
                    final = is_final(row["availability"])
                    row["last_checked"] = today
                    row["last_seen"] = today if final else None
                    if old is None:
                        row["first_seen"] = today
                        row["baseline"] = is_baseline_day
                        n_new += 1
                    else:                                         # relisted: keep original history
                        row["first_seen"] = old.get("first_seen") or today
                        row["baseline"] = old.get("baseline")
                    if final:
                        n_sold += 1
                    master[url] = changes[url] = row

                done += 1
                save_checkpoint(bar, done, len(tasks))
                bar.set_postfix(failed=len(failed_urls), sold=n_sold)
                bar.update(1)
            bar.close()

            # ---- Phase 3: sold-status check on every active car ----
            tasks = [asyncio.create_task(status_record(session, sem, u)) for u in check_jobs]
            bar = tqdm(total=len(tasks), desc="Status checks", unit="car", mininterval=10)
            done = 0
            for coro in asyncio.as_completed(tasks):
                url, status = await coro
                old = master[url]

                if status is None:                                # network failure: retry next run
                    failed_urls.append(url)
                else:
                    row = dict(old)
                    if status is GONE:                            # 404: listing taken down
                        row.update(availability="Removed", last_seen=today, last_checked=today)
                        n_removed += 1
                    else:
                        final = is_final(status)
                        row.update(availability=status, last_checked=today,
                                   last_seen=today if final else None)
                        if final:
                            n_sold += 1
                    master[url] = changes[url] = row

                done += 1
                save_checkpoint(bar, done, len(tasks))
                bar.set_postfix(failed=len(failed_urls), sold=n_sold, removed=n_removed)
                bar.update(1)
            bar.close()
            ok = True

    finally:
        finalize(master)
        state["last_run_date"] = today
        state["last_run_at"] = now_nairobi().isoformat(timespec="seconds")
        state["last_run_ok"] = ok
        save_state(state)

    print(f"Done -> {MASTER}  ({len(master)} total | {n_new} new | "
          f"{n_sold} sold | {n_removed} removed | {len(failed_urls)} failed)")
    return failed_urls


if __name__ == "__main__":
    asyncio.run(run_async())
