# ============================================================
# Self-learning year / make / model extraction  (no hard-coded make list)
#
# Makes are LEARNED from the titles in your own data:
#   * a make sits at the START of titles and almost never in the middle
#   * "Prado", "Range Rover", "C200" are models: they also appear after a make, so they're rejected
#   * titles with no make ("Prado TX-L", "LX600") get it inferred from what the model usually pairs with
#   * typos ("Mercedez Benz", "Peugot") are fuzzy-matched to learned makes
# Optional: pass extra_makes=fetch_nhtsa_makes() to also recognise makes seen only 1-2 times.
#
# Usage after scraping:   cars = fix_titles(cars)
# ============================================================
import re
import difflib
import datetime
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

# Pattern-based (not name-based) clutter that sellers put in titles. Edit freely.
NOISE_PATTERNS = [
    r"\b(kes|ksh?s?)\b.*$",                             # price tails: "KES 2,7..."
    r"\bfor\s+sale(\s+in\s+\w+)?\b", r"\bin\s+kenya\b", r"\bon\s+sale\b",
    r"\b\d[\d.]*\s*(cc|ltr|litres?|l)\b",                # engine sizes: 2000cc, 2.5L
    r"\b\d+\s*km\b",                                     # mileage
    r"\b[24]\s?wd\b", r"\bawd\b", r"\b4x4\b",            # drivetrain
    r"\b(automatic|manual|petrol|diesel)\b",
    r"\bmodel\b", r"\b(clean|fresh|digs|imported?)\b",
]
_NOISE = re.compile("|".join(NOISE_PATTERNS), re.I)
_YEAR = re.compile(r"(?<!\d)((?:19[5-9]\d|20[0-3]\d))\d?(?=[\s/]|$)")


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _ngrams(toks, i, nmax=3):
    return [(n, _norm(" ".join(toks[i:i + n]))) for n in range(1, nmax + 1) if i + n <= len(toks)]


def _digits(s):
    return re.sub(r"\D", "", s)


def fetch_nhtsa_makes(timeout=20):
    """Optional: every vehicle make known to the US NHTSA (needs internet, e.g. Colab)."""
    import requests
    r = requests.get("https://vpic.nhtsa.dot.gov/api/vehicles/GetAllMakes?format=json", timeout=timeout)
    r.raise_for_status()
    return {m["Make_Name"] for m in r.json()["Results"]}


class TitleParser:
    def __init__(self, min_support=3, start_ratio=0.8, strong_support=None,
                 purity=0.9, fuzzy_cutoff=0.9, make_fuzzy_cutoff=0.85, extra_makes=None):
        self.min_support, self.start_ratio = min_support, start_ratio
        self.strong_support, self.purity, self.fuzzy_cutoff = strong_support, purity, fuzzy_cutoff
        self.make_fuzzy_cutoff = make_fuzzy_cutoff
        self.extra = {_norm(m) for m in (extra_makes or [])}

    # ---------- text prep ----------
    def _prep(self, title):
        """-> (year, tokens). Handles fancy unicode, emoji, junk after the year, noise phrases."""
        t = unicodedata.normalize("NFKC", str(title))          # 𝐓𝐎𝐘𝐎𝐓𝐀 -> TOYOTA
        t = re.sub(r"[^\w\s\-./+&'’,:]", " ", t)                # emoji / symbols
        t = t.replace(",", " ").replace("–", " ").strip()
        year = None
        m = _YEAR.match(t) or _YEAR.search(t[:12])
        if m and int(m.group(1)) <= datetime.date.today().year + 1:
            year = int(m.group(1))
            t = t[:m.start()] + " " + t[m.end():]
        t = re.sub(r"^(?:/\S*|[\s\-.:]+)+", "", t)             # leading junk like '/11'
        t = re.sub(r"\s+", " ", _NOISE.sub(" ", t)).strip()
        return year, [x for x in t.split(" ") if x.strip("-./:'’")]

    # ---------- learning ----------
    def fit(self, titles):
        toks_list = [tk for tk in (self._prep(t)[1] for t in titles if isinstance(t, str)) if tk]
        N = len(toks_list)
        self.strong_support = self.strong_support or max(10, int(0.01 * N))

        start, elsewhere = Counter(), Counter()
        next_tok, surface = defaultdict(Counter), defaultdict(Counter)
        for toks in toks_list:
            for i in range(len(toks)):
                for n, key in _ngrams(toks, i):
                    surface[key][" ".join(toks[i:i + n])] += 1
                    if i == 0:
                        start[key] += 1
                        next_tok[key][_norm(toks[n]) if n < len(toks) else ""] += 1
                    else:
                        elsewhere[key] += 1
        ratio = lambda k: start[k] / (start[k] + elsewhere[k])

        # 1. make-like: starts titles, (almost) never appears mid-title
        cands = {k for k in start if k and not k.isdigit() and start[k] >= self.min_support
                 and ratio(k) >= self.start_ratio}
        # 1b. optional external list rescues rare makes (needs a model after it, never seen mid-title)
        for k in start:
            if (k in self.extra and k not in cands and k.isalpha() and len(k) >= 4
                    and elsewhere[k] == 0 and any(next_tok[k])):
                cands.add(k)
        strong = {k for k in cands if start[k] >= self.strong_support and ratio(k) >= 0.9}

        # 2. noise prefixes: what follows is a strong make ("Clean Toyota ...") -> not a make
        follows = Counter()
        for toks in toks_list:
            for n, key in _ngrams(toks, 0):
                if key in cands and key not in strong and any(k2 in strong for _, k2 in _ngrams(toks, n)):
                    follows[key] += 1
        bad = {k for k, c in follows.items() if c / start[k] > 0.5}
        cands -= bad
        strong -= bad

        # 3. a longer candidate that only extends a shorter, equally common one adds nothing
        #    ("Ford Ranger" under "Ford"), but "Mercedes-Benz" beats "Mercedes" (bigger support)
        cands = {b for b in cands
                 if not any(a != b and b.startswith(a) and start[a] >= start[b] for a in cands)}
        strong &= cands

        def subseq(a, b):
            it = iter(b)
            return all(c in it for c in a)

        def build(cands):
            alias = {}
            for a in cands:
                pref = [b for b in cands if b != a and start[b] > start[a] and b.startswith(a) and len(a) >= 4]
                if pref:
                    alias[a] = max(pref, key=lambda b: start[b])
                elif len(a) <= 3 and a not in strong:               # "VW" -> "Volkswagen"
                    subs = [b for b in strong if b[0] == a[0] and len(b) > len(a) and subseq(a, b)]
                    if len(subs) == 1:
                        alias[a] = subs[0]
            self.makes, self.alias = cands, alias
            hint, pre = defaultdict(Counter), defaultdict(Counter)
            for toks in toks_list:
                hit = self._find_make(toks)
                if hit:
                    mk, s_, e_ = hit
                    rest = toks[e_:]
                    for i in range(min(len(rest), 4)):
                        for n, key in _ngrams(rest, i, nmax=2):
                            if key and not key.isdigit() and len(key) > 1:
                                hint[key][mk] += 1
                        m = re.match(r"([a-z]{2,})(?=\d)", _norm(rest[i]))   # GLE300d -> 'gle'
                        if m:
                            pre[m.group(1)][mk] += 1
            def pick(counter, min_n, pur):
                out = {}
                for key, c in counter.items():
                    mk, top = c.most_common(1)[0]
                    if sum(c.values()) >= min_n and top / sum(c.values()) >= pur:
                        out[key] = mk
                return out
            self.hint, self.hint_pre = pick(hint, 2, self.purity), pick(pre, 3, 0.95)

        self.start, self.surface, self.strong, self.elsewhere = start, surface, strong, elsewhere
        build(cands)
        # 4. weak multi-word "makes" whose first word is really a model ("Hilux GR", "ZX V8") are not makes
        def ntok(k):
            return len(surface[k].most_common(1)[0][0].split())
        weak = {k for k in cands if k not in strong and ntok(k) > 1
                and _norm(surface[k].most_common(1)[0][0].split()[0]) in self.hint}
        if weak:
            cands = cands - weak
            strong &= cands
            build(cands)

        def display(k):
            forms = surface[k]
            top = forms.most_common(1)[0][0]
            if top.isupper() and len(_norm(top)) <= 3:           # BMW, VW, MG
                return top
            nonupper = [f for f, _ in forms.most_common() if not f.isupper() and not f.islower()]
            return nonupper[0] if nonupper else top.title()
        self.start, self.surface, self.strong = start, surface, strong
        self.display = {k: display(self.alias.get(k, k)) for k in cands}
        self._hint_keys, self._make_keys = list(self.hint), list(self.makes)
        self._fz = {}
        return self

    # ---------- parsing ----------
    def _find_make(self, toks):
        for i in range(min(len(toks), 4)):                       # earliest position wins
            opts = [(self.start[k], -n, k, n) for n, k in _ngrams(toks, i) if k in self.makes]
            if opts:
                _, _, k, n = max(opts)                           # biggest support; ties -> shortest
                return self.alias.get(k, k), i, i + n
        return None

    def _fuzzy(self, key, pool, min_len, cutoff=None):
        """Close match that keeps digits identical (so 'Discovery 3' never becomes 'Discovery 4')."""
        if len(key) < min_len or sum(c.isalpha() for c in key) < 4:
            return None
        cutoff = cutoff or self.fuzzy_cutoff
        ck = (key, id(pool), cutoff)
        if ck not in self._fz:
            self._fz[ck] = next((m for m in difflib.get_close_matches(key, pool, n=3, cutoff=cutoff)
                                 if _digits(m) == _digits(key)), None)
        return self._fz[ck]

    def parse(self, title):
        """-> (year, make, model, source)  source: title | fuzzy | inferred | unknown"""
        if not isinstance(title, str) or not title.strip():
            return None, None, None, "empty"
        year, toks = self._prep(title)
        if not toks:
            return year, None, None, "empty"

        hit, source = self._find_make(toks), "title"
        if hit is None:                                          # typo / abbreviation: "Mercedez", "Peugot", "Merc Benz"
            keys = self._make_keys
            for i in range(min(len(toks), 2)):
                k = _norm(toks[i])
                f = None
                if len(k) >= 5:
                    f = self._fuzzy(k, keys, 5, self.make_fuzzy_cutoff)
                if not f and len(k) >= 4:                        # unique abbreviation prefix: "Merc" -> Mercedes-Benz
                    cand = {self.alias.get(m, m) for m in self.strong if m.startswith(k)}
                    if len(cand) == 1 and k not in self.hint and self.elsewhere[k] < 3:   # not a model word ("Land")
                        f = cand.pop()
                if f:
                    full, e = self.alias.get(f, f), i + 1
                    if e < len(toks):                            # swallow completing word: "Mercedez [Benz]"
                        nk = _norm(toks[e])
                        if len(nk) >= 3 and full.endswith(nk) and not full.startswith(nk):
                            e += 1
                    hit, source = (full, i, e), "fuzzy"
                    break
        if hit:
            mk, s, e = hit
            make, rest = self.display[mk], toks[e:]              # words before the make are dropped (e.g. "Hot Honda")
        else:                                                    # make absent: infer from the model name
            make, rest = None, list(toks)
            for i in range(len(toks)):
                for n in (2, 1):
                    if i + n > len(toks):
                        continue
                    k = _norm(" ".join(toks[i:i + n]))
                    got, src = self.hint.get(k), "inferred"
                    if not got:
                        f = self._fuzzy(k, self._hint_keys, 9 if n == 2 else 6)
                        if f:
                            got, src = self.hint[f], "fuzzy"
                            rest[i:i + n] = [self.surface[f].most_common(1)[0][0]]
                    if got:
                        make, source = self.display.get(got, got), src
                        break
                if make:
                    break
            if not make:                                         # last resort: alphabetic prefix, GLE53AMG -> 'gle'
                for t in toks[:3]:
                    m = re.match(r"([a-z]{2,})(?=\d)", _norm(t))
                    if m and m.group(1) in self.hint_pre:
                        make, source = self.display.get(self.hint_pre[m.group(1)], self.hint_pre[m.group(1)]), "inferred"
                        break
            if not make:
                source = "unknown"
        rest = [t for t in rest if not (year and t == str(year))]        # "...HARRIER 2015" repeated year
        model = " ".join(rest).strip(" -/.:") or None
        return year, make, model, source


def fix_titles(df, title_col="title", **kw):
    """Re-derive year/make/model from titles, learning makes from the data. Adds a make_source column."""
    p = TitleParser(**kw).fit(df[title_col])
    parsed = df[title_col].apply(lambda t: pd.Series(p.parse(t), index=["_y", "_mk", "_md", "make_source"]))
    out = df.copy()
    out["year"] = parsed["_y"].astype("Int64").fillna(pd.Series(out["year"]).round().astype("Int64"))
    out["make"], out["model"], out["make_source"] = parsed["_mk"], parsed["_md"], parsed["make_source"]
    # same model, different casing ("LAND CRUISER PRADO" vs "Land Cruiser Prado") -> one spelling
    key = out["make"].fillna("") + "|" + out["model"].fillna("").map(_norm)
    def best_form(col):
        c = Counter(col.dropna())
        if not c:
            return None
        top = c.most_common(1)[0][0]
        if top.isupper() and sum(ch.isalpha() for ch in top) <= 4:      # acronym-like: CX-5, GLC, X5
            return top
        return next((f for f, _ in c.most_common() if not f.isupper() and not f.islower()), top)
    forms = out.groupby(key)["model"].agg(best_form)
    out["model"] = key.map(forms).where(out["model"].notna())
    out.attrs["title_parser"] = p
    print("make_source:", out["make_source"].value_counts().to_dict(),
          "| distinct makes:", out["make"].nunique())
    return out
