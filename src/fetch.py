"""pytrends fetching with CSV cache, retry/backoff, and manual-export fallback."""
import os
import time
import pandas as pd
from src.config import RAW_DIR, FULL_TIMEFRAME


def _slug(name):
    return name.replace(" ", "_").lower()


def _iot_path(ingredient, tag):
    return RAW_DIR / f"{_slug(ingredient)}_iot_{tag}.csv"


def _region_path(ingredient, tag):
    return RAW_DIR / f"{_slug(ingredient)}_region_{tag}.csv"


def _new_client():
    from pytrends.request import TrendReq
    # NOTE: pytrends' own retries= arg crashes on urllib3 v2 (removed method_whitelist)
    # and doesn't help against sustained IP-level blocks anyway, so we rely on our
    # operation-level _fetch_with_retry plus the CSV cache/fallback instead.
    #
    # If a TRENDS_NID env var is set, pass ONLY that NID cookie through to the
    # underlying session. A valid browser session cookie often clears Google's
    # bot-detection block. We deliberately send nothing but NID — never the full
    # browser Cookie header, which would leak account-auth tokens. The value lives
    # only in the environment; it is never logged, printed, or written to disk.
    client = TrendReq(hl="en-US", tz=360)

    nid = os.environ.get("TRENDS_NID", "").strip()
    if nid:
        # Strip an optional "NID=" prefix so we store just the value.
        value = nid[4:] if nid.lower().startswith("nid=") else nid
        # pytrends fetches its own NID during __init__ (via GetGoogleCookie) and
        # uses self.cookies (a plain dict in this version) for the actual data
        # requests. Overwrite that NID with our browser NID so the data calls carry
        # a valid session. We touch only the NID cookie — never the full browser
        # Cookie header (account-auth tokens). The value stays in memory; it is
        # never logged or written.
        if isinstance(client.cookies, dict):
            client.cookies["NID"] = value
        else:  # CookieJar in other pytrends versions
            client.cookies.set("NID", value, domain=".google.com")
    return client


def _fetch_with_retry(build_and_get, retries=3):
    """Call a pytrends operation with exponential backoff on failure (e.g. 429)."""
    delay = 2.0
    for attempt in range(retries):
        try:
            return build_and_get()
        except Exception as e:  # pytrends raises ResponseError / TooManyRequests
            if attempt == retries - 1:
                raise
            print(f"  pytrends failed ({e}); retry in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2


def fetch_interest_over_time(ingredient, timeframe=FULL_TIMEFRAME, tag="full"):
    """Return a DataFrame with a 'value' column. Cache-first, then live, then manual."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _iot_path(ingredient, tag)
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)

    try:
        client = _new_client()

        def op():
            client.build_payload([ingredient], cat=0, timeframe=timeframe)
            return client.interest_over_time()

        df = _fetch_with_retry(op)
        time.sleep(1)  # rate-limit courtesy
        df = df[[ingredient]].rename(columns={ingredient: "value"})
        df.to_csv(path)
        return df
    except Exception as e:
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        raise RuntimeError(
            f"Could not fetch interest_over_time for '{ingredient}'. "
            f"pytrends failed ({e}). Manually export the CSV from "
            f"trends.google.com and save it to: {path}"
        )


def fetch_interest_by_region(ingredient, timeframe, tag):
    """Per-window region interest. Cache-first, then live, then manual fallback."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _region_path(ingredient, tag)
    if path.exists():
        return pd.read_csv(path)

    try:
        client = _new_client()

        def op():
            client.build_payload([ingredient], cat=0, timeframe=timeframe)
            return client.interest_by_region()

        region = _fetch_with_retry(op)
        time.sleep(1)
        region = region.rename(columns={ingredient: "interest"})[["interest"]]
        region.to_csv(path, index=False)
        return region
    except Exception as e:
        if path.exists():
            return pd.read_csv(path)
        raise RuntimeError(
            f"Could not fetch interest_by_region for '{ingredient}' [{tag}]. "
            f"pytrends failed ({e}). Save a manual export to: {path}"
        )
