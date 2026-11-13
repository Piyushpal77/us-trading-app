from __future__ import annotations

import io
from dataclasses import dataclass
import httpx
import pandas as pd


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    name: str
    sector: str
    index_membership: str  # "SP500", "NDX100", or "SP500+NDX100"
    is_shortable: bool = True


SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
NASDAQ100_API_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"

# Curated fallback of major S&P 500 & Nasdaq 100 leaders
FALLBACK_US_STOCKS: list[tuple[str, str, str]] = [
    ("AAPL", "Information Technology", "SP500+NDX100"),
    ("MSFT", "Information Technology", "SP500+NDX100"),
    ("NVDA", "Information Technology", "SP500+NDX100"),
    ("AMZN", "Consumer Discretionary", "SP500+NDX100"),
    ("GOOGL", "Communication Services", "SP500+NDX100"),
    ("META", "Communication Services", "SP500+NDX100"),
    ("TSLA", "Consumer Discretionary", "SP500+NDX100"),
    ("AVGO", "Information Technology", "SP500+NDX100"),
    ("AMD", "Information Technology", "SP500+NDX100"),
    ("NFLX", "Communication Services", "SP500+NDX100"),
    ("COST", "Consumer Staples", "SP500+NDX100"),
    ("PEP", "Consumer Staples", "SP500+NDX100"),
    ("ADBE", "Information Technology", "SP500+NDX100"),
    ("CSCO", "Information Technology", "SP500+NDX100"),
    ("TMUS", "Communication Services", "SP500+NDX100"),
    ("INTU", "Information Technology", "SP500+NDX100"),
    ("TXN", "Information Technology", "SP500+NDX100"),
    ("QCOM", "Information Technology", "SP500+NDX100"),
    ("AMAT", "Information Technology", "SP500+NDX100"),
    ("ISRG", "Health Care", "SP500+NDX100"),
    ("BKNG", "Consumer Discretionary", "SP500+NDX100"),
    ("PANW", "Information Technology", "SP500+NDX100"),
    ("CRWD", "Information Technology", "SP500+NDX100"),
    ("ASML", "Information Technology", "NDX100"),
    ("ARM", "Information Technology", "NDX100"),
    ("MELI", "Consumer Discretionary", "NDX100"),
    ("MRVL", "Information Technology", "NDX100"),
    ("JPM", "Financials", "SP500"),
    ("V", "Financials", "SP500"),
    ("MA", "Financials", "SP500"),
    ("GS", "Financials", "SP500"),
    ("MS", "Financials", "SP500"),
    ("BRK-B", "Financials", "SP500"),
    ("UNH", "Health Care", "SP500"),
    ("LLY", "Health Care", "SP500"),
    ("JNJ", "Health Care", "SP500"),
    ("MRK", "Health Care", "SP500"),
    ("ABBV", "Health Care", "SP500"),
    ("XOM", "Energy", "SP500"),
    ("CVX", "Energy", "SP500"),
    ("COP", "Energy", "SP500"),
    ("SLB", "Energy", "SP500"),
    ("CAT", "Industrials", "SP500"),
    ("GE", "Industrials", "SP500"),
    ("LMT", "Industrials", "SP500"),
    ("BA", "Industrials", "SP500"),
    ("WMT", "Consumer Staples", "SP500"),
    ("HD", "Consumer Discretionary", "SP500"),
    ("DIS", "Communication Services", "SP500"),
    ("FCX", "Materials", "SP500"),
]


FALLBACK_US_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "NVDA": "NVIDIA Corp.",
    "AMZN": "Amazon.com Inc.", "GOOGL": "Alphabet Inc.", "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.", "AVGO": "Broadcom Inc.", "AMD": "Advanced Micro Devices",
    "NFLX": "Netflix Inc.", "COST": "Costco Wholesale", "PEP": "PepsiCo Inc.",
    "ADBE": "Adobe Inc.", "CSCO": "Cisco Systems", "TMUS": "T-Mobile US",
    "INTU": "Intuit Inc.", "TXN": "Texas Instruments", "QCOM": "Qualcomm Inc.",
    "AMAT": "Applied Materials", "ISRG": "Intuitive Surgical", "BKNG": "Booking Holdings",
    "PANW": "Palo Alto Networks", "CRWD": "CrowdStrike Holdings", "ASML": "ASML Holding",
    "ARM": "Arm Holdings", "MELI": "MercadoLibre Inc.", "MRVL": "Marvell Technology",
    "JPM": "JPMorgan Chase & Co.", "V": "Visa Inc.", "MA": "Mastercard Inc.",
    "GS": "Goldman Sachs Group", "MS": "Morgan Stanley", "BRK-B": "Berkshire Hathaway",
    "UNH": "UnitedHealth Group", "LLY": "Eli Lilly and Co.", "JNJ": "Johnson & Johnson",
    "MRK": "Merck & Co.", "ABBV": "AbbVie Inc.", "XOM": "Exxon Mobil Corp.",
    "CVX": "Chevron Corp.", "COP": "ConocoPhillips", "SLB": "SLB",
    "CAT": "Caterpillar Inc.", "GE": "GE Aerospace", "LMT": "Lockheed Martin",
    "BA": "Boeing Co.", "WMT": "Walmart Inc.", "HD": "Home Depot Inc.",
    "DIS": "Walt Disney Co.", "FCX": "Freeport-McMoRan Inc.",
}


def _normalize_us_ticker(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")


def fetch_us_universe(limit: int = 550) -> list[UniverseMember]:
    """Fetch S&P 500 + Nasdaq 100 constituents with resilient fallback."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    sp500_map: dict[str, tuple[str, str]] = {}
    ndx_map: dict[str, tuple[str, str]] = {}

    try:
        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            resp = client.get(SP500_CSV_URL, headers=headers)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if "Symbol" in df.columns:
                for _, row in df.iterrows():
                    sym = _normalize_us_ticker(row["Symbol"])
                    if sym and sym != "NAN":
                        nm = str(row.get("Security", sym)).strip()
                        sec = str(row.get("GICS Sector", "Equities")).strip()
                        sp500_map[sym] = (nm or sym, sec)
    except Exception:
        pass

    try:
        ndx_headers = {**headers, "Accept": "application/json"}
        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            resp = client.get(NASDAQ100_API_URL, headers=ndx_headers)
            resp.raise_for_status()
            rows = resp.json().get("data", {}).get("data", {}).get("rows") or []
            for row in rows:
                sym = _normalize_us_ticker(row.get("symbol", ""))
                if sym and sym != "NAN" and len(sym) <= 6:
                    nm = str(row.get("companyName") or sym).replace(" Common Stock", "").strip()
                    sec = str(row.get("sector") or "Technology").strip() or "Technology"
                    ndx_map[sym] = (nm or sym, sec)
    except Exception:
        pass

    if sp500_map or ndx_map:
        all_syms = sorted(set(sp500_map.keys()) | set(ndx_map.keys()))[:limit]
        members: list[UniverseMember] = []
        for sym in all_syms:
            in_sp = sym in sp500_map
            in_ndx = sym in ndx_map
            tag = "SP500+NDX100" if (in_sp and in_ndx) else ("SP500" if in_sp else "NDX100")
            nm, sector = sp500_map.get(sym) or ndx_map.get(sym) or (sym, "Equities")
            members.append(
                UniverseMember(symbol=sym, name=nm, sector=sector, index_membership=tag, is_shortable=True)
            )
        return members

    return [
        UniverseMember(
            symbol=s,
            name=FALLBACK_US_NAMES.get(s, s),
            sector=sec,
            index_membership=tag,
            is_shortable=True,
        )
        for s, sec, tag in FALLBACK_US_STOCKS[:limit]
    ]
