from app.main import (
    ScanRequest,
    _bootstrap_state,
    get_backtest,
    get_latest_scan,
    get_ledger,
    get_stock_detail,
    health,
    run_scan,
    view_ledger_spreadsheet,
)


def main() -> None:
    _bootstrap_state()

    h = health()
    assert h["status"] == "ok", h

    s = get_latest_scan()
    assert 0 <= len(s["top_longs"]) <= 5, s
    assert 0 <= len(s["top_shorts"]) <= 5, s
    for card in s["top_longs"] + s["top_shorts"]:
        assert card["target_gain_pct"] >= 5.0, card
        assert len(card["drivers"]) > 0, card

    p = run_scan(ScanRequest(simulate_1550_provisional=True))
    assert p["scan_mode"] == "PROVISIONAL_1550", p

    b = get_backtest()
    assert "metrics" in b and "equity_curve" in b

    sym = (s["top_longs"][0]["symbol"] if s["top_longs"] else "AAPL")
    st = get_stock_detail(sym)
    assert len(st["candles"]) > 0

    ledger = get_ledger()
    html_resp = view_ledger_spreadsheet()
    assert "US Trading App" in html_resp.body.decode("utf-8")

    print("HEALTH:", h)
    if s["top_longs"]:
        print("TOP LONG #1:", s["top_longs"][0]["symbol"], s["top_longs"][0]["grade"], s["top_longs"][0]["win_prob_pct"])
    else:
        print("TOP LONG #1: NONE (>=48% prob & >=5% gain)")
    if s["top_shorts"]:
        print("TOP SHORT #1:", s["top_shorts"][0]["symbol"], s["top_shorts"][0]["grade"], s["top_shorts"][0]["win_prob_pct"])
    else:
        print("TOP SHORT #1: NONE (>=48% prob & >=5% gain)")
    print("PROVISIONAL MODE:", p["scan_mode"])
    print("BACKTEST METRICS:", b["metrics"])
    print("LEDGER COUNT:", ledger["count"])
    print("ALL US SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
