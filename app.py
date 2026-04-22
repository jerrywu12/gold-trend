"""
Gold Research Panel — Streamlit web app
=======================================
Interactive gold research dashboard with 5 modules: macro, positioning,
technical, cross-asset, and seasonality.

Run
---
    pip install -r requirements.txt
    streamlit run app.py

Browser opens to http://localhost:8501 automatically.

Data sources (all free, no API key)
-----------------------------------
- Prices : yfinance  (GC=F, DX-Y.NYB, SPY, BTC-USD, SI=F, HG=F, GLD)
- Macro  : FRED      (DFII10, CPIAUCSL, T10YIE)
- COT    : CFTC Public Reporting  (Legacy Futures-Only)
"""

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# =====================================================================
# Page config & styling
# =====================================================================
st.set_page_config(
    page_title="Gold Research Panel",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand colors
GOLD = "#D4AF37"
BLUE = "#1f77b4"
RED = "#e04848"
GREEN = "#2ca02c"
PURPLE = "#7b4bb0"
GRAY = "#888888"
BTC_ORANGE = "#f7931a"

PLOTLY_TEMPLATE = "plotly_white"

# Lightweight CSS — consistent metric typography, tighter sidebar
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #666; }
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
h1 { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Robust HTTP fetch with retries + exponential backoff
# =====================================================================
def _http_get(url: str, timeout: int = 20, retries: int = 2,
              headers: dict | None = None) -> str:
    """
    GET a URL with retries. Handles transient timeouts and 5xx errors
    with exponential backoff (1s, 2s). Raises on permanent failure.
    Kept short (20s / 2 retries) so cloud deploys don't hang.
    """
    req = urllib.request.Request(
        url, headers=headers or {"User-Agent": "Mozilla/5.0"}
    )
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode()
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError) as e:
            last_err = e
            # Don't retry on permanent client errors (except rate limits)
            if isinstance(e, urllib.error.HTTPError) and \
                    e.code not in (429, 500, 502, 503, 504):
                raise
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise last_err  # type: ignore[misc]


# =====================================================================
# Cached data fetchers
# =====================================================================
# =====================================================================
# Cached data fetchers
# =====================================================================

# All price-only tickers batched into a single yfinance call
_PRICE_TICKERS = ["GC=F", "DX-Y.NYB", "SPY", "BTC-USD", "SI=F", "HG=F"]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_closes(start: str = "2010-01-01") -> pd.DataFrame:
    """Batch-download Close prices for all tickers in one network call."""
    raw = yf.download(
        _PRICE_TICKERS, start=start,
        end=datetime.today().strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False, threads=True,
    )
    return raw["Close"]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gld(start: str = "2010-01-01") -> pd.DataFrame:
    """GLD full OHLCV (needed for volume chart)."""
    df = yf.download(
        "GLD", start=start,
        end=datetime.today().strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_id: str) -> pd.Series:
    """FRED public CSV with retries. Returns empty Series on failure."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        txt = _http_get(url)
        df = pd.read_csv(StringIO(txt), parse_dates=[0], na_values=".")
        df.columns = ["date", series_id]
        return df.dropna().set_index("date")[series_id]
    except Exception as e:
        print(f"FRED {series_id} failed: {e}")   # visible in Streamlit Cloud logs
        return pd.Series(dtype=float, name=series_id)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cftc(commodity: str = "GOLD", limit: int = 400) -> pd.DataFrame:
    """CFTC Legacy Futures-Only report (published weekly on Friday)."""
    url = (
        "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        f"?$limit={limit}&$order=report_date_as_yyyy_mm_dd%20DESC"
        f"&commodity_name={commodity}"
    )
    try:
        txt = _http_get(url)
        data = json.loads(txt)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        df = df.set_index("date").sort_index()
        for c in df.columns:
            if any(k in c for k in ["positions", "_all", "open_interest"]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        print(f"CFTC failed: {e}")
        return pd.DataFrame()


# =====================================================================
# Technical indicators
# =====================================================================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


# =====================================================================
# Sidebar — controls
# =====================================================================
with st.sidebar:
    st.markdown("## 🪙 Gold Panel")
    st.caption("Quant gold research dashboard")

    st.markdown("### ⚙️ Settings")
    lookback_opt = st.selectbox(
        "Lookback window",
        ["6M", "1Y", "2Y", "5Y", "10Y", "All"],
        index=2,
        help="Applies to most charts. Seasonality always uses full history.",
    )
    lookback_days = {"6M": 180, "1Y": 365, "2Y": 730,
                     "5Y": 1825, "10Y": 3650, "All": 99999}[lookback_opt]

    st.markdown("---")
    if st.button("🔄 Force Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        f"Data auto-refreshes every hour.  \n"
        f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    with st.expander("📖 About"):
        st.markdown("""
        **Five modules**
        - Macro drivers
        - Positioning
        - Technical analysis
        - Cross-asset
        - Seasonality

        **Data sources (all free, no API key)**
        - [yfinance](https://github.com/ranaroussi/yfinance)
        - [FRED CSV](https://fred.stlouisfed.org)
        - [CFTC Public Reporting](https://publicreporting.cftc.gov)
        """)


# =====================================================================
# Load data
# =====================================================================
with st.spinner("Loading data…"):
    # --- Prices: one batch yfinance call instead of 6 separate ones ---
    closes = fetch_all_closes()
    gld = fetch_gld()

    # --- Macro + COT: 4 calls fired in parallel ---
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_ry  = pool.submit(fetch_fred, "DFII10")
        f_cpi = pool.submit(fetch_fred, "CPIAUCSL")
        f_be  = pool.submit(fetch_fred, "T10YIE")
        f_cot = pool.submit(fetch_cftc, "GOLD")

    real_yield = f_ry.result().rename("Real_10Y")
    cpi        = f_cpi.result().rename("CPI")
    breakeven  = f_be.result().rename("Breakeven_10Y")
    cot        = f_cot.result()

    # --- Extract individual series from batch Close DataFrame ---
    gold   = closes["GC=F"].rename("Gold").dropna()
    dxy    = closes["DX-Y.NYB"].rename("DXY").dropna()
    spy    = closes["SPY"].rename("SPY").dropna()
    btc    = closes["BTC-USD"].rename("BTC").dropna()
    silver = closes["SI=F"].rename("Silver").dropna()
    copper = closes["HG=F"].rename("Copper").dropna()

# Surface any FRED/CFTC failures now that we're back on the main thread
if real_yield.empty:
    st.warning("10Y Real Yield (FRED DFII10) unavailable — macro charts will be partial. Try Refresh.")
if cpi.empty:
    st.warning("CPI (FRED CPIAUCSL) unavailable — macro charts will be partial. Try Refresh.")
if breakeven.empty:
    st.warning("Breakeven inflation (FRED T10YIE) unavailable. Try Refresh.")
if cot.empty:
    st.warning("CFTC COT data unavailable — positioning chart will be skipped. Try Refresh.")

if gold.empty:
    st.error("Could not load gold price data from yfinance. "
             "Check your network and click Refresh.")
    st.stop()

cutoff = gold.index.max() - pd.Timedelta(days=lookback_days)
gold_lb = gold.loc[gold.index >= cutoff]


# =====================================================================
# Header — title + key metrics
# =====================================================================
st.title("🪙 Gold Research Panel")
st.caption(
    f"Data as of {gold.index[-1].date()}  ·  "
    f"Lookback: {lookback_opt}  ·  Cache: 1 hour"
)

price = float(gold.iloc[-1])
prev = float(gold.iloc[-2]) if len(gold) > 1 else price
daily_pct = (price / prev - 1) * 100

ret_1m = (price / float(gold.iloc[-21]) - 1) * 100 if len(gold) > 21 else np.nan
ret_1y = (price / float(gold.iloc[-252]) - 1) * 100 if len(gold) > 252 else np.nan
ytd_start = gold[gold.index.year == datetime.today().year]
ret_ytd = (price / float(ytd_start.iloc[0]) - 1) * 100 if not ytd_start.empty else np.nan

rsi_now = float(rsi(gold, 14).iloc[-1])
ma200 = float(gold.rolling(200).mean().iloc[-1])
gs_ratio_now = float((gold / silver).dropna().iloc[-1])

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Gold ($/oz)", f"${price:,.2f}", f"{daily_pct:+.2f}% d/d")
c2.metric("YTD", f"{ret_ytd:+.1f}%")
c3.metric("1M", f"{ret_1m:+.1f}%")
c4.metric("1Y", f"{ret_1y:+.1f}%")
c5.metric(
    "RSI(14)", f"{rsi_now:.1f}",
    delta="Overbought" if rsi_now > 70 else "Oversold" if rsi_now < 30 else "Neutral",
    delta_color="off",
)
c6.metric(
    "vs MA200", f"{(price/ma200-1)*100:+.1f}%",
    delta="Above" if price > ma200 else "Below",
    delta_color="normal" if price > ma200 else "inverse",
)

st.markdown("")

# =====================================================================
# Tabs
# =====================================================================
tab_overview, tab_macro, tab_pos, tab_tech, tab_cross, tab_seas = st.tabs([
    "📊 Overview",
    "🌍 Macro",
    "📈 Positioning",
    "🔧 Technical",
    "🔄 Cross-Asset",
    "📅 Seasonality",
])


# ---------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------
with tab_overview:
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("Gold Price")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gold_lb.index, y=gold_lb.values,
            line=dict(color=GOLD, width=2),
            name="Gold",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=gold_lb.index, y=gold_lb.rolling(50).mean(),
            line=dict(color=BLUE, width=1), name="MA50",
        ))
        fig.add_trace(go.Scatter(
            x=gold_lb.index, y=gold_lb.rolling(200).mean(),
            line=dict(color=RED, width=1), name="MA200",
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified",
            yaxis_title="$/oz",
            legend=dict(orientation="h", y=1.02, x=0),
        )
        st.plotly_chart(fig, width="stretch")

    with col_r:
        st.subheader("Snapshot")

        ma50_now = float(gold.rolling(50).mean().iloc[-1])
        if price > ma50_now > ma200:
            trend, trend_emoji = "Uptrend", "🟢"
        elif price < ma50_now < ma200:
            trend, trend_emoji = "Downtrend", "🔴"
        else:
            trend, trend_emoji = "Mixed", "🟡"

        mac_line, mac_sig, _ = macd(gold)
        mac_bias = "Bullish (MACD > signal)" if mac_line.iloc[-1] > mac_sig.iloc[-1] \
                   else "Bearish (MACD < signal)"

        gs_series = (gold / silver).dropna()
        gs_mean = gs_series.mean()
        gs_dev = (gs_ratio_now - gs_mean) / gs_series.std()

        snap_rows = [
            ("Trend", f"{trend_emoji} {trend}"),
            ("MACD", mac_bias),
            ("Gold/Silver", f"{gs_ratio_now:.1f}  ({gs_dev:+.1f}σ from mean)"),
        ]
        if not cot.empty and "noncomm_positions_long_all" in cot.columns:
            net = (cot["noncomm_positions_long_all"].iloc[-1]
                   - cot["noncomm_positions_short_all"].iloc[-1])
            snap_rows.append(
                ("CFTC Net Long",
                 f"{int(net):+,} contracts ({cot.index[-1].date()})")
            )

        for label, val in snap_rows:
            st.markdown(f"**{label}**  \n{val}")

        st.markdown("---")
        st.markdown("**Latest Quotes**")
        quote_df = pd.DataFrame({
            "Asset": ["Gold", "DXY", "Silver", "Copper", "SPY", "BTC"],
            "Price": [
                f"${price:,.2f}",
                f"{float(dxy.iloc[-1]):.2f}",
                f"${float(silver.iloc[-1]):.2f}",
                f"${float(copper.iloc[-1]):.2f}",
                f"${float(spy.iloc[-1]):.2f}",
                f"${float(btc.iloc[-1]):,.0f}",
            ],
        })
        st.dataframe(quote_df, hide_index=True, width="stretch")


# ---------------------------------------------------------------------
# Macro tab
# ---------------------------------------------------------------------
with tab_macro:
    st.subheader("Module 1 — Macro Drivers")

    def dual_axis_chart(title, macro_s, macro_name, macro_color,
                        invert_macro=False):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=gold_lb.index, y=gold_lb.values,
            line=dict(color=GOLD, width=1.8), name="Gold",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
        ), secondary_y=False)
        if not macro_s.empty:
            s = macro_s.loc[macro_s.index >= cutoff]
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values,
                line=dict(color=macro_color, width=1.2),
                name=macro_name, opacity=0.85,
            ), secondary_y=True)
            if invert_macro:
                fig.update_yaxes(autorange="reversed", secondary_y=True)
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=340, title=title,
            margin=dict(l=10, r=10, t=40, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=0),
        )
        fig.update_yaxes(title_text="Gold $/oz", secondary_y=False, color=GOLD)
        fig.update_yaxes(title_text=macro_name, secondary_y=True, color=macro_color)
        return fig

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        dual_axis_chart("Gold vs 10Y Real Yield (inverted)",
                        real_yield, "10Y Real Yield %", BLUE, invert_macro=True),
        width="stretch",
    )
    c2.plotly_chart(
        dual_axis_chart("Gold vs USD Index (DXY)",
                        dxy, "DXY", GREEN),
        width="stretch",
    )

    c3, c4 = st.columns(2)
    cpi_yoy = cpi.pct_change(12) * 100 if not cpi.empty else pd.Series(dtype=float)
    c3.plotly_chart(
        dual_axis_chart("Gold vs US CPI YoY",
                        cpi_yoy, "US CPI YoY %", RED),
        width="stretch",
    )
    c4.plotly_chart(
        dual_axis_chart("Gold vs 10Y Breakeven Inflation",
                        breakeven, "10Y Breakeven %", PURPLE),
        width="stretch",
    )

    # Correlation matrix
    st.markdown("#### 1-Year Correlation Matrix")
    macro_df = pd.concat([gold, dxy, real_yield, breakeven],
                         axis=1, sort=True).ffill().dropna()
    if not macro_df.empty:
        recent = macro_df.loc[macro_df.index >= macro_df.index.max() - pd.Timedelta(days=365)]
        corr = recent.corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr.values, texttemplate="%{text:.2f}",
            textfont={"size": 13},
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320,
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Correlation matrix needs macro data — try Refresh.")


# ---------------------------------------------------------------------
# Positioning tab
# ---------------------------------------------------------------------
with tab_pos:
    st.subheader("Module 2 — Positioning")

    if not cot.empty and "noncomm_positions_long_all" in cot.columns:
        cot["net_noncomm"] = (cot["noncomm_positions_long_all"]
                              - cot["noncomm_positions_short_all"])

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=cot.index, y=cot["net_noncomm"] / 1000,
            name="Non-Comm Net (k contracts)",
            marker_color=BLUE, opacity=0.55,
            hovertemplate="%{x|%Y-%m-%d}<br>Net: %{y:,.1f}k<extra></extra>",
        ), secondary_y=False)

        gold_cot = gold.reindex(cot.index, method="ffill")
        fig.add_trace(go.Scatter(
            x=gold_cot.index, y=gold_cot.values,
            line=dict(color=GOLD, width=1.8), name="Gold",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
        ), secondary_y=True)

        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=420,
            title="CFTC Non-Commercial Net Position vs Gold Price",
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Net (k contracts)", secondary_y=False, color=BLUE)
        fig.update_yaxes(title_text="Gold $/oz", secondary_y=True, color=GOLD)
        st.plotly_chart(fig, width="stretch")

        latest = cot.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Report Date", cot.index[-1].strftime("%Y-%m-%d"))
        c2.metric("Non-Comm Long", f"{int(latest['noncomm_positions_long_all']):,}")
        c3.metric("Non-Comm Short", f"{int(latest['noncomm_positions_short_all']):,}")
        c4.metric("Net Long", f"{int(latest['net_noncomm']):+,}")
    else:
        st.info("CFTC COT data unavailable — try the Refresh button.")

    # GLD
    st.markdown("#### SPDR Gold Trust (GLD)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    gld_lb = gld.loc[gld.index >= cutoff]
    fig.add_trace(go.Scatter(
        x=gld_lb.index, y=gld_lb["Close"],
        line=dict(color=GOLD, width=1.8), name="GLD Price",
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=gld_lb.index, y=gld_lb["Volume"].rolling(20).mean() / 1e6,
        name="20d Avg Volume (M)", marker_color=BLUE, opacity=0.35,
    ), secondary_y=True)
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="GLD $", secondary_y=False, color=GOLD)
    fig.update_yaxes(title_text="Volume (M)", secondary_y=True, color=BLUE)
    st.plotly_chart(fig, width="stretch")

    st.caption("💡 For actual GLD tonnage, download the daily CSV from the SSGA website.")


# ---------------------------------------------------------------------
# Technical tab
# ---------------------------------------------------------------------
with tab_tech:
    st.subheader("Module 3 — Technical Analysis")

    g = gold.to_frame()
    g["MA50"] = g["Gold"].rolling(50).mean()
    g["MA200"] = g["Gold"].rolling(200).mean()
    g["BB_mid"] = g["Gold"].rolling(20).mean()
    _bb = g["Gold"].rolling(20).std()
    g["BB_up"] = g["BB_mid"] + 2 * _bb
    g["BB_lo"] = g["BB_mid"] - 2 * _bb
    g["RSI14"] = rsi(g["Gold"], 14)
    g["MACD"], g["MACD_sig"], g["MACD_hist"] = macd(g["Gold"])
    gp = g.loc[g.index >= cutoff]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price + Moving Averages + Bollinger Bands",
                        "RSI(14)",
                        "MACD (12,26,9)"),
    )

    # Bollinger fill
    fig.add_trace(go.Scatter(
        x=gp.index, y=gp["BB_up"], line=dict(color=PURPLE, width=0),
        showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=gp.index, y=gp["BB_lo"], line=dict(color=PURPLE, width=0),
        fill="tonexty", fillcolor="rgba(123,75,176,0.1)",
        name="BB 20,2σ", hoverinfo="skip"), row=1, col=1)
    # Price + MAs
    fig.add_trace(go.Scatter(
        x=gp.index, y=gp["Gold"], line=dict(color=GOLD, width=1.8),
        name="Gold"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=gp.index, y=gp["MA50"], line=dict(color=BLUE, width=1),
        name="MA50"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=gp.index, y=gp["MA200"], line=dict(color=RED, width=1),
        name="MA200"), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=gp.index, y=gp["RSI14"], line=dict(color=PURPLE, width=1.2),
        name="RSI", showlegend=False), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line=dict(color=RED, dash="dash", width=0.8))
    fig.add_hline(y=30, row=2, col=1, line=dict(color=GREEN, dash="dash", width=0.8))

    # MACD
    hist_colors = [GREEN if v >= 0 else RED for v in gp["MACD_hist"]]
    fig.add_trace(go.Bar(
        x=gp.index, y=gp["MACD_hist"], marker_color=hist_colors,
        opacity=0.5, name="Hist", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=gp.index, y=gp["MACD"], line=dict(color=BLUE, width=1),
        name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=gp.index, y=gp["MACD_sig"], line=dict(color=RED, width=1),
        name="Signal"), row=3, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=720,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_yaxes(title_text="$/oz", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    st.plotly_chart(fig, width="stretch")

    last = g.dropna(subset=["MA200"]).iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gold", f"${float(last['Gold']):,.2f}")
    c2.metric("MA50", f"${float(last['MA50']):,.2f}",
              "above" if price > float(last["MA50"]) else "below")
    c3.metric("MA200", f"${float(last['MA200']):,.2f}",
              "above" if price > float(last["MA200"]) else "below")
    c4.metric("RSI(14)", f"{float(last['RSI14']):.1f}",
              "overbought" if last["RSI14"] > 70 else
              "oversold" if last["RSI14"] < 30 else "neutral",
              delta_color="off")


# ---------------------------------------------------------------------
# Cross-Asset tab
# ---------------------------------------------------------------------
with tab_cross:
    st.subheader("Module 4 — Cross-Asset")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Normalized Performance (base = 100)**")
        start_ra = gold_lb.index[0]
        fig = go.Figure()
        for name, s, c in [("Gold", gold, GOLD), ("SPY", spy, BLUE),
                           ("BTC", btc, BTC_ORANGE), ("Silver", silver, GRAY)]:
            s2 = s.loc[start_ra:].dropna()
            if not s2.empty:
                fig.add_trace(go.Scatter(
                    x=s2.index, y=s2 / s2.iloc[0] * 100,
                    line=dict(color=c, width=1.3), name=name,
                ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified", yaxis_type="log",
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        gs = (gold / silver).dropna()
        gs_mean, gs_std = gs.mean(), gs.std()
        gs_lb = gs.loc[gs.index >= cutoff]
        st.markdown(
            f"**Gold/Silver Ratio** (current {float(gs.iloc[-1]):.1f}, "
            f"mean {gs_mean:.0f}, σ {gs_std:.0f})"
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gs_lb.index, y=gs_lb.values,
            line=dict(color=PURPLE, width=1.3), name="Gold/Silver",
        ))
        for lvl, color, label in [
            (gs_mean, "black", f"Mean {gs_mean:.0f}"),
            (gs_mean + gs_std, RED, "+1σ"),
            (gs_mean - gs_std, GREEN, "-1σ"),
        ]:
            fig.add_hline(y=lvl, line=dict(color=color, dash="dot", width=0.8),
                          annotation_text=label, annotation_position="right")
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Gold 60-day Rolling Correlation vs Other Assets**")
        rets = pd.concat([
            gold.pct_change(),
            spy.pct_change().rename("SPY"),
            btc.pct_change().rename("BTC"),
            silver.pct_change().rename("Silver"),
        ], axis=1, sort=True).dropna()
        fig = go.Figure()
        for col, c in [("SPY", BLUE), ("BTC", BTC_ORANGE), ("Silver", GRAY)]:
            roll = rets["Gold"].rolling(60).corr(rets[col])
            roll_lb = roll.loc[roll.index >= cutoff]
            fig.add_trace(go.Scatter(
                x=roll_lb.index, y=roll_lb.values,
                line=dict(color=c, width=1.1), name=col,
            ))
        fig.add_hline(y=0, line=dict(color="black", width=0.5))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(fig, width="stretch")

    with c4:
        cg = (copper / gold * 1000).dropna()
        cg_lb = cg.loc[cg.index >= cutoff]
        st.markdown(
            f"**Copper/Gold Ratio (×1000)** — risk-appetite gauge, "
            f"current {float(cg.iloc[-1]):.2f}"
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cg_lb.index, y=cg_lb.values,
            line=dict(color=GREEN, width=1.2), name="Copper/Gold",
            fill="tozeroy", fillcolor="rgba(44,160,44,0.1)",
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------
# Seasonality tab
# ---------------------------------------------------------------------
with tab_seas:
    st.subheader("Module 5 — Seasonality")
    st.caption("Seasonality uses the full price history, regardless of the Lookback setting.")

    gold_m = gold.resample("ME").last()
    gold_mr = gold_m.pct_change() * 100
    seas = gold_mr.to_frame("ret").assign(
        year=lambda d: d.index.year, month=lambda d: d.index.month
    )
    heat = seas.pivot(index="year", columns="month", values="ret")
    heat.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    avg_month = heat.mean()

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("**Monthly Returns Heatmap (%)**")
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=heat.columns, y=heat.index,
            colorscale="RdYlGn", zmid=0,
            text=heat.values, texttemplate="%{text:.1f}",
            textfont={"size": 10},
            colorbar=dict(title="%"),
            hovertemplate="%{y} %{x}<br>Return: %{z:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown(f"**Average Monthly Return ({heat.index.min()}–{heat.index.max()})**")
        bar_colors = [GREEN if v >= 0 else RED for v in avg_month.values]
        fig = go.Figure(go.Bar(
            x=heat.columns, y=avg_month.values,
            marker_color=bar_colors,
            text=[f"{v:.2f}%" for v in avg_month.values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig.add_hline(y=0, line=dict(color="black", width=0.5))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="% Return",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    # Current year vs historical path
    st.markdown("#### YTD Path vs Historical")
    cur_year = datetime.today().year
    gd = gold.to_frame("Gold")
    gd["year"] = gd.index.year
    gd["doy"] = gd.index.dayofyear

    fig = go.Figure()

    for y in range(cur_year - 5, cur_year):
        sub = gd[gd["year"] == y].copy()
        if sub.empty:
            continue
        sub["norm"] = sub["Gold"] / sub["Gold"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=sub["doy"], y=sub["norm"],
            line=dict(width=0.9), opacity=0.4, name=str(y),
        ))

    hist = gd[gd["year"] < cur_year].copy()
    if not hist.empty:
        hist["norm"] = hist.groupby("year")["Gold"].transform(
            lambda x: x / x.iloc[0] * 100
        )
        avg_path = hist.groupby("doy")["norm"].mean()
        fig.add_trace(go.Scatter(
            x=avg_path.index, y=avg_path.values,
            line=dict(color="black", width=2, dash="dash"),
            name="All-years Average",
        ))

    cur = gd[gd["year"] == cur_year].copy()
    if not cur.empty:
        cur["norm"] = cur["Gold"] / cur["Gold"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=cur["doy"], y=cur["norm"],
            line=dict(color=GOLD, width=3),
            name=f"{cur_year} YTD",
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Day of Year",
        yaxis_title="Rebased (start = 100)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig, width="stretch")


# =====================================================================
# Footer
# =====================================================================
st.markdown("---")
st.caption(
    "⚠️  For research only — not investment advice.  ·  "
    "Data: yfinance, FRED, CFTC  ·  "
    "Built with Streamlit + Plotly"
)