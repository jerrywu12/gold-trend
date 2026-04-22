# %% [markdown]
# # 黄金研究面板 | Gold Research Panel
#
# 综合宏观、持仓、技术、跨资产与季节性五大模块的黄金研究仪表板。
# A consolidated gold research dashboard covering macro, positioning,
# technical, cross-asset, and seasonality modules.
#
# **使用方法 | Usage**
# - 在 VSCode / Cursor 中打开 → 按 cell (`# %%`) 逐段运行
# - 或在 Jupyter 中 `jupytext --to notebook gold_research_panel.py`
# - 或作为普通脚本 `python gold_research_panel.py` 依次生成所有图表
#
# **依赖 | Requirements**
# ```
# pip install yfinance pandas numpy matplotlib
# ```
# 无需 API key(FRED/CFTC 均为公开 CSV/JSON 端点)。
#
# **数据源 | Data sources**
# - 价格 Prices : yfinance (GC=F, DX-Y.NYB, SPY, BTC-USD, SI=F, HG=F, GLD)
# - 宏观 Macro  : FRED (DFII10, CPIAUCSL, T10YIE, DTWEXBGS)
# - 持仓 COT    : CFTC Public Reporting API (Legacy Futures-Only)

# %% ============================================================
#    Setup — imports, styling, helpers
# ==============================================================
import json
import urllib.request
import warnings
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# 中文字体自动检测 | Auto-detect CJK font across platforms
from matplotlib import font_manager

def _pick_cjk_font():
    """按优先级挑一个系统已安装的中日韩字体,找不到则返回 None。"""
    candidates = [
        "Microsoft JhengHei", "Microsoft YaHei",        # Windows (TC / SC)
        "PingFang TC", "PingFang SC", "Heiti TC", "Heiti SC",  # macOS
        "Noto Sans CJK TC", "Noto Sans CJK SC",         # Linux / modern
        "Source Han Sans TC", "Source Han Sans SC",     # Adobe
        "PMingLiU", "SimHei", "SimSun",                 # 传统 fallback
        "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return None

_cjk = _pick_cjk_font()
if _cjk:
    print(f"[font] 使用中文字体: {_cjk}")
else:
    print("[font] 未检测到中文字体 — 中文标签可能显示为方框 (仅影响显示, 不影响功能)")

# 图表样式 | Chart styling
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": [_cjk, "DejaVu Sans"] if _cjk else "DejaVu Sans",
    "font.size": 10,
    "axes.unicode_minus": False,  # 避免负号显示为方框
})

# 配色 | Colors
GOLD = "#D4AF37"
BLUE = "#1f77b4"
RED = "#d62728"
GREEN = "#2ca02c"
PURPLE = "#7b4bb0"
GRAY = "#888888"

# 时间范围 | Time range
START = "2010-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")


# -------------------- 数据拉取工具函数 | Data helpers --------------------
def fetch_yf(ticker: str, start: str = START, end: str = TODAY) -> pd.DataFrame:
    """从 yfinance 下载 OHLCV 数据并展平多级列。"""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fred(series_id: str, retries: int = 3) -> pd.Series:
    """从 FRED 公开 CSV 端点拉取宏观序列,失败返回空 Series。"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                txt = r.read().decode()
            df = pd.read_csv(StringIO(txt), parse_dates=[0], na_values=".")
            df.columns = ["date", series_id]
            return df.dropna().set_index("date")[series_id]
        except Exception as e:
            if i == retries - 1:
                print(f"[FRED {series_id}] 拉取失败: {e}")
                return pd.Series(dtype=float, name=series_id)
            import time
            time.sleep(2 ** i)


def cftc_cot(commodity: str = "GOLD", limit: int = 400) -> pd.DataFrame:
    """CFTC Legacy Futures-Only 报告 (每周五发布,数据截至上一个周二)。"""
    # Socrata dataset: 6dca-aqww
    url = (
        "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        f"?$limit={limit}&$order=report_date_as_yyyy_mm_dd%20DESC"
        f"&commodity_name={commodity}"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode())
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        df = df.set_index("date").sort_index()
        # 尝试转换数值列 | Convert numeric columns
        for c in df.columns:
            if any(k in c for k in ["positions", "_all", "open_interest"]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        print(f"[CFTC] 拉取失败: {e}")
        return pd.DataFrame()


print("Setup complete — 数据助手已就绪.")


# %% ============================================================
#    Module 1: 价格与宏观驱动 | Price & Macro Drivers
# ==============================================================
print("\n=== Module 1: Price & Macro Drivers ===")

gold = fetch_yf("GC=F")["Close"].rename("Gold")
dxy = fetch_yf("DX-Y.NYB")["Close"].rename("DXY")
real_yield = fred("DFII10").rename("Real_10Y")       # 10年期 TIPS 实际收益率
cpi = fred("CPIAUCSL").rename("CPI")                 # CPI 指数
breakeven = fred("T10YIE").rename("Breakeven_10Y")   # 10年期盈亏平衡通胀

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("Module 1  黄金 vs 宏观驱动因素 | Gold vs Macro Drivers",
             fontsize=13, fontweight="bold")

# 1a. 黄金 vs 10Y 实际收益率(倒挂显示,便于观察负相关)
ax = axes[0, 0]
ax.plot(gold.index, gold, color=GOLD, lw=1.5, label="Gold")
ax.set_ylabel("Gold ($/oz)", color=GOLD)
ax.tick_params(axis="y", labelcolor=GOLD)
if not real_yield.empty:
    ax2 = ax.twinx()
    ax2.plot(real_yield.index, real_yield, color=BLUE, lw=1.0, alpha=0.75)
    ax2.invert_yaxis()
    ax2.set_ylabel("10Y Real Yield % (inverted)", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
ax.set_title("Gold vs 10Y 实际收益率(Y轴倒置)")

# 1b. 黄金 vs 美元指数
ax = axes[0, 1]
ax.plot(gold.index, gold, color=GOLD, lw=1.5)
ax.set_ylabel("Gold ($/oz)", color=GOLD)
ax.tick_params(axis="y", labelcolor=GOLD)
ax2 = ax.twinx()
ax2.plot(dxy.index, dxy, color=GREEN, lw=1.0, alpha=0.75)
ax2.set_ylabel("DXY", color=GREEN)
ax2.tick_params(axis="y", labelcolor=GREEN)
ax.set_title("Gold vs 美元指数 DXY")

# 1c. 黄金 vs CPI 同比
ax = axes[1, 0]
ax.plot(gold.index, gold, color=GOLD, lw=1.5)
ax.set_ylabel("Gold ($/oz)", color=GOLD)
ax.tick_params(axis="y", labelcolor=GOLD)
if not cpi.empty:
    cpi_yoy = cpi.pct_change(12) * 100
    ax2 = ax.twinx()
    ax2.plot(cpi_yoy.index, cpi_yoy, color=RED, lw=1.3)
    ax2.set_ylabel("US CPI YoY %", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
ax.set_title("Gold vs 美国 CPI 同比")

# 1d. 黄金 vs 10Y 盈亏平衡通胀
ax = axes[1, 1]
ax.plot(gold.index, gold, color=GOLD, lw=1.5)
ax.set_ylabel("Gold ($/oz)", color=GOLD)
ax.tick_params(axis="y", labelcolor=GOLD)
if not breakeven.empty:
    ax2 = ax.twinx()
    ax2.plot(breakeven.index, breakeven, color=PURPLE, lw=1.0)
    ax2.set_ylabel("10Y Breakeven Inflation %", color=PURPLE)
    ax2.tick_params(axis="y", labelcolor=PURPLE)
ax.set_title("Gold vs 10Y 盈亏平衡通胀")

plt.tight_layout()
plt.show()

# 近1年水平相关性 | 1-year level correlation
macro = pd.concat([gold, dxy, real_yield, breakeven], axis=1).ffill().dropna()
if not macro.empty:
    recent = macro.loc[macro.index >= macro.index.max() - pd.Timedelta(days=365)]
    print("\n近1年相关性矩阵 (level correlation, last 365d):")
    print(recent.corr().round(2).to_string())


# %% ============================================================
#    Module 2: 持仓 | Positioning (CFTC + ETF)
# ==============================================================
print("\n=== Module 2: Positioning ===")

cot = cftc_cot("GOLD")
gld = fetch_yf("GLD")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Module 2  持仓数据 | Positioning", fontsize=13, fontweight="bold")

# 2a. CFTC 非商业净多头(管理基金代理)
ax = axes[0]
if not cot.empty and "noncomm_positions_long_all" in cot.columns:
    cot["net_noncomm"] = (
        cot["noncomm_positions_long_all"] - cot["noncomm_positions_short_all"]
    )
    ax.bar(cot.index, cot["net_noncomm"] / 1000, width=5,
           color=BLUE, alpha=0.55, label="Non-Comm Net (k合约)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("非商业净多头 (千张)", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax.twinx()
    gold_cot = gold.reindex(cot.index, method="ffill")
    ax2.plot(gold_cot.index, gold_cot, color=GOLD, lw=1.4, label="Gold")
    ax2.set_ylabel("Gold ($/oz)", color=GOLD)
    ax2.tick_params(axis="y", labelcolor=GOLD)
    ax.set_title("CFTC 非商业净多头 vs 金价 | Non-Commercial Net vs Gold")

    # 打印最新 | Latest snapshot
    latest = cot.iloc[-1]
    print(f"最新 COT 报告日期: {cot.index[-1].date()}")
    print(f"  非商业多头 Non-comm long : {int(latest['noncomm_positions_long_all']):,}")
    print(f"  非商业空头 Non-comm short: {int(latest['noncomm_positions_short_all']):,}")
    print(f"  净多头 Net long          : {int(latest['net_noncomm']):,}")
else:
    ax.text(0.5, 0.5, "CFTC 数据不可用 — 检查网络 / CFTC API 状态",
            ha="center", transform=ax.transAxes)

# 2b. GLD ETF 价格与 20日成交量(持仓量代理)
ax = axes[1]
ax.plot(gld.index, gld["Close"], color=GOLD, lw=1.5)
ax.set_ylabel("GLD Price ($)", color=GOLD)
ax.tick_params(axis="y", labelcolor=GOLD)
ax2 = ax.twinx()
vol20 = gld["Volume"].rolling(20).mean() / 1e6
ax2.fill_between(vol20.index, vol20, alpha=0.3, color=BLUE)
ax2.set_ylabel("20-day avg volume (M shares)", color=BLUE)
ax2.tick_params(axis="y", labelcolor=BLUE)
ax.set_title("SPDR Gold Trust (GLD) 价格与成交量")

plt.tight_layout()
plt.show()

# 提示: 若需要 GLD 实际持仓吨数,可从 SSGA 官网每日 CSV 下载
# Note: For actual GLD tonnage, download the daily CSV from
# https://www.ssga.com/us/en/individual/etfs/gld-gld


# %% ============================================================
#    Module 3: 技术分析 | Technical Analysis
# ==============================================================
print("\n=== Module 3: Technical Analysis ===")


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指数 | Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD: (line, signal, histogram)."""
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


g = gold.to_frame()
g["MA50"] = g["Gold"].rolling(50).mean()
g["MA200"] = g["Gold"].rolling(200).mean()
g["BB_mid"] = g["Gold"].rolling(20).mean()
_bb = g["Gold"].rolling(20).std()
g["BB_up"] = g["BB_mid"] + 2 * _bb
g["BB_lo"] = g["BB_mid"] - 2 * _bb
g["RSI14"] = rsi(g["Gold"], 14)
g["MACD"], g["MACD_sig"], g["MACD_hist"] = macd(g["Gold"])

# 只看最近 24 个月,保持图表清晰
gp = g.loc[g.index >= g.index.max() - pd.Timedelta(days=730)]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1]})
fig.suptitle("Module 3  技术分析 | Technical Analysis  (近 24 个月)",
             fontsize=13, fontweight="bold")

# 3a. 价格 + 均线 + 布林带
ax = axes[0]
ax.plot(gp.index, gp["Gold"], color=GOLD, lw=1.6, label="Gold")
ax.plot(gp.index, gp["MA50"], color=BLUE, lw=1.0, label="MA50")
ax.plot(gp.index, gp["MA200"], color=RED, lw=1.0, label="MA200")
ax.fill_between(gp.index, gp["BB_up"], gp["BB_lo"],
                alpha=0.10, color=PURPLE, label="BB 20,2σ")
ax.set_ylabel("Gold ($/oz)")
ax.set_title("价格 + 均线 + 布林带")
ax.legend(loc="upper left", ncol=4, fontsize=9)

# 3b. RSI(14)
ax = axes[1]
ax.plot(gp.index, gp["RSI14"], color=PURPLE, lw=1.1)
ax.axhline(70, color=RED, ls="--", lw=0.8)
ax.axhline(30, color=GREEN, ls="--", lw=0.8)
ax.axhline(50, color="gray", ls=":", lw=0.5)
ax.set_ylim(0, 100)
ax.set_ylabel("RSI(14)")

# 3c. MACD
ax = axes[2]
ax.bar(gp.index, gp["MACD_hist"], width=1.0,
       color=np.where(gp["MACD_hist"] >= 0, GREEN, RED), alpha=0.55)
ax.plot(gp.index, gp["MACD"], color=BLUE, lw=1.0, label="MACD")
ax.plot(gp.index, gp["MACD_sig"], color=RED, lw=1.0, label="Signal")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("MACD")
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.show()

# 当前快照 | Current technical snapshot
last = g.dropna(subset=["MA200"]).iloc[-1]
price = float(last["Gold"])
ma50, ma200 = float(last["MA50"]), float(last["MA200"])
r = float(last["RSI14"])
mac, sig = float(last["MACD"]), float(last["MACD_sig"])
print(f"\n技术指标快照 (as of {g.index[-1].date()}):")
print(f"  Gold:      ${price:,.2f}")
print(f"  MA50:      ${ma50:,.2f}   ({'↑ above' if price > ma50 else '↓ below'})")
print(f"  MA200:     ${ma200:,.2f}   ({'↑ above' if price > ma200 else '↓ below'})")
print(f"  RSI(14):   {r:.1f}   ("
      f"{'overbought 超买' if r > 70 else 'oversold 超卖' if r < 30 else 'neutral 中性'})")
print(f"  MACD:      {mac:.2f}  signal={sig:.2f}   "
      f"({'bullish 金叉' if mac > sig else 'bearish 死叉'})")


# %% ============================================================
#    Module 4: 跨资产与相关性 | Cross-Asset & Correlations
# ==============================================================
print("\n=== Module 4: Cross-Asset & Correlations ===")

spy = fetch_yf("SPY")["Close"].rename("SPY")
btc = fetch_yf("BTC-USD")["Close"].rename("BTC")
silver = fetch_yf("SI=F")["Close"].rename("Silver")
copper = fetch_yf("HG=F")["Close"].rename("Copper")

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("Module 4  跨资产 | Cross-Asset", fontsize=13, fontweight="bold")

# 4a. 标准化走势(log 坐标)
ax = axes[0, 0]
start_ra = "2020-01-01"
for name, s, c in [("Gold", gold, GOLD), ("SPY", spy, BLUE),
                   ("BTC", btc, "#f7931a"), ("Silver", silver, GRAY)]:
    s2 = s.loc[start_ra:].dropna()
    if not s2.empty:
        ax.plot(s2.index, s2 / s2.iloc[0] * 100, color=c, lw=1.3, label=name)
ax.set_title(f"标准化表现 (基准=100, 起始 {start_ra})  — log scale")
ax.set_ylabel("Rebased")
ax.set_yscale("log")
ax.legend()

# 4b. 金银比
ax = axes[0, 1]
gs = (gold / silver).dropna()
ax.plot(gs.index, gs, color=PURPLE, lw=1.2)
mean_gs, std_gs = gs.mean(), gs.std()
ax.axhline(mean_gs, color="black", ls="--", lw=0.8, label=f"均值 {mean_gs:.0f}")
ax.axhline(mean_gs + std_gs, color=RED, ls=":", lw=0.7, label="+1σ")
ax.axhline(mean_gs - std_gs, color=GREEN, ls=":", lw=0.7, label="-1σ")
ax.set_title(f"金银比 Gold/Silver Ratio  (当前 {float(gs.iloc[-1]):.1f})")
ax.set_ylabel("Ratio")
ax.legend(fontsize=9)

# 4c. Gold 与其他资产的 60 日滚动相关性
ax = axes[1, 0]
window = 60
rets = pd.concat([
    gold.pct_change(),
    spy.pct_change().rename("SPY"),
    btc.pct_change().rename("BTC"),
    silver.pct_change().rename("Silver"),
], axis=1).dropna()
for col, c in [("SPY", BLUE), ("BTC", "#f7931a"), ("Silver", GRAY)]:
    roll = rets["Gold"].rolling(window).corr(rets[col])
    ax.plot(roll.index, roll, lw=1.1, color=c, label=col)
ax.axhline(0, color="black", lw=0.5)
ax.set_title(f"Gold vs 其他资产 {window}-day 滚动相关性")
ax.set_ylabel("Correlation")
ax.legend()

# 4d. 铜金比 — 经济周期 / 风险偏好指标
ax = axes[1, 1]
cg = (copper / gold * 1000).dropna()
ax.plot(cg.index, cg, color=GREEN, lw=1.1)
ax.set_title("铜金比 Copper/Gold (×1000)  — 风险偏好指标")
ax.set_ylabel("Ratio ×1000")

plt.tight_layout()
plt.show()


# %% ============================================================
#    Module 5: 季节性 | Seasonality
# ==============================================================
print("\n=== Module 5: Seasonality ===")

# 月度回报热力图 | Monthly return heatmap
gold_m = gold.resample("ME").last()
gold_mr = gold_m.pct_change() * 100
seas = gold_mr.to_frame("ret").assign(
    year=lambda d: d.index.year, month=lambda d: d.index.month
)
heat = seas.pivot(index="year", columns="month", values="ret")
heat.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
avg_month = heat.mean()

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
fig.suptitle("Module 5  季节性 | Seasonality", fontsize=13, fontweight="bold")

# 5a. 月度回报热力图
ax = axes[0]
vmax = max(abs(heat.min().min()), abs(heat.max().max()))
im = ax.imshow(heat.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(12))
ax.set_xticklabels(heat.columns)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index)
ax.set_title("月度回报 (%)  | Monthly Returns")
plt.colorbar(im, ax=ax, label="%")
for i in range(len(heat.index)):
    for j in range(12):
        v = heat.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > vmax * 0.5 else "black")

# 5b. 平均月度回报柱状图
ax = axes[1]
colors = [GREEN if v >= 0 else RED for v in avg_month.values]
ax.bar(range(1, 13), avg_month.values, color=colors, alpha=0.75)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(heat.columns)
ax.axhline(0, color="black", lw=0.5)
ax.set_title(f"平均月度回报 ({heat.index.min()}–{heat.index.max()})")
ax.set_ylabel("% Return")
for i, v in enumerate(avg_month.values):
    ax.text(i + 1, v + (0.08 if v >= 0 else -0.15), f"{v:.2f}",
            ha="center", fontsize=8)

plt.tight_layout()
plt.show()

# 年内走势对比 | Current-year path vs historical
cur_year = datetime.today().year
gd = gold.to_frame("Gold")
gd["year"] = gd.index.year
gd["doy"] = gd.index.dayofyear

fig, ax = plt.subplots(figsize=(13, 6))

# 过去 5 年的个年路径
for y in range(cur_year - 5, cur_year):
    sub = gd[gd["year"] == y].copy()
    if sub.empty:
        continue
    sub["norm"] = sub["Gold"] / sub["Gold"].iloc[0] * 100
    ax.plot(sub["doy"], sub["norm"], lw=0.9, alpha=0.45, label=str(y))

# 历史平均(所有过去年份)
hist = gd[gd["year"] < cur_year].copy()
if not hist.empty:
    hist["norm"] = hist.groupby("year")["Gold"].transform(lambda x: x / x.iloc[0] * 100)
    avg_path = hist.groupby("doy")["norm"].mean()
    ax.plot(avg_path.index, avg_path.values,
            color="black", lw=2, ls="--", label="历年平均 (all years avg)")

# 当前年份
cur = gd[gd["year"] == cur_year].copy()
if not cur.empty:
    cur["norm"] = cur["Gold"] / cur["Gold"].iloc[0] * 100
    ax.plot(cur["doy"], cur["norm"], lw=2.8, color=GOLD,
            label=f"{cur_year} (今年 YTD)")

ax.set_title("黄金年内走势对比 | Gold YTD Path vs Historical",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Day of Year")
ax.set_ylabel("Rebased (start = 100)")
ax.legend(ncol=4, fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% ============================================================
#    Summary Dashboard — 一屏汇总 | One-screen snapshot
# ==============================================================
print("\n" + "=" * 60)
print("  GOLD RESEARCH PANEL — SNAPSHOT")
print("=" * 60)

ret_1m = (float(gold.iloc[-1]) / float(gold.iloc[-21]) - 1) * 100 if len(gold) > 21 else np.nan
ret_3m = (float(gold.iloc[-1]) / float(gold.iloc[-63]) - 1) * 100 if len(gold) > 63 else np.nan
ret_1y = (float(gold.iloc[-1]) / float(gold.iloc[-252]) - 1) * 100 if len(gold) > 252 else np.nan
ytd_start = gold[gold.index.year == cur_year]
ret_ytd = (float(gold.iloc[-1]) / float(ytd_start.iloc[0]) - 1) * 100 if not ytd_start.empty else np.nan

print(f"As of       : {g.index[-1].date()}")
print(f"Gold        : ${price:,.2f}  /oz")
print(f"Returns     : 1M {ret_1m:+.1f}%   3M {ret_3m:+.1f}%   YTD {ret_ytd:+.1f}%   1Y {ret_1y:+.1f}%")
print(f"Trend       : {'Uptrend 上升' if price > ma50 > ma200 else 'Downtrend 下降' if price < ma50 < ma200 else 'Mixed 混合'}")
print(f"RSI(14)     : {r:.1f}")
print(f"MACD        : {'金叉 bullish' if mac > sig else '死叉 bearish'}")
print(f"Gold/Silver : {float(gs.iloc[-1]):.1f}   (mean {mean_gs:.0f}, ±1σ {std_gs:.0f})")
if not cot.empty and "net_noncomm" in cot.columns:
    print(f"CFTC Net    : {int(cot['net_noncomm'].iloc[-1]):+,} contracts "
          f"({cot.index[-1].date()})")
print("=" * 60)
