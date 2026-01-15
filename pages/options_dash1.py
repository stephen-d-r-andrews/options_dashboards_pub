import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Options Analysis", page_icon="📊", layout="wide")

st.title("📊 Options Put Analysis Dashboard")
st.markdown("---")

# Sidebar controls
st.sidebar.header("Configuration")

# Tickers input
default_tickers = "NVDA,AMD,CRWV,ORCL,IREN,OWL,AVGO,SNDK,MU,000660.KS,NBIS"
tickers_input = st.sidebar.text_area(
    "Stock Tickers (comma-separated)",
    value=default_tickers,
    help="Enter stock tickers separated by commas"
)
TICKERS = [t.strip() for t in tickers_input.split(",") if t.strip()]

# Other parameters
TARGET_MONTHS = st.sidebar.slider(
    "Target Months",
    min_value=1,
    max_value=60,
    value=36,
    help="Target expiration in months"
)

STRIKE_PCT = st.sidebar.slider(
    "Strike Percentage",
    min_value=0.1,
    max_value=1.0,
    value=0.30,
    step=0.05,
    format="%.2f",
    help="Strike as percentage of spot price"
)

MIN_DTE_DAYS = st.sidebar.number_input(
    "Minimum DTE (days)",
    min_value=0,
    max_value=365,
    value=30,
    help="Ignore expiries too close"
)

PRICE_FIELD = st.sidebar.selectbox(
    "Price Field",
    options=["mid", "last"],
    index=0,
    help="Use bid/ask midpoint or last traded price"
)

RISK_FREE_RATE = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=4.5,
    step=0.1,
    format="%.2f",
    help="Annual risk-free interest rate as a percentage"
) / 100.0  # Convert percentage to decimal

GPU_PRICE_CHANGE_RATE = st.sidebar.number_input(
    "GPU Price Change Rate (% per year)",
    min_value=-50.0,
    max_value=50.0,
    value=-20.0,
    step=1.0,
    format="%.1f",
    help="Annual GPU price change rate (negative for depreciation, positive for appreciation)"
) / 100.0  # Convert percentage to decimal


# -----------------------------
# Helpers
# -----------------------------
def _to_date(expiry_str: str) -> date:
    y, m, d = map(int, expiry_str.split("-"))
    return date(y, m, d)

def choose_expiry_near_months(expiries, target_months: float, min_dte_days: int = 0):
    """
    Pick the listed expiry with DTE closest to target_months (converted to days),
    subject to a minimum DTE filter.
    """
    if not expiries:
        return None, None

    today = date.today()
    target_days = int(round(target_months * 365.25 / 12.0))

    candidates = []
    for e in expiries:
        ed = _to_date(e)
        dte = (ed - today).days
        if dte >= min_dte_days:
            candidates.append((e, dte))

    if not candidates:
        return None, None

    expiry, dte = min(candidates, key=lambda x: abs(x[1] - target_days))
    return expiry, dte

def safe_mid(bid, ask):
    if pd.isna(bid) or pd.isna(ask):
        return np.nan
    if bid is None or ask is None:
        return np.nan
    if bid <= 0 or ask <= 0:
        return np.nan
    return 0.5 * (bid + ask)

def get_spot(t: yf.Ticker):
    # Fast path
    info = getattr(t, "fast_info", None)
    if info and info.get("last_price") is not None:
        return float(info["last_price"])

    # Reliable fallback (usually faster than t.info)
    try:
        h = t.history(period="5d")
        if not h.empty and "Close" in h.columns:
            return float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass

    # Last resort (can be slow / brittle)
    try:
        inf = t.info
        for k in ["regularMarketPrice", "currentPrice", "previousClose"]:
            v = inf.get(k, None)
            if v is not None:
                return float(v)
    except Exception:
        pass

    return np.nan


# -----------------------------
# Main Analysis
# -----------------------------

# Run button
if st.sidebar.button("🔄 Run Analysis", type="primary"):
    st.session_state['run_analysis'] = True

if not st.session_state.get('run_analysis', False):
    st.info("👈 Configure parameters in the sidebar and click 'Run Analysis' to start")
    st.stop()

# Progress tracking
progress_bar = st.progress(0)
status_text = st.empty()

rows = []
today = date.today()

for idx, symbol in enumerate(TICKERS):
    row = {
        "Stock": symbol,
        "Current Spot": np.nan,
        "Selected Expiry": None,
        "DTE (days)": np.nan,
        "Target Strike %": STRIKE_PCT,
        "Forward Value": np.nan,
        "Closest Strike": np.nan,
        "Put Cost": np.nan,
        "Implied Vol": np.nan,
        "Relative % (Put/Spot)": np.nan,
        "Yield % (Put/Strike)": np.nan,
    }

    # Update progress
    progress = (idx + 1) / len(TICKERS)
    progress_bar.progress(progress)
    status_text.text(f"Processing {symbol}... ({idx + 1}/{len(TICKERS)})")

    try:
        t = yf.Ticker(symbol)

        # Spot
        spot = get_spot(t)
        row["Current Spot"] = spot
        if not np.isfinite(spot) or spot <= 0:
            rows.append(row)
            continue

        # Expiry nearest TARGET_MONTHS
        expiries = getattr(t, "options", []) or []
        expiry, dte = choose_expiry_near_months(expiries, TARGET_MONTHS, min_dte_days=MIN_DTE_DAYS)
        row["Selected Expiry"] = expiry
        row["DTE (days)"] = dte
        if not expiry:
            rows.append(row)
            continue

        # Calculate Forward Value first: STRIKE_PCT * spot * e^(rate * time)
        if np.isfinite(spot) and np.isfinite(dte):
            time_years = dte / 365.0
            forward_value = STRIKE_PCT * spot * np.exp(RISK_FREE_RATE * time_years)
            row["Forward Value"] = forward_value
        else:
            rows.append(row)
            continue

        # Option chain
        chain = t.option_chain(expiry)
        puts = chain.puts.copy()
        if puts.empty:
            rows.append(row)
            continue

        # Find strike closest to Forward Value
        puts["strike_diff"] = (puts["strike"] - forward_value).abs()
        best = puts.sort_values(["strike_diff"]).iloc[0]

        strike = float(best["strike"])
        row["Closest Strike"] = strike

        # Put price (mid or last)
        if PRICE_FIELD.lower() == "mid":
            put_cost = safe_mid(best.get("bid", np.nan), best.get("ask", np.nan))
            if not np.isfinite(put_cost) or put_cost <= 0:
                put_cost = float(best.get("lastPrice", np.nan))
        else:
            put_cost = float(best.get("lastPrice", np.nan))

        row["Put Cost"] = put_cost
        row["Implied Vol"] = float(best.get("impliedVolatility", np.nan))

        # Ratios
        if np.isfinite(put_cost) and put_cost > 0 and spot > 0:
            row["Relative % (Put/Spot)"] = put_cost / spot
        if np.isfinite(put_cost) and put_cost > 0 and strike > 0:
            row["Yield % (Put/Strike)"] = put_cost / strike

    except Exception as e:
        row["Error"] = str(e)

    rows.append(row)

# Clear progress indicators
progress_bar.empty()
status_text.empty()

df = pd.DataFrame(rows)

st.success(f"✅ Analysis complete! Processed {len(TICKERS)} tickers")

# -----------------------------
# Display Results
# -----------------------------
st.markdown("---")
st.header("Results")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    valid_data = df.dropna(subset=["Put Cost"])
    st.metric("Valid Results", len(valid_data))
with col2:
    if not valid_data.empty:
        avg_yield = valid_data["Yield % (Put/Strike)"].mean() * 100
        st.metric("Avg Yield %", f"{avg_yield:.2f}%")
with col3:
    if not valid_data.empty:
        avg_iv = valid_data["Implied Vol"].mean() * 100
        st.metric("Avg Implied Vol", f"{avg_iv:.2f}%")
with col4:
    if not valid_data.empty:
        avg_dte = valid_data["DTE (days)"].mean()
        st.metric("Avg DTE", f"{avg_dte:.0f} days")

# -----------------------------
# Scatter Plot: Yield vs Implied Vol + best-fit line
# -----------------------------
st.subheader("📈 Put Yield vs Implied Volatility")

plot_df = df.dropna(subset=["Implied Vol", "Yield % (Put/Strike)"])

if not plot_df.empty:
    x = plot_df["Implied Vol"].values
    y = plot_df["Yield % (Put/Strike)"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.6, s=100)

    # Add ticker labels to points
    for i, ticker in enumerate(plot_df["Stock"]):
        ax.annotate(ticker, (x[i], y[i]), fontsize=8, alpha=0.7)

    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = m * x_fit + b
        ax.plot(x_fit, y_fit, 'r--', alpha=0.8, label=f'Fit: y = {m:.3f}x + {b:.3f}')
        ax.legend()

    ax.set_xlabel("Implied Volatility", fontsize=12)
    ax.set_ylabel("Yield (Put / Strike)", fontsize=12)
    ax.set_title("Put Yield vs Implied Volatility", fontsize=14)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
else:
    st.warning("No valid data points for plotting")

# -----------------------------
# Display formatted dataframe
# -----------------------------
st.subheader("📋 Detailed Results")

df_out = df.copy()

money_cols = ["Current Spot", "Forward Value", "Closest Strike", "Put Cost"]
pct_cols = ["Target Strike %", "Relative % (Put/Spot)", "Yield % (Put/Strike)", "Implied Vol"]

for col in money_cols:
    if col in df_out.columns:
        df_out[col] = df_out[col].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "")

for col in pct_cols:
    if col in df_out.columns:
        df_out[col] = df_out[col].map(lambda v: f"{100*v:.2f}%" if pd.notna(v) else "")

st.dataframe(df_out, use_container_width=True)

# Download button
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name=f"options_analysis_{today.strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# -----------------------------
# GPU Section
# -----------------------------
st.markdown("---")
st.subheader("🖥️ GPU Pricing Analysis")

gpu_data = {
    "GPU": ["H100SXM", "H100PCIE", "H200", "B200", "MI300X", "RTX5090"],
    "MSRP": [30000, 27500, 35000, 40000, 10000, 1999],
    "Release Date": ["2022-03-22", "2022-03-22", "2023-11-13", "2024-03-18", "2023-12-06", "2025-01-06"]
}

gpu_df = pd.DataFrame(gpu_data)

# Convert release dates to datetime
gpu_df["Release Date"] = pd.to_datetime(gpu_df["Release Date"])

# Calculate days since release
gpu_df["Days Since Release"] = (pd.Timestamp(today) - gpu_df["Release Date"]).dt.days

# Calculate depreciated value from MSRP to today based on days since release
gpu_df["Depreciated Value"] = gpu_df.apply(
    lambda row: row["MSRP"] * np.exp(GPU_PRICE_CHANGE_RATE * (row["Days Since Release"] / 365.0)),
    axis=1
)

# Calculate 30% of depreciated value
gpu_df["30% of Depreciated Value"] = gpu_df["Depreciated Value"] * 0.30

# Format for display
gpu_df_display = gpu_df.copy()
gpu_df_display["Release Date"] = gpu_df_display["Release Date"].dt.strftime("%Y-%m-%d")
gpu_df_display["MSRP"] = gpu_df_display["MSRP"].map(lambda v: f"${v:,.2f}")
gpu_df_display["Depreciated Value"] = gpu_df_display["Depreciated Value"].map(lambda v: f"${v:,.2f}")
gpu_df_display["30% of Depreciated Value"] = gpu_df_display["30% of Depreciated Value"].map(lambda v: f"${v:,.2f}")

st.dataframe(gpu_df_display, use_container_width=True)

# Download button for GPU data
gpu_csv = gpu_df.to_csv(index=False)
st.download_button(
    label="📥 Download GPU CSV",
    data=gpu_csv,
    file_name=f"gpu_pricing_{today.strftime('%Y%m%d')}.csv",
    mime="text/csv"
)
