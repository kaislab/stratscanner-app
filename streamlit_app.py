import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import time

st.set_page_config(page_title="TheStrat Multi-Timeframe Scanner", layout="wide")

# =========================
# Utility & Caching Functions
# =========================

@st.cache_data(ttl=43200)  # cache for 12 hours
def load_universe():
    try:
        # Force UTF-8 encoding, explicitly set the first row as header, and use semicolon as delimiter
        df = pd.read_csv("universe.csv", encoding='utf-8', header=0, sep=';')
        # Check for 'Ticker' column (Sector and Industry are optional)
        if not any(col.strip().lower() == 'ticker' for col in df.columns):
            st.error("Error: 'Ticker' column missing in universe.csv. Please check the file format.")
            return pd.DataFrame(columns=['Ticker'])
        # Rename columns to match expected case if needed
        df.columns = [col.strip() for col in df.columns]
        ticker_col = next(col for col in df.columns if col.lower() == 'ticker')
        df = df.rename(columns={ticker_col: 'Ticker'})
        # Optionally include Sector if present
        if any(col.strip().lower() == 'sector' for col in df.columns):
            sector_col = next(col for col in df.columns if col.lower() == 'sector')
            df = df.rename(columns={sector_col: 'Sector'})
            df['Sector'] = df['Sector'].fillna('Unknown')
        else:
            df['Sector'] = 'Unknown'  # Default value if Sector is missing
        # Optionally include Industry if present
        industry_col = next((col for col in df.columns if col.strip().lower() == 'industry'), None)
        if industry_col:
            df = df.rename(columns={industry_col: 'Industry'})
            df['Industry'] = df['Industry'].fillna('Unknown')
        else:
            df['Industry'] = 'Unknown'  # Default value if Industry is missing
        tickers = df["Ticker"].dropna().unique().tolist()
        return df  # Return full DataFrame to access Ticker, Sector, and Industry
    except Exception as e:
        st.error(f"Error loading universe.csv: {str(e)}. Please check the file exists and is accessible.")
        return pd.DataFrame(columns=['Ticker'])

@st.cache_data(ttl=43200)
def fetch_data_batch(tickers, start_date, end_date, interval):
    """Fetches data in batches of 100 tickers, retries failed ones."""
    all_data = {}
    batch_size = 100
    success, failed = 0, 0
    progress = st.progress(0)
    status_area = st.empty()
    failed_tickers = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        progress.progress(min(1.0, i / len(tickers)))
        status_area.write(f"Fetching batch {i//batch_size + 1} of {len(tickers)//batch_size + 1}...")
        try:
            data = yf.download(batch, start=start_date, end=end_date, interval=interval, group_by='ticker', threads=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    if t in data.columns.get_level_values(0):
                        all_data[t] = data[t].dropna()
                        success += 1
                    else:
                        failed_tickers.append(t)
                        failed += 1
            else:
                all_data[batch[0]] = data.dropna()
                success += 1
        except Exception as e:
            failed_tickers.extend(batch)
            failed += len(batch)

    # Retry failed tickers individually
    missing = [t for t in tickers if t not in all_data]
    for t in missing:
        try:
            df = yf.download(t, start=start_date, end=end_date, interval=interval)
            if not df.empty:
                all_data[t] = df
                success += 1
            else:
                failed_tickers.append(t)
                failed += 1
        except Exception:
            failed_tickers.append(t)
            failed += 1

    progress.progress(1.0)
    status_area.write("Fetch complete ✅")
    return all_data, success, failed, failed_tickers, interval, datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def resample_ohlc(df, timeframe, interval):
    if timeframe == "Daily":
        if interval != "1d":
            # Resample intraday to daily
            return df.resample("1D").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
        return df
    mapping = {"Weekly": "W-FRI", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
    rule = mapping.get(timeframe)
    return df.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()

def classify_strat(row_prev, row_curr):
    if row_curr["High"] <= row_prev["High"] and row_curr["Low"] >= row_prev["Low"]:
        return "1"
    elif row_curr["High"] > row_prev["High"] and row_curr["Low"] >= row_prev["Low"]:
        return "2u"
    elif row_curr["High"] <= row_prev["High"] and row_curr["Low"] < row_prev["Low"]:
        return "2d"
    elif row_curr["High"] > row_prev["High"] and row_curr["Low"] < row_prev["Low"]:
        return "3"
    else:
        return "N/A"  # Should not happen

def get_pattern_with_color(pattern, color):
    if pattern == "N/A":
        return "N/A"
    return f"{pattern}-{color}"

def compute_strat_table(data_dict, end_date, timeframe, interval):
    rows = []
    all_timeframes = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    for ticker, df in data_dict.items():
        try:
            # Compute patterns for the current timeframe
            df_main = resample_ohlc(df, timeframe, interval)
            if len(df_main) < 5:
                row = {"Ticker": ticker, "CC": "N/A", "C1": "N/A", "C2": "N/A", "C3": "N/A"}
                for tf in all_timeframes:
                    row[f"{tf[0]}_Color"] = "N/A"
                rows.append(row)
                continue
            df_main = df_main.tail(5)
            colors = ["Green" if c > o else "Red" for o, c in zip(df_main["Open"], df_main["Close"])]
            patterns = []
            for i in range(len(df_main) - 1):
                patterns.append(classify_strat(df_main.iloc[i], df_main.iloc[i+1]))
            cc = get_pattern_with_color(patterns[-1], colors[-1]) if len(patterns) >= 1 else "N/A"
            c1 = get_pattern_with_color(patterns[-2], colors[-2]) if len(patterns) >= 2 else "N/A"
            c2 = get_pattern_with_color(patterns[-3], colors[-3]) if len(patterns) >= 3 else "N/A"
            c3 = get_pattern_with_color(patterns[-4], colors[-4]) if len(patterns) >= 4 else "N/A"

            # Compute colors for all timeframes
            row = {"Ticker": ticker, "CC": cc, "C1": c1, "C2": c2, "C3": c3}
            now = datetime.datetime.now()
            is_trading_hours = (now.hour >= 14 and now.hour < 21) and now.weekday() < 5  # Approx NYSE hours in CEST
            for tf in all_timeframes:
                df_tf = resample_ohlc(df, tf, interval)
                if tf in ["Weekly", "Monthly"] and not is_trading_hours:
                    if tf == "Weekly":
                        df_tf = df_tf[df_tf.index <= end_date - pd.offsets.Week(weekday=4)]  # Last Friday
                    else:  # Monthly
                        df_tf = df_tf[df_tf.index <= end_date - pd.offsets.MonthEnd()]
                    df_tf = df_tf.iloc[-1:]
                else:
                    df_tf = df_tf.tail(1)
                if df_tf.empty or pd.isna(df_tf["Open"].iloc[0]) or pd.isna(df_tf["Close"].iloc[0]):
                    row[f"{tf[0]}_Color"] = "N/A"
                else:
                    row[f"{tf[0]}_Color"] = "🟩" if df_tf["Close"].iloc[0] > df_tf["Open"].iloc[0] else "🟥"
            rows.append(row)
        except Exception:
            # Ensure 'Ticker' is included even on error
            row = {"Ticker": ticker, "CC": "N/A", "C1": "N/A", "C2": "N/A", "C3": "N/A"}
            for tf in all_timeframes:
                row[f"{tf[0]}_Color"] = "N/A"
            rows.append(row)
    return pd.DataFrame(rows)

def compute_sector_table(universe_df, data_dict, end_date, timeframe, interval):
    # Aggregate by sector from universe.csv
    sector_counts = universe_df.groupby("Sector").size().to_dict()
    rows = []
    for sector, count in sector_counts.items():
        sector_tickers = universe_df[universe_df["Sector"] == sector]["Ticker"].tolist()
        two_up = 0
        two_down = 0
        total_valid = 0
        two_up_green = 0
        two_up_total = 0
        two_down_green = 0
        two_down_total = 0

        for ticker in sector_tickers:
            if ticker in data_dict and len(data_dict[ticker]) >= 5:
                df_main = resample_ohlc(data_dict[ticker], timeframe, interval).tail(5)
                patterns = [classify_strat(df_main.iloc[i], df_main.iloc[i+1]) for i in range(len(df_main) - 1)]
                cc = patterns[-1] if patterns else "N/A"
                if cc == "2u":
                    two_up += 1
                    candle_colors = ["Green" if c > o else "Red" for o, c in zip(df_main["Open"], df_main["Close"])]
                    two_up_green += sum(1 for color in candle_colors if color == "Green")
                    two_up_total += len(candle_colors)
                elif cc == "2d":
                    two_down += 1
                    candle_colors = ["Green" if c > o else "Red" for o, c in zip(df_main["Open"], df_main["Close"])]
                    two_down_green += sum(1 for color in candle_colors if color == "Green")
                    two_down_total += len(candle_colors)
                total_valid += 1

        two_up_pct = (two_up / total_valid * 100) if total_valid > 0 else 0
        two_down_pct = (two_down / total_valid * 100) if total_valid > 0 else 0
        two_up_green_pct = (two_up_green / two_up_total * 100) if two_up_total > 0 else 0
        two_down_green_pct = (two_down_green / two_down_total * 100) if two_down_total > 0 else 0

        rows.append({
            "Sector": sector,
            "Holdings": count,
            "2u %": two_up_pct,
            "2d %": two_down_pct,
            "2u Green %": two_up_green_pct,
            "2d Green %": two_down_green_pct
        })
    if any(sector == "Unknown" for sector in sector_counts):
        st.warning("Some tickers have no sector assigned and are grouped as 'Unknown'. Consider updating universe.csv.")
    df = pd.DataFrame(rows)
    # Convert percentages to numeric for sorting
    df['2u %'] = df['2u %'].astype(float)
    df['2d %'] = df['2d %'].astype(float)
    df['2u Green %'] = df['2u Green %'].astype(float)
    df['2d Green %'] = df['2d Green %'].astype(float)
    return df

def compute_industry_table(universe_df, data_dict, sector, end_date, timeframe, interval):
    # Aggregate by industry within the specified sector
    industry_data = universe_df[universe_df["Sector"] == sector]
    industry_counts = industry_data.groupby("Industry").size().to_dict()
    rows = []
    for industry, count in industry_counts.items():
        industry_tickers = industry_data[industry_data["Industry"] == industry]["Ticker"].tolist()
        two_up = 0
        two_down = 0
        total_valid = 0

        for ticker in industry_tickers:
            if ticker in data_dict and len(data_dict[ticker]) >= 5:
                df_main = resample_ohlc(data_dict[ticker], timeframe, interval).tail(5)
                patterns = [classify_strat(df_main.iloc[i], df_main.iloc[i+1]) for i in range(len(df_main) - 1)]
                cc = patterns[-1] if patterns else "N/A"
                if cc == "2u":
                    two_up += 1
                elif cc == "2d":
                    two_down += 1
                total_valid += 1

        two_up_pct = (two_up / total_valid * 100) if total_valid > 0 else 0
        two_down_pct = (two_down / total_valid * 100) if total_valid > 0 else 0

        rows.append({
            "Industry": industry,
            "Holdings": count,
            "2u %": two_up_pct,
            "2d %": two_down_pct
        })
    df = pd.DataFrame(rows)
    # Convert percentages to numeric for sorting
    df['2u %'] = df['2u %'].astype(float)
    df['2d %'] = df['2d %'].astype(float)
    return df

# =========================
# Streamlit UI
# =========================

st.title("📊 TheStrat Multi-Timeframe Scanner")
st.caption("Built with yfinance data — includes caching, backtesting, and diagnostics")

universe_df = load_universe()  # Load full DataFrame with Sector and Industry columns
tickers = universe_df["Ticker"].dropna().unique().tolist()
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode:", ["⚡ Fast (Cached)", "🔄 Full Refresh"])
backtest_date = st.sidebar.date_input("Select End Date:", datetime.date.today())
live_mode = st.sidebar.checkbox("Use Live 5-Min Data (only for Daily)", value=False)
ticker_search = st.sidebar.text_input("Drill Down Tickers (space-separated, e.g., AAPL MSFT GOOGL)", "")

if st.sidebar.button("Run Scan"):
    today = datetime.date.today()
    is_today = backtest_date == today
    interval = "5m" if live_mode and is_today else "1d"
    end_date_dt = datetime.datetime.combine(backtest_date, datetime.datetime.min.time()) + datetime.timedelta(days=1)  # Include the end date
    if interval == "5m":
        start_date = end_date_dt - datetime.timedelta(days=2)  # Last 2 days for intraday
    else:
        start_date = end_date_dt - datetime.timedelta(days=365 * 6)  # ~6 years for yearly

    with st.spinner("Fetching data..."):
        if mode == "⚡ Fast (Cached)":
            cached = st.session_state.get("cached_data")
            if cached and cached[4] == interval:  # Check if interval matches cached
                data_dict, last_success, last_fail, failed_tickers, interval_cached, last_updated = cached
            else:
                data_dict, last_success, last_fail, failed_tickers, interval_cached, last_updated = fetch_data_batch(tickers, start_date.date(), end_date_dt.date(), interval)
                st.session_state["cached_data"] = (data_dict, last_success, last_fail, failed_tickers, interval, last_updated)
        else:
            data_dict, last_success, last_fail, failed_tickers, interval_cached, last_updated = fetch_data_batch(tickers, start_date.date(), end_date_dt.date(), interval)
            st.session_state["cached_data"] = (data_dict, last_success, last_fail, failed_tickers, interval, last_updated)

    st.session_state["data_dict"] = data_dict
    st.session_state["last_success"] = last_success
    st.session_state["last_fail"] = last_fail
    st.session_state["failed_tickers"] = failed_tickers
    st.session_state["last_updated"] = last_updated
    st.session_state["end_date_dt"] = end_date_dt
    st.session_state["interval"] = interval
    st.success(f"✅ Scan complete! {last_success} with data, {last_fail} missing — Last updated {last_updated}")
    if failed_tickers:
        with st.expander("Failed/Invalid Tickers"):
            st.write(", ".join(failed_tickers))

# Check if data is available in session state to display tabs
if "data_dict" in st.session_state:
    tabs = st.tabs(["Daily", "Weekly", "Monthly", "Quarterly", "Yearly", "Sectors"])
    timeframes = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    
    # Initialize or update strat tables if not present or if settings changed
    if "strat_tables" not in st.session_state:
        st.session_state["strat_tables"] = {}
    for tf in timeframes:
        if tf not in st.session_state["strat_tables"]:
            st.session_state["strat_tables"][tf] = compute_strat_table(st.session_state["data_dict"], st.session_state["end_date_dt"], tf, st.session_state["interval"])

    # Process ticker search
    search_tickers = [t.strip().upper() for t in ticker_search.split() if t.strip()] if ticker_search else []

    for tab, tf in zip(tabs[:-1], timeframes):  # Exclude "Sectors" from timeframe loop
        with tab:
            st.subheader(f"{tf} Timeframe")
            df_strat = st.session_state["strat_tables"][tf].copy()
            if f"{tf}_filters" not in st.session_state:
                st.session_state[f"{tf}_filters"] = {"Ticker": [], "CC": [], "C1": [], "C2": [], "C3": [], "D_Color": [], "W_Color": [], "M_Color": [], "Q_Color": [], "Y_Color": []}
            
            # Apply ticker search filter
            if search_tickers:
                df_strat = df_strat[df_strat["Ticker"].isin(search_tickers)]
            
            # Create filter row above the table
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
            filters = {}
            with col1:
                filters["Ticker"] = st.multiselect("Ticker", options=df_strat["Ticker"].dropna().unique().tolist(), default=st.session_state[f"{tf}_filters"]["Ticker"], key=f"{tf}_Ticker")
            with col2:
                filters["CC"] = st.multiselect("CC", options=df_strat["CC"].dropna().unique().tolist(), default=st.session_state[f"{tf}_filters"]["CC"], key=f"{tf}_CC")
            with col3:
                filters["C1"] = st.multiselect("C1", options=df_strat["C1"].dropna().unique().tolist(), default=st.session_state[f"{tf}_filters"]["C1"], key=f"{tf}_C1")
            with col4:
                filters["C2"] = st.multiselect("C2", options=df_strat["C2"].dropna().unique().tolist(), default=st.session_state[f"{tf}_filters"]["C2"], key=f"{tf}_C2")
            with col5:
                filters["C3"] = st.multiselect("C3", options=df_strat["C3"].dropna().unique().tolist(), default=st.session_state[f"{tf}_filters"]["C3"], key=f"{tf}_C3")
            with col6:
                filters["D_Color"] = st.multiselect("D", options=df_strat["D_Color"].dropna().unique().tolist(), default=st.session_state[f"{tf}_filters"]["D_Color"], key=f"{tf}_D_Color")
            with col7:
                filters["W_Color"] = st.multiselect("W", options=df_strat["W_Color"].dropna().unique().tolist(), default=st.session_state[f"{tf}_filters"]["W_Color"], key=f"{tf}_W_Color")
            with col8:
                filters["M_Color"] = st.multiselect("M", options=df_strat["M_Color"].dropna().unique().tolist(), default=st.session_state[f"{tf}_filters"]["M_Color"], key=f"{tf}_M_Color")
            with col9:
                filters["Q_Color"] = st.multiselect("Q", options=df_strat["Q_Color"].dropna().unique().tolist(), default=st.session_state[f"{tf}_filters"]["Q_Color"], key=f"{tf}_Q_Color")
            with col10:
                filters["Y_Color"] = st.multiselect("Y", options=df_strat["Y_Color"].dropna().unique().tolist(), default=st.session_state[f"{tf}_filters"]["Y_Color"], key=f"{tf}_Y_Color")
            
            st.session_state[f"{tf}_filters"] = filters
            filtered_df = df_strat
            for col, selected in filters.items():
                if selected:
                    filtered_df = filtered_df[filtered_df[col].isin(selected)]
            st.dataframe(filtered_df, use_container_width=True)

    # Sectors tab
    with tabs[-1]:  # "Sectors" tab
        st.subheader("Sector Performance")
        # Add timeframe filter
        timeframe = st.selectbox("Select Timeframe", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], index=0)
        if "sector_table" not in st.session_state or f"sector_table_{timeframe}" not in st.session_state:
            st.session_state[f"sector_table_{timeframe}"] = compute_sector_table(universe_df, st.session_state["data_dict"], st.session_state["end_date_dt"], timeframe, st.session_state["interval"])
        df_sector = st.session_state[f"sector_table_{timeframe}"].copy()
        
        # Summary table and progress bars side by side
        for index, row in df_sector.iterrows():
            cols = st.columns([2, 1, 1, 1, 1, 1])  # Sector, Holdings, 2u %, 2d %, 2u Bar, 2d Bar
            cols[0].write(row['Sector'])
            cols[1].write(row['Holdings'])
            # 2u Section
            cols[2].write("2u Candle Trend")
            cols[2].markdown(
                f"<div style='position: relative;'><span style='position: absolute; top: -20px; left: 50%; transform: translateX(-50%); font-size: 12px;'>{row['2u Green %']:.1f}%</span><div style='background: linear-gradient(to right, #32CD32 {row['2u Green %']}%, #FF4500 {row['2u Green %']}%); height: 15px; width: 100%;'></div></div>",
                unsafe_allow_html=True
            )
            # 2d Section
            cols[3].write("2d Candle Trend")
            cols[3].markdown(
                f"<div style='position: relative;'><span style='position: absolute; top: -20px; left: 50%; transform: translateX(-50%); font-size: 12px;'>{row['2d Green %']:.1f}%</span><div style='background: linear-gradient(to right, #32CD32 {row['2d Green %']}%, #FF4500 {row['2d Green %']}%); height: 15px; width: 100%;'></div></div>",
                unsafe_allow_html=True
            )
            cols[4].empty()  # Placeholder to maintain column structure
            cols[5].empty()  # Placeholder to maintain column structure

        # Detailed table with expanders for industries
        for index, row in df_sector.iterrows():
            with st.expander(f"{row['Sector']} (Holdings: {row['Holdings']}, 2u: {row['2u %']:.1f}%, 2d: {row['2d %']:.1f}%)"):
                df_industry = compute_industry_table(universe_df, st.session_state["data_dict"], row['Sector'], st.session_state["end_date_dt"], timeframe, st.session_state["interval"])
                st.dataframe(df_industry.style.apply(
                    lambda x: ['background-color: #32CD32' if float(x['2u %']) > float(x['2d %']) else 'background-color: #FF4500' if float(x['2u %']) < float(x['2d %']) else '' for _ in x],
                    axis=1
                ).format({"2u %": "{:.1f}%", "2d %": "{:.1f}%"}), use_container_width=True)
else:
    st.info("👈 Select your settings and click **Run Scan** to start.")
