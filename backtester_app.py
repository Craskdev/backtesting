import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta

cg = CoinGeckoAPI()

SYMBOL_MAP = {
    'XRP/USDT': 'ripple',
    'BTC/USDT': 'bitcoin',
    'ETH/USDT': 'ethereum',
}

TIMEFRAME_MAP = {
    '1H': '1H',
    '3H': '3H',
    '6H': '6H',
    '12H': '12H',
    '24H': '24H'
}

def fetch_ohlcv(symbol_id, hours, resample_tf='3H'):
    try:
        coin_id = SYMBOL_MAP[symbol_id]
        to_ts = int(datetime.now().timestamp())
        from_ts = int((datetime.now() - timedelta(hours=hours)).timestamp())

        data = cg.get_coin_market_chart_range_by_id(
            id=coin_id,
            vs_currency='usd',
            from_timestamp=from_ts,
            to_timestamp=to_ts
        )

        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').resample(resample_tf).ohlc()
        df.columns = ['open', 'high', 'low', 'close']
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df['volume'] = 1
        df['datetime'] = df['timestamp'].dt.tz_localize('UTC')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_indicators(df, fast_ema, slow_ema, adx_len):
    df['ema_fast'] = df['close'].ewm(span=fast_ema).mean()
    df['ema_slow'] = df['close'].ewm(span=slow_ema).mean()
    df['tr'] = df['high'] - df['low']
    df['plus_dm'] = df['high'].diff()
    df['minus_dm'] = df['low'].diff()
    df['plus_dm'] = df.apply(lambda r: r['plus_dm'] if r['plus_dm'] > r['minus_dm'] and r['plus_dm'] > 0 else 0, axis=1)
    df['minus_dm'] = df.apply(lambda r: r['minus_dm'] if r['minus_dm'] > r['plus_dm'] and r['minus_dm'] > 0 else 0, axis=1)
    atr = df['tr'].rolling(window=adx_len).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=adx_len).sum() / atr)
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=adx_len).sum() / atr)
    df['dx'] = (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
    df['adx'] = df['dx'].rolling(window=adx_len).mean()
    return df

def run_backtest(df, fast_ema, slow_ema, adx_len, adx_thresh, trade_size, leverage):
    df = calculate_indicators(df.copy(), fast_ema, slow_ema, adx_len)
    last_signal = None
    entry_price = None
    position = None
    trades = []
    balance = 0.0

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        adx_ok = row['adx'] >= adx_thresh
        golden = prev['ema_fast'] < prev['ema_slow'] and row['ema_fast'] > row['ema_slow']
        death = prev['ema_fast'] > prev['ema_slow'] and row['ema_fast'] < row['ema_slow']
        signal = None
        if golden and adx_ok and last_signal != 'long':
            signal = 'long'
            last_signal = 'long'
        elif death and adx_ok and last_signal != 'short':
            signal = 'short'
            last_signal = 'short'

        if signal:
            price = row['close']
            if position is None:
                entry_price = price
                position = signal
            elif position != signal:
                exit_price = price
                raw_return = (exit_price - entry_price) if position == 'long' else (entry_price - exit_price)
                pnl = (raw_return / entry_price) * trade_size * leverage
                trades.append({
                    'entry_time': row['datetime'],
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': round(pnl, 2)
                })
                entry_price = price
                position = signal
                balance += pnl

    if trades:
        pnl_series = pd.Series([t['pnl'] for t in trades])
        equity = pnl_series.cumsum()
        for idx, eq in enumerate(equity):
            trades[idx]['equity'] = eq
    else:
        trades = []

    return pd.DataFrame(trades), round(balance, 2)

def run_batch(df, trade_size, leverage):
    best_result = None
    best_pnl = float('-inf')
    summary = []

    for fast in range(2, 10):
        for slow in range(fast + 1, 16):
            for adx in range(3, 10):
                for thresh in range(20, 40, 5):
                    _, pnl = run_backtest(df, fast, slow, adx, thresh, trade_size, leverage)
                    summary.append({
                        'Fast EMA': fast,
                        'Slow EMA': slow,
                        'ADX Len': adx,
                        'ADX Threshold': thresh,
                        'PnL': round(pnl, 2)
                    })
                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_result = (fast, slow, adx, thresh, pnl)

    return best_result, pd.DataFrame(summary)

# === Streamlit GUI ===
st.title("📊 EMA + ADX Backtester with Optimization, Timeframes & CSV Export")

symbol = st.selectbox("Symbol", list(SYMBOL_MAP.keys()), index=0)
hours = st.slider("Backtest Hours", min_value=24, max_value=720, step=3, value=120)
fast_ema = st.number_input("Fast EMA", value=4, min_value=2)
slow_ema = st.number_input("Slow EMA", value=8, min_value=3)
adx_len = st.number_input("ADX Length", value=5, min_value=1)
adx_thresh = st.number_input("ADX Threshold", value=27, min_value=5, max_value=50)
trade_size = st.number_input("Trade Size ($)", value=1000, min_value=10)
leverage = st.number_input("Leverage (x)", value=5.0, min_value=1.0, step=0.5)

col1, col2 = st.columns(2)

with col1:
    if st.button("Run Single Backtest"):
        df = fetch_ohlcv(symbol, hours)
        if df.empty:
            st.stop()
        trades_df, total_pnl = run_backtest(df, fast_ema, slow_ema, adx_len, adx_thresh, trade_size, leverage)

        if trades_df.empty:
            st.warning("No trades executed.")
        else:
            st.success(f"Total PnL: ${total_pnl:.2f} from {len(trades_df)} trades")
            colors = ['green' if x > 0 else 'red' for x in trades_df['pnl']]
            fig, ax = plt.subplots()
            trades_df.plot(x='entry_time', y='pnl', kind='bar', color=colors, ax=ax, legend=False)
            st.pyplot(fig)

            st.subheader("Equity Curve")
            fig2, ax2 = plt.subplots()
            trades_df['equity'] = trades_df['equity'].fillna(method='ffill')
            trades_df.plot(x='entry_time', y='equity', ax=ax2, legend=False)
            st.pyplot(fig2)

            csv = trades_df.to_csv(index=False).encode()
            st.download_button("📥 Download Trades CSV", data=csv, file_name="single_backtest_trades.csv", mime="text/csv")

with col2:
    use_custom_tf = st.checkbox("Use Different Timeframe for Batch")
    tf = '3H'
    if use_custom_tf:
        tf = st.selectbox("Timeframe (Batch Only)", list(TIMEFRAME_MAP.keys()), index=1)

    if st.button("Run Batch Optimization"):
        df = fetch_ohlcv(symbol, hours, resample_tf=tf)
        if df.empty:
            st.stop()
        result, batch_df = run_batch(df, trade_size, leverage)
        if result:
            f, s, a, t, p = result
            st.success(f"Best PnL: ${p:.2f}")
            st.info(f"Fast EMA: {f}, Slow EMA: {s}, ADX Len: {a}, Threshold: {t}")

            batch_csv = batch_df.to_csv(index=False).encode()
            st.download_button("📥 Download Batch Results CSV", data=batch_csv, file_name="batch_optimization_results.csv", mime="text/csv")
        else:
            st.warning("No profitable combination found.")
