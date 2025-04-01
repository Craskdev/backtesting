import ccxt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def fetch_ohlcv(hour_count):
    exchange = ccxt.bybit({'enableRateLimit': True})

    data = exchange.fetch_ohlcv('XRP/USDT', timeframe='1h', limit=hour_count)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df.resample('3H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    return df

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
    return pd.DataFrame(trades), round(balance, 2)

def run_batch_test(df, trade_size, leverage):
    best_pnl = -999999
    best_params = None

    for fast in range(2, 10):
        for slow in range(fast + 1, 16):
            for adx in range(3, 10):
                for thresh in range(20, 40, 5):
                    trades, pnl = run_backtest(df, fast, slow, adx, thresh, trade_size, leverage)
                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_params = (fast, slow, adx, thresh)

    return best_params, best_pnl

# Streamlit App
st.title("EMA + ADX Crypto Backtester")

with st.sidebar:
    st.header("Backtest Settings")
    hours = st.number_input("Backtest Hours (x3):", min_value=6, max_value=1000, value=120, step=3)
    fast_ema = st.number_input("Fast EMA:", min_value=2, value=4)
    slow_ema = st.number_input("Slow EMA:", min_value=3, value=8)
    adx_len = st.number_input("ADX Length:", min_value=1, value=5)
    adx_thresh = st.number_input("ADX Threshold:", min_value=5, max_value=50, value=27)
    trade_size = st.number_input("Trade Size ($):", min_value=10, value=1000)
    leverage = st.number_input("Leverage:", min_value=1.0, value=5.0, step=0.5)

col1, col2 = st.columns(2)
with col1:
    if st.button("Run Single Backtest"):
        with st.spinner("Running single test..."):
            df = fetch_ohlcv(hours)
            trades_df, total_pnl = run_backtest(df, fast_ema, slow_ema, adx_len, adx_thresh, trade_size, leverage)

        if trades_df.empty:
            st.warning("No trades triggered.")
        else:
            st.success(f"Total PnL: ${total_pnl:.2f} from {len(trades_df)} trades")
            colors = ['green' if x > 0 else 'red' for x in trades_df['pnl']]
            fig, ax = plt.subplots()
            trades_df.plot(x='entry_time', y='pnl', kind='bar', color=colors, ax=ax, legend=False)
            st.pyplot(fig)

with col2:
    if st.button("Run Batch Optimization"):
        with st.spinner("Running parameter optimization..."):
            df = fetch_ohlcv(hours)
            best_params, best_pnl = run_batch_test(df, trade_size, leverage)

        if best_params:
            st.success(f"Best PnL: ${best_pnl:.2f}")
            st.info(f"Best Parameters:\nFast EMA: {best_params[0]}\nSlow EMA: {best_params[1]}\nADX Len: {best_params[2]}\nADX Threshold: {best_params[3]}")
