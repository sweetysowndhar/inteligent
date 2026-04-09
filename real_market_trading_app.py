import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from collections import namedtuple

StockScore = namedtuple('StockScore', ['symbol', 'price', 'score', 'signal', 'confidence', 'reasons'])
DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'INTC', 'AMD', 'IBM']
COMMODITY_ALIASES = {
    'SILVER': 'SI=F',
    'NIPPON SILVER': 'SI=F',
    'SILVER FUTURES': 'SI=F',
    'GOLD': 'GC=F',
    'GOLD FUTURES': 'GC=F',
    'CRUDE OIL': 'CL=F',
    'NSE SILVER': 'SI=F',
}


def resolve_symbol(symbol):
    key = symbol.strip().upper()
    return COMMODITY_ALIASES.get(key, key)


def fetch_stock_data(symbol, period='120d'):
    try:
        df = yf.download(symbol, period=period, progress=False)
        if df.empty:
            return None
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except Exception as e:
        return None


def calculate_sma(prices, period):
    if len(prices) < period:
        return None
    return prices.rolling(period).mean().iloc[-1]


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    delta = prices.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def score_stock(symbol):
    df = fetch_stock_data(symbol)
    if df is None:
        return None

    close = df['Close'].iloc[-1]
    sma20 = calculate_sma(df['Close'], 20)
    sma50 = calculate_sma(df['Close'], 50)
    rsi = calculate_rsi(df['Close'], 14)

    score = 0
    reasons = []

    if sma20 is not None and sma50 is not None:
        if close > sma20 and sma20 > sma50:
            score += 4
            reasons.append('Bullish SMA crossover: close > SMA20 > SMA50')
        elif close > sma20:
            score += 1
            reasons.append('Price above SMA20')
        elif close < sma20 and sma20 < sma50:
            score -= 4
            reasons.append('Bearish SMA: close < SMA20 < SMA50')
        elif close < sma20:
            score -= 1
            reasons.append('Price below SMA20')

    if rsi is not None:
        if rsi < 35:
            score += 3
            reasons.append(f'RSI {rsi:.1f} indicates oversold')
        elif rsi > 70:
            score -= 3
            reasons.append(f'RSI {rsi:.1f} indicates overbought')
        else:
            reasons.append(f'RSI {rsi:.1f} is neutral')

    if score >= 5:
        signal = 'STRONG BUY'
    elif score >= 2:
        signal = 'BUY'
    elif score <= -5:
        signal = 'STRONG SELL'
    elif score <= -2:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    confidence = min(0.95, 0.50 + abs(score) * 0.08)
    return StockScore(symbol, close, score, signal, confidence, reasons), df


def main():
    st.set_page_config(page_title='Real Market Trading App', page_icon='📈', layout='wide')
    st.title('📈 Real Market Trading App')
    st.write('Fetch real stock data from Yahoo Finance and show today’s purchase recommendation.')
    st.markdown('---')

    st.sidebar.header('Watchlist & Settings')
    symbols = st.sidebar.multiselect('Choose symbols to evaluate:', DEFAULT_SYMBOLS, default=DEFAULT_SYMBOLS[:6])
    extra = st.sidebar.text_input('Add custom symbols or names (comma separated):')
    if extra.strip():
        more = [s.strip() for s in extra.split(',') if s.strip()]
        symbols.extend(more)

    resolved_symbols = []
    alias_notes = []
    for sym in symbols:
        resolved = resolve_symbol(sym)
        resolved_symbols.append(resolved)
        if resolved != sym.strip().upper():
            alias_notes.append(f'{sym} → {resolved}')

    symbols = list(dict.fromkeys(resolved_symbols))[:12]

    if alias_notes:
        st.sidebar.info('Symbol mapping: ' + ', '.join(alias_notes))

    st.sidebar.markdown('**Examples:** `AAPL`, `GOOGL`, `MSFT`, `SI=F` (Silver), `GC=F` (Gold), `NSE:RELIANCE`')

    if not symbols:
        st.warning('Select at least one stock symbol in the sidebar.')
        return

    if st.button('Evaluate Now'):
        with st.spinner('Fetching live market data...'):
            results = []
            for symbol in symbols:
                out = score_stock(symbol)
                if out:
                    results.append(out)

        if not results:
            st.error('Unable to fetch data for any selected stock. Please check the symbols.')
            return

        results.sort(key=lambda item: item[0].score, reverse=True)
        best, best_df = results[0]

        st.subheader('🎯 Best stock to buy today')
        if best.signal in ['STRONG BUY', 'BUY']:
            st.success(f'Buy {best.symbol} now at around ${best.price:.2f}')
            st.write(f'Confidence: **{best.confidence:.0%}**')
            st.write('Reasons:')
            for reason in best.reasons:
                st.write(f'- {reason}')
        else:
            st.info('No strong buy recommendation today. The market is not favorable for purchase.')
            st.write(f'Best candidate: **{best.symbol}** with signal **{best.signal}**')
            for reason in best.reasons:
                st.write(f'- {reason}')

        buy_candidates = [item[0] for item in results if item[0].signal in ['STRONG BUY', 'BUY']]
        sell_candidates = [item[0] for item in results if item[0].signal in ['STRONG SELL', 'SELL']]

        if buy_candidates:
            st.subheader('📌 Buy candidates')
            buy_df = pd.DataFrame([{
                'Symbol': item.symbol,
                'Price': f'${item.price:.2f}',
                'Signal': item.signal,
                'Confidence': f'{item.confidence:.0%}',
                'Score': item.score
            } for item in buy_candidates])
            st.table(buy_df)

        if sell_candidates:
            st.subheader('🚨 Sell candidates')
            sell_df = pd.DataFrame([{
                'Symbol': item.symbol,
                'Price': f'${item.price:.2f}',
                'Signal': item.signal,
                'Confidence': f'{item.confidence:.0%}',
                'Score': item.score
            } for item in sell_candidates])
            st.table(sell_df)

        st.subheader('📋 All ranked stocks')
        all_df = pd.DataFrame([{
            'Symbol': item[0].symbol,
            'Price': f'${item[0].price:.2f}',
            'Signal': item[0].signal,
            'Confidence': f'{item[0].confidence:.0%}',
            'Score': item[0].score
        } for item in results])
        st.table(all_df)

        st.subheader(f'📈 {best.symbol} last 60 days price')
        history = best_df['Close'].tail(60)
        st.line_chart(history)

        st.markdown('---')
        st.caption('Real market data from Yahoo Finance. This app provides technical buy/sell signals for guidance, not a guaranteed profit.')
    else:
        st.info('Press Evaluate Now to fetch real market data and get today’s buy recommendation.')


if __name__ == '__main__':
    main()
