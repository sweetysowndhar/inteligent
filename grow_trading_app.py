import streamlit as st
import random
from datetime import datetime, timedelta
import pandas as pd
from collections import namedtuple

StockScore = namedtuple('StockScore', ['symbol', 'price', 'score', 'signal', 'confidence', 'reasons'])

DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']


def generate_demo_prices(symbol, days=120):
    base_price = random.uniform(80, 450)
    prices = []
    dates = []
    current = base_price
    trend = random.choice(['up', 'down', 'flat'])
    trend_factor = 0.001 if trend == 'flat' else 0.003 if trend == 'up' else -0.003

    for i in range(days):
        dates.append(datetime.now() - timedelta(days=days - i))
        noise = random.uniform(-0.02, 0.02)
        current *= 1 + trend_factor + noise
        current = max(current, 1.0)
        prices.append(round(current, 2))

    return dates, prices


def calculate_sma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def evaluate_stock(symbol):
    dates, prices = generate_demo_prices(symbol)
    current_price = prices[-1]
    sma20 = calculate_sma(prices, 20)
    sma50 = calculate_sma(prices, 50)
    rsi = calculate_rsi(prices, 14)

    score = 0
    reasons = []

    if sma20 and sma50:
        if current_price > sma20 and sma20 > sma50:
            score += 3
            reasons.append('Price above SMA20 and SMA50')
        elif current_price < sma20 and sma20 < sma50:
            score -= 3
            reasons.append('Price below SMA20 and SMA50')
        elif current_price > sma20:
            score += 1
            reasons.append('Price above SMA20')
        elif current_price < sma20:
            score -= 1
            reasons.append('Price below SMA20')

    if rsi is not None:
        if rsi < 35:
            score += 2
            reasons.append(f'RSI {rsi:.0f} indicates oversold')
        elif rsi > 65:
            score -= 2
            reasons.append(f'RSI {rsi:.0f} indicates overbought')
        else:
            reasons.append(f'RSI {rsi:.0f} is neutral')

    if score >= 4:
        signal = 'STRONG BUY'
    elif score >= 2:
        signal = 'BUY'
    elif score <= -4:
        signal = 'STRONG SELL'
    elif score <= -2:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    confidence = min(0.95, 0.5 + abs(score) * 0.1)
    return StockScore(symbol, current_price, score, signal, confidence, reasons), dates, prices


def main():
    st.set_page_config(page_title='Grow-style Trading App', page_icon='📈', layout='wide')
    st.title('📈 Grow-style Trading App')
    st.write('This app ranks stocks and shows the best buy choice today.')

    st.markdown('---')

    with st.expander('Step 1: Select stocks to evaluate'):
        selected = st.multiselect('Choose symbols:', DEFAULT_SYMBOLS, default=DEFAULT_SYMBOLS)
        custom = st.text_input('Or enter your own symbols separated by commas:', '')

    symbols = selected.copy()
    if custom.strip():
        symbols.extend([s.strip().upper() for s in custom.split(',') if s.strip()])
    symbols = list(dict.fromkeys(symbols))[:12]

    if not symbols:
        st.warning('Please select or enter at least one stock symbol.')
        return

    if st.button('Evaluate Stocks'):
        results = []
        for sym in symbols:
            score, dates, prices = evaluate_stock(sym)
            results.append((score, dates, prices))

        results.sort(key=lambda item: item[0].score, reverse=True)

        best = results[0][0]
        st.subheader('🎯 Best stock to buy today')
        if best.signal in ['STRONG BUY', 'BUY']:
            st.success(f'Buy {best.symbol} now at around ${best.price:.2f}')
            st.write(f'Confidence: **{best.confidence:.0%}**')
            for reason in best.reasons:
                st.write(f'- {reason}')
        else:
            st.info('No clear buy signal today. Hold or watch the market.')
            st.write(f'Best symbol based on ranking: **{best.symbol}**')

        buy_list = [item[0] for item in results if item[0].signal in ['STRONG BUY', 'BUY']]
        sell_list = [item[0] for item in results if item[0].signal in ['STRONG SELL', 'SELL']]

        if buy_list:
            st.subheader('📌 Buy candidates')
            buy_df = pd.DataFrame([{
                'Symbol': item.symbol,
                'Price': f'${item.price:.2f}',
                'Signal': item.signal,
                'Confidence': f'{item.confidence:.0%}',
                'Score': item.score
            } for item in buy_list])
            st.table(buy_df)

        if sell_list:
            st.subheader('🚨 Sell candidates')
            sell_df = pd.DataFrame([{
                'Symbol': item.symbol,
                'Price': f'${item.price:.2f}',
                'Signal': item.signal,
                'Confidence': f'{item.confidence:.0%}',
                'Score': item.score
            } for item in sell_list])
            st.table(sell_df)

        all_df = pd.DataFrame([{
            'Symbol': item[0].symbol,
            'Price': f'${item[0].price:.2f}',
            'Signal': item[0].signal,
            'Confidence': f'{item[0].confidence:.0%}',
            'Score': item[0].score
        } for item in results])
        st.subheader('📋 All ranked stocks')
        st.table(all_df)

        st.markdown('---')
        st.subheader(f'{best.symbol} price history')
        price_data = pd.DataFrame({best.symbol: results[0][2]}, index=results[0][1])
        st.line_chart(price_data)

        st.markdown('**Note:** This is a demo prediction model. Use it as a signal guide, not a guarantee.')


if __name__ == '__main__':
    main()
