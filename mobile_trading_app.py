import streamlit as st
import random
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict

# Set page config for mobile
st.set_page_config(
    page_title="Stock Trading App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile optimization
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .signal-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .buy-signal { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .sell-signal { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    .hold-signal { background-color: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .trade-history {
        max-height: 300px;
        overflow-y: auto;
    }
    @media (max-width: 768px) {
        .main-header { font-size: 1.5rem; }
        .signal-box { font-size: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

class StockAnalyzer:
    """Analyzes stocks and generates trading signals using demo data"""

    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.data = None
        self.trades = []
        self.portfolio = defaultdict(lambda: {'quantity': 0, 'avg_cost': 0})

    def fetch_data(self):
        """Generate demo stock data for testing"""
        try:
            # Generate 100 days of demo data
            base_price = random.uniform(50, 500)
            dates = []
            closes = []
            opens = []
            highs = []
            lows = []
            volumes = []

            current_price = base_price

            for i in range(100):
                change_percent = random.uniform(-0.05, 0.05)
                current_price *= (1 + change_percent)
                current_price = max(current_price, 1.0)

                volatility = current_price * 0.02
                daily_open = current_price
                daily_close = current_price * (1 + random.uniform(-0.02, 0.02))
                daily_high = max(daily_open, daily_close) + random.uniform(0, volatility)
                daily_low = min(daily_open, daily_close) - random.uniform(0, volatility)
                volume = random.randint(100000, 10000000)

                dates.append(datetime.now() - timedelta(days=99-i))
                opens.append(round(daily_open, 2))
                highs.append(round(daily_high, 2))
                lows.append(round(daily_low, 2))
                closes.append(round(daily_close, 2))
                volumes.append(volume)

                current_price = daily_close

            self.data = {
                'dates': dates,
                'closes': closes,
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'volumes': volumes
            }
            return True
        except Exception as e:
            st.error(f"Error generating data: {e}")
            return False

    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return None

        deltas = []
        for i in range(1, len(prices)):
            deltas.append(prices[i] - prices[i-1])

        gains = [d for d in deltas[-period:] if d > 0]
        losses = [-d for d in deltas[-period:] if d < 0]

        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0

        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_sma(prices, period):
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    def generate_signal(self):
        """Generate BUY/SELL/HOLD signal based on technical analysis"""
        if not self.data or len(self.data['closes']) < 50:
            return {'action': 'HOLD', 'reasons': ['Insufficient data'], 'price': 0, 'strength': 0}

        closes = self.data['closes']
        current_price = closes[-1]

        sma20 = self._calculate_sma(closes, 20)
        sma50 = self._calculate_sma(closes, 50)
        rsi = self._calculate_rsi(closes, 14)

        signal = {
            'action': 'HOLD',
            'reasons': [],
            'price': current_price,
            'strength': 0,
            'rsi': rsi,
            'sma20': sma20,
            'sma50': sma50
        }

        if not all([sma20, sma50, rsi]):
            return signal

        # SMA Strategy
        if current_price > sma20 and sma20 > sma50:
            signal['reasons'].append("📈 Bullish: Price above both SMA20 and SMA50")
            signal['strength'] += 2
        elif current_price < sma20 and sma20 < sma50:
            signal['reasons'].append("📉 Bearish: Price below both SMA20 and SMA50")
            signal['strength'] -= 2

        # RSI Strategy
        if rsi < 30:
            signal['reasons'].append(f"💰 Oversold (RSI: {rsi:.1f}) - Buy Signal")
            signal['strength'] += 2
        elif rsi > 70:
            signal['reasons'].append(f"⚠️ Overbought (RSI: {rsi:.1f}) - Sell Signal")
            signal['strength'] -= 2

        # Determine action
        if signal['strength'] >= 3:
            signal['action'] = 'STRONG BUY'
        elif signal['strength'] >= 1:
            signal['action'] = 'BUY'
        elif signal['strength'] <= -3:
            signal['action'] = 'STRONG SELL'
        elif signal['strength'] <= -1:
            signal['action'] = 'SELL'

        return signal

    def buy(self, quantity):
        """Record a buy trade"""
        signal = self.generate_signal()
        price = signal['price']

        old_qty = self.portfolio[self.symbol]['quantity']
        old_cost = old_qty * self.portfolio[self.symbol]['avg_cost']

        self.portfolio[self.symbol]['quantity'] += quantity
        self.portfolio[self.symbol]['avg_cost'] = (old_cost + quantity * price) / self.portfolio[self.symbol]['quantity']

        trade = {
            'date': datetime.now(),
            'action': 'BUY',
            'symbol': self.symbol,
            'quantity': quantity,
            'price': price,
            'total': quantity * price
        }
        self.trades.append(trade)
        return trade

    def sell(self, quantity):
        """Record a sell trade"""
        if self.portfolio[self.symbol]['quantity'] < quantity:
            return None

        signal = self.generate_signal()
        price = signal['price']

        self.portfolio[self.symbol]['quantity'] -= quantity
        trade = {
            'date': datetime.now(),
            'action': 'SELL',
            'symbol': self.symbol,
            'quantity': quantity,
            'price': price,
            'total': quantity * price
        }
        self.trades.append(trade)
        return trade

def create_price_chart(data):
    """Create interactive price chart"""
    df = pd.DataFrame({
        'Date': data['dates'],
        'Close': data['closes'],
        'SMA20': [None]*20 + [StockAnalyzer._calculate_sma(data['closes'][:i+1], 20) for i in range(20, len(data['closes']))],
        'SMA50': [None]*50 + [StockAnalyzer._calculate_sma(data['closes'][:i+1], 50) for i in range(50, len(data['closes']))]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Price', line=dict(color='blue')))
    if df['SMA20'].notna().any():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], mode='lines', name='SMA20', line=dict(color='orange')))
    if df['SMA50'].notna().any():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='red')))

    fig.update_layout(
        title="Stock Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def main():
    st.markdown('<div class="main-header">📈 Stock Trading App</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = ""
    if 'stock_loaded' not in st.session_state:
        st.session_state.stock_loaded = False

    # Auto-load AAPL if no stock is loaded yet
    if not st.session_state.stock_loaded:
        st.session_state.current_symbol = "AAPL"
        st.session_state.analyzer = StockAnalyzer("AAPL")
        if st.session_state.analyzer.fetch_data():
            st.session_state.stock_loaded = True
        else:
            st.error("❌ Failed to auto-load AAPL data")

    # Sidebar for navigation
    with st.sidebar:
        st.header("📊 Trading Menu")
        menu = st.radio("Choose Action:", [
            "🏠 Dashboard",
            "📈 Analysis",
            "💰 Trade",
            "📋 Portfolio",
            "📚 History"
        ])

    # Main content
    if menu == "🏠 Dashboard":
        st.subheader("Welcome to Stock Trading App")
        st.write(f"Currently analyzing: **{st.session_state.current_symbol}**")

        # Stock symbol input form
        with st.form(key="load_stock_form"):
            col1, col2 = st.columns([3, 1])
            with col1:
                symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL, MSFT):", value=st.session_state.current_symbol, key="symbol_input")
            with col2:
                load_button = st.form_submit_button("Load Stock")

            if load_button:
                if symbol.strip():
                    new_symbol = symbol.upper()
                    analyzer = StockAnalyzer(new_symbol)
                    if analyzer.fetch_data():
                        st.session_state.current_symbol = new_symbol
                        st.session_state.analyzer = analyzer
                        st.session_state.stock_loaded = True
                        st.success(f"✅ Loaded {new_symbol} data!")
                    else:
                        st.error("❌ Failed to load stock data")
                else:
                    st.warning("Please enter a stock symbol")

        if st.session_state.stock_loaded:
            analyzer = st.session_state.analyzer
            signal = analyzer.generate_signal()

            st.info("✅ Stock data is loaded. Go to Analysis or Trade.")
            st.markdown(f"### Today's signal: **{signal['action']}**")
            st.write(f"- Current price: **${signal['price']:.2f}**")

            if signal['action'] in ['BUY', 'STRONG BUY']:
                st.success(f"Buy today at around ${signal['price']:.2f}")
            elif signal['action'] in ['SELL', 'STRONG SELL']:
                st.error(f"Sell today at around ${signal['price']:.2f}")
            else:
                st.warning("Hold for now; no clear buy/sell signal today.")

            if analyzer.portfolio[analyzer.symbol]['quantity'] > 0:
                qty = analyzer.portfolio[analyzer.symbol]['quantity']
                avg_cost = analyzer.portfolio[analyzer.symbol]['avg_cost']
                current_price = signal['price']
                st.write(f"- You own **{qty}** shares")
                st.write(f"- Average buy cost: **${avg_cost:.2f}**")
                st.write(f"- Current sell value: **${qty * current_price:.2f}**")
                st.write(f"- Estimated profit/loss: **${(current_price - avg_cost) * qty:.2f}**")
        else:
            st.warning("No stock data loaded yet.")

    elif menu == "📈 Analysis":
        if st.session_state.analyzer:
            analyzer = st.session_state.analyzer
            signal = analyzer.generate_signal()

            # Signal display
            signal_class = {
                'STRONG BUY': 'buy-signal',
                'BUY': 'buy-signal',
                'HOLD': 'hold-signal',
                'SELL': 'sell-signal',
                'STRONG SELL': 'sell-signal'
            }.get(signal['action'], 'hold-signal')

            st.markdown(f'<div class="signal-box {signal_class}">🎯 Signal: {signal["action"]} (Strength: {signal["strength"]})</div>', unsafe_allow_html=True)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${signal['price']:.2f}")
            with col2:
                st.metric("RSI (14)", f"{signal['rsi']:.1f}" if signal['rsi'] else "N/A")
            with col3:
                st.metric("SMA (20)", f"${signal['sma20']:.2f}" if signal['sma20'] else "N/A")
            with col4:
                st.metric("SMA (50)", f"${signal['sma50']:.2f}" if signal['sma50'] else "N/A")

            # Analysis reasons
            if signal['reasons']:
                st.subheader("📋 Analysis Details")
                for reason in signal['reasons']:
                    st.write(f"• {reason}")

            # Price chart
            st.subheader("📊 Price Chart")
            fig = create_price_chart(analyzer.data)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Please load a stock first from the Dashboard")

    elif menu == "💰 Trade":
        if st.session_state.analyzer:
            analyzer = st.session_state.analyzer

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🛒 Buy Shares")
                buy_qty = st.number_input("Quantity:", min_value=1, value=10, key="buy_qty")
                if st.button("Buy", use_container_width=True):
                    trade = analyzer.buy(buy_qty)
                    if trade:
                        st.success(f"✅ Bought {buy_qty} shares @ ${trade['price']:.2f}")
                    else:
                        st.error("❌ Buy failed")

            with col2:
                st.subheader("💸 Sell Shares")
                max_sell = analyzer.portfolio[analyzer.symbol]['quantity']
                sell_qty = st.number_input("Quantity:", min_value=1, max_value=max_sell if max_sell > 0 else 1, value=min(10, max_sell) if max_sell > 0 else 1, key="sell_qty")
                if st.button("Sell", use_container_width=True):
                    if max_sell >= sell_qty:
                        trade = analyzer.sell(sell_qty)
                        if trade:
                            st.success(f"✅ Sold {sell_qty} shares @ ${trade['price']:.2f}")
                        else:
                            st.error("❌ Sell failed")
                    else:
                        st.error(f"❌ Insufficient shares. You have {max_sell}")

        else:
            st.warning("Please load a stock first from the Dashboard")

    elif menu == "📋 Portfolio":
        if st.session_state.analyzer:
            analyzer = st.session_state.analyzer
            portfolio = analyzer.portfolio[analyzer.symbol]

            if portfolio['quantity'] > 0:
                signal = analyzer.generate_signal()
                current_value = portfolio['quantity'] * signal['price']
                cost_basis = portfolio['quantity'] * portfolio['avg_cost']
                profit = current_value - cost_basis
                profit_pct = (profit / cost_basis) * 100 if cost_basis > 0 else 0

                st.subheader("📊 Portfolio Summary")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Shares Owned", portfolio['quantity'])
                with col2:
                    st.metric("Avg Cost", f"${portfolio['avg_cost']:.2f}")
                with col3:
                    st.metric("Current Value", f"${current_value:.2f}")
                with col4:
                    st.metric("P&L", f"${profit:.2f} ({profit_pct:+.2f}%)", delta=f"{profit_pct:+.2f}%")
            else:
                st.info("No shares owned yet. Start trading!")

        else:
            st.warning("Please load a stock first from the Dashboard")

    elif menu == "📚 History":
        if st.session_state.analyzer and st.session_state.analyzer.trades:
            st.subheader("📋 Trade History")

            # Convert trades to DataFrame for better display
            trades_df = pd.DataFrame(st.session_state.analyzer.trades)
            trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            trades_df['total'] = trades_df['total'].map('${:.2f}'.format)
            trades_df['price'] = trades_df['price'].map('${:.2f}'.format)

            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades yet. Start trading to see history!")

    # Footer
    st.markdown("---")
    st.markdown("*Demo app with simulated data for educational purposes*")

if __name__ == "__main__":
    main()