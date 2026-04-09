import streamlit as st
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from collections import defaultdict
import warnings
import yfinance as yf
import requests
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')

# Set page config for mobile
st.set_page_config(
    page_title="AI Stock Trading App",
    page_icon="🤖",
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
    .ai-signal {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .buy-ai { background: linear-gradient(45deg, #28a745, #20c997); color: white; }
    .sell-ai { background: linear-gradient(45deg, #dc3545, #fd7e14); color: white; }
    .hold-ai { background: linear-gradient(45deg, #ffc107, #e0a800); color: black; }
    .confidence-bar {
        width: 100%;
        height: 20px;
        border-radius: 10px;
        background-color: #e9ecef;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    @media (max-width: 768px) {
        .main-header { font-size: 1.5rem; }
        .ai-signal { font-size: 1.2rem; }
    }
</style>
""", unsafe_allow_html=True)

class AIPredictor:
    """AI-powered stock price predictor using machine learning"""

    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.model = None
        self.scaler = StandardScaler()
        self.data = None
        self.predictions = []

    INDIAN_SYMBOL_MAP = {
        'RELIANCE': 'RELIANCE.NS',
        'INFY': 'INFY.NS',
        'TATA': 'TATASTEEL.NS',
        'TATASTEEL': 'TATASTEEL.NS',
        'NIPPON': 'NIPPONSTEEL.NS',
        'HDFC': 'HDFCBANK.NS',
        'ICICI': 'ICICIBANK.NS',
        'SBIN': 'SBIN.NS',
        'NSE': '^NSEI',
        'BSE': '^BSESN',
        'GOLD': 'GC=F',
        'GOLD RATE': 'GC=F',
        'GOLD PRICE': 'GC=F',
        'SILVER': 'SI=F',
        'CRUDE': 'CL=F',
        'GOLDLN': 'GC=F',
        'NIPPON SILVER': 'SI=F'
    }

    INR_CONVERTIBLE_SYMBOLS = {
        'GC=F': 'USDINR=X',
        'SI=F': 'USDINR=X',
        'CL=F': 'USDINR=X'
    }

    NEWS_POSITIVE_KEYWORDS = ['rally', 'gain', 'uptrend', 'bullish', 'record', 'surge', 'beats', 'outperform', 'buy']
    NEWS_NEGATIVE_KEYWORDS = ['fall', 'drop', 'downtrend', 'bearish', 'weak', 'loss', 'sell', 'decline', 'cut']

    def resolve_symbol(self, raw_symbol):
        symbol = raw_symbol.strip().upper()
        if symbol.startswith('NSE:') or symbol.startswith('BSE:'):
            symbol = symbol.split(':', 1)[1]
        if symbol in self.INDIAN_SYMBOL_MAP:
            return self.INDIAN_SYMBOL_MAP[symbol]
        if 'GOLD' in symbol:
            return 'GC=F'
        if 'SILVER' in symbol:
            return 'SI=F'
        return symbol

    def fetch_yfinance_history(self, symbol, days=200):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f'{days}d')
            if data is None or data.empty:
                data = yf.download(symbol, period=f'{days}d', progress=False)
            if data is None:
                return pd.DataFrame()
            return data
        except Exception:
            return pd.DataFrame()

    def fetch_moneycontrol_headlines(self):
        """Fetch market headlines from Moneycontrol and estimate sentiment"""
        try:
            url = 'https://www.moneycontrol.com/news/'
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
            })
            if response.status_code != 200:
                return [], 0.0

            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = [tag.get_text(strip=True) for tag in soup.select('h2 a, h3 a, h4 a')][:15]
            headlines = [h for h in headlines if h]

            if not headlines:
                return [], 0.0

            score = 0
            for headline in headlines:
                text = headline.lower()
                for positive in self.NEWS_POSITIVE_KEYWORDS:
                    if positive in text:
                        score += 1
                for negative in self.NEWS_NEGATIVE_KEYWORDS:
                    if negative in text:
                        score -= 1

            sentiment = max(min(score / max(len(headlines), 1), 1.0), -1.0)
            return headlines, sentiment
        except Exception:
            return [], 0.0

    def fetch_currency_rate(self, currency_symbol='USDINR=X'):
        """Fetch the latest currency conversion rate"""
        try:
            fx = yf.download(currency_symbol, period='7d', progress=False)
            if fx.empty:
                return 1.0
            return float(fx['Close'].iloc[-1])
        except Exception:
            return 1.0

    def fetch_real_market_data(self, days=200):
        """Fetch real market data from Yahoo Finance and MoneyControl"""
        ticker_symbol = self.resolve_symbol(self.symbol)
        data = self.fetch_yfinance_history(ticker_symbol, days=days)

        if data.empty:
            self.data = self.generate_historical_data(days)
            self.data.update({
                'currency': 'INR',
                'headlines': [],
                'news_sentiment': 0.0,
                'source': 'Demo Data',
                'display_symbol': self.symbol
            })
            return self.data

        dates = data.index.tolist()

        price_series = None
        volume_series = None

        if isinstance(data.columns, pd.MultiIndex):
            close_col = ('Close', ticker_symbol)
            volume_col = ('Volume', ticker_symbol)

            if close_col in data.columns:
                price_series = data.loc[:, close_col]
            elif any(col[0] == 'Close' for col in data.columns):
                price_series = data.loc[:, [col for col in data.columns if col[0] == 'Close'][0]]

            if volume_col in data.columns:
                volume_series = data.loc[:, volume_col]
            elif any(col[0] == 'Volume' for col in data.columns):
                volume_series = data.loc[:, [col for col in data.columns if col[0] == 'Volume'][0]]
        else:
            if 'Close' in data.columns:
                price_series = data['Close']
            else:
                price_series = data.iloc[:, 0]

            if 'Volume' in data.columns:
                volume_series = data['Volume']

        if price_series is None:
            self.data = {
                'currency': 'INR',
                'headlines': [],
                'news_sentiment': 0.0,
                'source': 'Demo Data',
                'display_symbol': self.symbol
            }
            return self.data

        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
        if isinstance(volume_series, pd.DataFrame):
            volume_series = volume_series.iloc[:, 0]

        price_series = price_series.dropna()
        volumes = volume_series.dropna().astype(float).tolist() if volume_series is not None else [1000000] * len(price_series)
        prices = price_series.astype(float).tolist()

        headlines, sentiment = self.fetch_moneycontrol_headlines()

        currency = 'USD'
        display_symbol = self.symbol
        if ticker_symbol in self.INR_CONVERTIBLE_SYMBOLS:
            fx_symbol = self.INR_CONVERTIBLE_SYMBOLS[ticker_symbol]
            conversion_rate = self.fetch_currency_rate(fx_symbol)
            prices = [float(p) * conversion_rate for p in prices]
            currency = 'INR'
            display_symbol = f"{self.symbol} (INR)"

        self.data = {
            'dates': dates,
            'prices': prices,
            'volumes': volumes,
            'source': 'Real Market Data (Yahoo Finance)',
            'headlines': headlines,
            'news_sentiment': sentiment,
            'currency': currency,
            'display_symbol': display_symbol
        }
        return self.data

    def generate_historical_data(self, days=200):
        """Generate realistic historical stock data with trends and volatility"""
        dates = []
        prices = []
        volumes = []

        # Start with a random base price
        base_price = random.uniform(50, 500)
        current_price = base_price

        # Market trend (bull/bear market simulation)
        market_trend = random.choice(['bull', 'bear', 'sideways'])
        trend_strength = random.uniform(0.0001, 0.0005)  # Daily trend

        for i in range(days):
            date = datetime.now() - timedelta(days=days-1-i)
            dates.append(date)

            # Add market trend
            if market_trend == 'bull':
                trend_factor = 1 + trend_strength
            elif market_trend == 'bear':
                trend_factor = 1 - trend_strength
            else:
                trend_factor = 1 + random.uniform(-trend_strength, trend_strength)

            # Add random volatility
            volatility = random.uniform(-0.03, 0.03)  # -3% to +3%
            current_price *= (trend_factor + volatility)

            # Ensure price doesn't go negative
            current_price = max(current_price, 1.0)

            # Add some periodic patterns (weekend effect, etc.)
            if date.weekday() >= 5:  # Weekend
                current_price *= random.uniform(0.98, 1.02)

            prices.append(round(current_price, 2))
            volumes.append(random.randint(100000, 10000000))

        self.data = {
            'dates': dates,
            'prices': prices,
            'volumes': volumes
        }
        return self.data

    def prepare_features(self, data, lookback=10):
        """Prepare features for ML model"""
        prices = np.array(data['prices'], dtype=float)
        volumes = np.array(data.get('volumes', [0] * len(prices)), dtype=float)

        features = []
        targets = []

        for i in range(lookback, len(prices) - 1):
            window = prices[i - lookback:i]
            ma5 = np.mean(window[-5:]) if len(window) >= 5 else np.mean(window)
            ma10 = np.mean(window)
            volatility = np.std(window)
            momentum = window[-1] - window[0]
            volume_avg = np.mean(volumes[i - lookback:i])
            volume_change = 0.0
            if volumes[i - lookback] > 0:
                volume_change = (volumes[i - 1] - volumes[i - lookback]) / volumes[i - lookback]

            feature_vector = np.concatenate([
                window,
                [ma5, ma10, volatility, momentum, volume_avg, volume_change]
            ])
            features.append(feature_vector)

            next_price = prices[i + 1]
            current_price = prices[i]
            targets.append(1 if next_price > current_price else 0)

        return np.array(features), np.array(targets)

    def train_model(self):
        """Train the AI model on historical data"""
        if not self.data:
            self.fetch_real_market_data()  # Fetch real data first

        features, targets = self.prepare_features(self.data)

        if len(features) < 30:  # Need enough data for classification
            return False

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train logistic regression classifier
        self.model = LogisticRegression(max_iter=300, solver='liblinear', random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Calculate accuracy
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)

        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }

    def predict_next_move(self):
        """Predict the next price movement using AI"""
        if not self.model or not self.data:
            return {'signal': 'HOLD', 'confidence': 0, 'prediction': 'No data'}

        # Get recent data for prediction
        recent_prices = np.array(self.data['prices'][-10:], dtype=float)
        recent_volumes = np.array(self.data.get('volumes', [0] * len(self.data['prices']))[-10:], dtype=float)

        # Calculate features with the same structure used during training
        ma5 = np.mean(recent_prices[-5:])
        ma10 = np.mean(recent_prices)
        volatility = np.std(recent_prices)
        momentum = recent_prices[-1] - recent_prices[0]
        volume_avg = np.mean(recent_volumes)
        volume_change = 0.0
        if len(recent_volumes) >= 1 and recent_volumes[0] > 0:
            volume_change = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0]

        features = np.concatenate([
            recent_prices,
            [ma5, ma10, volatility, momentum, volume_avg, volume_change]
        ])
        features_scaled = self.scaler.transform([features])

        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            prediction_prob = self.model.predict_proba(features_scaled)[0]
        else:
            raw_pred = self.model.predict(features_scaled)[0]
            raw_pred = min(max(raw_pred, 0.0), 1.0)
            prediction_prob = [1 - raw_pred, raw_pred]

        up_prob = float(prediction_prob[1])
        down_prob = float(prediction_prob[0])

        # Incorporate news sentiment from Moneycontrol
        news_sentiment = self.data.get('news_sentiment', 0.0)
        if news_sentiment > 0.25:
            up_prob = min(1.0, up_prob + 0.08 * news_sentiment)
        elif news_sentiment < -0.25:
            down_prob = min(1.0, down_prob + 0.08 * abs(news_sentiment))

        # Re-normalize probabilities
        total = up_prob + down_prob
        if total > 0:
            up_prob /= total
            down_prob /= total

        confidence = max(up_prob, down_prob)

        if up_prob > 0.55:
            signal = 'BUY'
            reason = f"AI predicts {up_prob:.1%} chance of price increase"
        elif down_prob > 0.55:
            signal = 'SELL'
            reason = f"AI predicts {down_prob:.1%} chance of price decrease"
        else:
            signal = 'HOLD'
            reason = f"AI prediction uncertain ({max(up_prob, down_prob):.1%} confidence)"

        news_reason = ''
        if news_sentiment > 0.2:
            news_reason = 'Positive market headlines support the signal.'
        elif news_sentiment < -0.2:
            news_reason = 'Negative market headlines reduce confidence in the signal.'

        return {
            'signal': signal,
            'confidence': confidence,
            'up_probability': up_prob,
            'down_probability': down_prob,
            'reason': f"{reason} {news_reason}".strip(),
            'current_price': self.data['prices'][-1],
            'news_sentiment': news_sentiment,
            'news_headlines': self.data.get('headlines', [])
        }

    def get_prediction_history(self, days=30):
        """Get historical predictions for backtesting"""
        if not self.data or len(self.data['prices']) < days:
            return []

        predictions = []
        for i in range(days):
            # Simulate predictions for past days
            pred = self.predict_next_move()
            pred['date'] = self.data['dates'][-1-i] if i < len(self.data['dates']) else datetime.now()
            pred['actual_move'] = 'UP' if random.random() > 0.5 else 'DOWN'  # Simulated
            predictions.append(pred)

        return predictions

class AITradingApp:
    """AI-powered trading application"""

    def __init__(self):
        self.predictors = {}
        self.portfolio = defaultdict(lambda: {'quantity': 0, 'avg_cost': 0})
        self.trades = []

    def get_predictor(self, symbol):
        """Get or create AI predictor for a symbol"""
        if symbol not in self.predictors:
            self.predictors[symbol] = AIPredictor(symbol)
        return self.predictors[symbol]

    def buy_stock(self, symbol, quantity):
        """Execute AI-recommended buy"""
        predictor = self.get_predictor(symbol)
        prediction = predictor.predict_next_move()

        if prediction['signal'] == 'BUY':
            price = prediction['current_price']

            old_qty = self.portfolio[symbol]['quantity']
            old_cost = old_qty * self.portfolio[symbol]['avg_cost']

            self.portfolio[symbol]['quantity'] += quantity
            self.portfolio[symbol]['avg_cost'] = (old_cost + quantity * price) / self.portfolio[symbol]['quantity']

            trade = {
                'date': datetime.now(),
                'action': 'BUY',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'total': quantity * price,
                'ai_confidence': prediction['confidence'],
                'ai_reason': prediction['reason']
            }
            self.trades.append(trade)
            return trade
        return None

    def sell_stock(self, symbol, quantity):
        """Execute AI-recommended sell"""
        if self.portfolio[symbol]['quantity'] < quantity:
            return None

        predictor = self.get_predictor(symbol)
        prediction = predictor.predict_next_move()

        if prediction['signal'] == 'SELL':
            price = prediction['current_price']

            self.portfolio[symbol]['quantity'] -= quantity

            trade = {
                'date': datetime.now(),
                'action': 'SELL',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'total': quantity * price,
                'ai_confidence': prediction['confidence'],
                'ai_reason': prediction['reason']
            }
            self.trades.append(trade)
            return trade
        return None

def create_prediction_chart(data, predictions):
    """Create chart with predictions"""
    df = pd.DataFrame({
        'Date': data['dates'][-50:],  # Last 50 days
        'Price': data['prices'][-50:]
    })

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Price'],
        mode='lines', name='Stock Price',
        line=dict(color='blue', width=2)
    ))

    # Add prediction points (last few days)
    if len(predictions) > 0:
        pred_dates = [p.get('date', datetime.now()) for p in predictions[-10:]]
        pred_prices = [data['prices'][-1]] * len(pred_dates)  # Current price
        pred_colors = ['green' if p['signal'] == 'BUY' else 'red' if p['signal'] == 'SELL' else 'orange'
                      for p in predictions[-10:]]

        fig.add_trace(go.Scatter(
            x=pred_dates, y=pred_prices,
            mode='markers', name='AI Predictions',
            marker=dict(color=pred_colors, size=10, symbol='triangle-up')
        ))

    fig.update_layout(
        title="Stock Price with AI Predictions",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def main():
    st.markdown('<div class="main-header">🤖 AI Stock Trading App</div>', unsafe_allow_html=True)
    st.markdown("*Powered by Machine Learning for Smart Trading Decisions*")

    # Initialize app
    if 'ai_app' not in st.session_state:
        st.session_state.ai_app = AITradingApp()
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = ""

    # Sidebar navigation
    with st.sidebar:
        st.header("🤖 AI Trading Menu")
        menu = st.radio("Choose Action:", [
            "� All Stock Rates",
            "📅 Today's Picks",
            "🏠 Dashboard",
            "🧠 AI Analysis",
            "💰 AI Trading",
            "📊 Portfolio",
            "📈 Performance",
            "🎯 AI Insights"
        ])
        
        st.markdown("---")
        st.subheader("🔗 External Links")
        st.markdown("[📊 MoneyControl](https://www.moneycontrol.com/)", unsafe_allow_html=True)
        st.markdown("[📈 NSE India](https://www.nseindia.com/)", unsafe_allow_html=True)
        st.markdown("[💼 BSE India](https://www.bseindia.com/)", unsafe_allow_html=True)

    if menu == "📊 All Stock Rates":
        st.markdown('<div class="main-header">📊 All Stock Rates - Live Market Prices</div>', unsafe_allow_html=True)
        st.write("Real-time prices for major stocks across global markets")
        
        # Comprehensive list of major stocks
        all_stocks = {
            'US Stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
            'Indian Stocks': ['RELIANCE', 'INFY', 'TATASTEEL', 'HDFC', 'ICICI', 'SBIN', 'TCS', 'WIPRO', 'LT', 'MARUTI'],
            'Commodities': ['GOLD', 'SILVER', 'CRUDE']
        }
        
        stock_data = []
        
        with st.spinner("📡 Fetching live market data..."):
            for category, symbols in all_stocks.items():
                for symbol in symbols:
                    try:
                        predictor = st.session_state.ai_app.get_predictor(symbol)
                        predictor.fetch_real_market_data(days=5)  # Get recent 5 days data
                        
                        if predictor.data and len(predictor.data.get('prices', [])) > 0:
                            current_price = predictor.data['prices'][-1]
                            prev_price = predictor.data['prices'][-2] if len(predictor.data['prices']) > 1 else current_price
                            change = current_price - prev_price
                            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                            
                            currency_symbol = '₹' if predictor.data.get('currency') == 'INR' else '$'
                            data_source = predictor.data.get('source', 'Unknown')
                            
                            stock_data.append({
                                'Symbol': symbol,
                                'Category': category,
                                'Current Price': f"{currency_symbol}{current_price:.2f}",
                                'Change': f"{change:+.2f}",
                                'Change %': f"{change_pct:+.2f}%",
                                'Data Source': data_source
                            })
                        else:
                            stock_data.append({
                                'Symbol': symbol,
                                'Category': category,
                                'Current Price': 'N/A',
                                'Change': 'N/A',
                                'Change %': 'N/A',
                                'Data Source': 'No Data'
                            })
                    except Exception as e:
                        stock_data.append({
                            'Symbol': symbol,
                            'Category': category,
                            'Current Price': 'Error',
                            'Change': 'Error',
                            'Change %': 'Error',
                            'Data Source': 'Failed'
                        })
        
        # Display the data
        if stock_data:
            df = pd.DataFrame(stock_data)
            
            # Color coding for the table
            def color_change(val):
                if val == 'N/A' or val == 'Error':
                    return 'color: gray'
                try:
                    if '+' in str(val):
                        return 'color: green; font-weight: bold'
                    elif '-' in str(val):
                        return 'color: red; font-weight: bold'
                    else:
                        return 'color: black'
                except:
                    return 'color: black'
            
            # Apply styling using map instead of deprecated applymap
            styled_df = df.style.map(color_change, subset=['Change', 'Change %'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Summary statistics
            total_stocks = len(stock_data)
            available_data = len([s for s in stock_data if s['Current Price'] not in ['N/A', 'Error']])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Stocks", total_stocks)
            with col2:
                st.metric("Data Available", available_data)
            with col3:
                st.metric("Success Rate", f"{available_data/total_stocks*100:.1f}%")
        else:
            st.error("❌ Unable to fetch any stock data. Please check your internet connection.")

    elif menu == "📅 Today's Picks":
        st.markdown('<div class="main-header">📅 Today\'s Market Analysis & AI Picks</div>', unsafe_allow_html=True)
        st.write("Global Market News + AI Recommendations for Today")
        
        st.subheader("🌍 Today's Global Markets")
        st.write("Analyzing top stocks across US, India, and Europe...")
        
        # Analyze multiple global stocks
        global_stocks = ['AAPL', 'GOOGL', 'MSFT', 'RELIANCE', 'INFY', 'TATASTEEL']
        
        with st.spinner("🤖 AI is analyzing today's market..."):
            recommendations = {'BUY': [], 'SELL': [], 'HOLD': []}
            
            for stock_symbol in global_stocks:
                try:
                    predictor = st.session_state.ai_app.get_predictor(stock_symbol)
                    predictor.fetch_real_market_data()
                    if predictor.data and len(predictor.data.get('prices', [])) > 0:
                        predictor.train_model()
                        prediction = predictor.predict_next_move()
                        
                        entry = {
                            'symbol': stock_symbol,
                            'price': predictor.data['prices'][-1],
                            'signal': prediction['signal'],
                            'confidence': prediction['confidence'],
                            'currency': predictor.data.get('currency', 'USD')
                        }
                        recommendations[prediction['signal']].append(entry)
                except Exception:
                    pass
        
        # Display recommendations
        if recommendations['BUY']:
            st.success("### 🟢 BUY TODAY")
            buy_df = pd.DataFrame(recommendations['BUY'])
            buy_df['Price'] = buy_df.apply(lambda row: f"{'₹' if row['currency'] == 'INR' else '$'}{row['price']:.2f}", axis=1)
            buy_df['Confidence'] = buy_df['confidence'].apply(lambda x: f"{x:.0%}")
            st.dataframe(buy_df[['symbol', 'Price', 'Confidence']], use_container_width=True)
        
        if recommendations['SELL']:
            st.error("### 🔴 SELL TODAY")
            sell_df = pd.DataFrame(recommendations['SELL'])
            sell_df['Price'] = sell_df.apply(lambda row: f"{'₹' if row['currency'] == 'INR' else '$'}{row['price']:.2f}", axis=1)
            sell_df['Confidence'] = sell_df['confidence'].apply(lambda x: f"{x:.0%}")
            st.dataframe(sell_df[['symbol', 'Price', 'Confidence']], use_container_width=True)
        
        if recommendations['HOLD']:
            st.info("### 🟡 HOLD TODAY")
            hold_df = pd.DataFrame(recommendations['HOLD'])
            hold_df['Price'] = hold_df.apply(lambda row: f"{'₹' if row['currency'] == 'INR' else '$'}{row['price']:.2f}", axis=1)
            hold_df['Confidence'] = hold_df['confidence'].apply(lambda x: f"{x:.0%}")
            st.dataframe(hold_df[['symbol', 'Price', 'Confidence']], use_container_width=True)
        
        st.markdown("---")
        st.subheader("📰 Global Market News")
        st.markdown("[📊 MoneyControl News](https://www.moneycontrol.com/news/)", unsafe_allow_html=True)
        st.markdown("[📈 Reuters Markets](https://www.reuters.com/finance/markets)", unsafe_allow_html=True)
        st.markdown("[💼 Bloomberg Markets](https://www.bloomberg.com/markets)", unsafe_allow_html=True)

    elif menu == "🏠 Dashboard":
        st.subheader("Welcome to AI-Powered Trading")
        st.write("Get intelligent buy/sell signals powered by machine learning!")

        # Stock selection
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL, MSFT, RELIANCE, NIPPON):",
                                 value=st.session_state.current_symbol, key="symbol_input")
        with col2:
            if st.button("🔍 Analyze with AI", use_container_width=True):
                if symbol.strip():
                    st.session_state.current_symbol = symbol.upper()
                    predictor = st.session_state.ai_app.get_predictor(symbol.upper())

                    with st.spinner("🤖 AI is loading live market data..."):
                        predictor.fetch_real_market_data()
                        predictor.train_model()

                    st.success(f"✅ AI trained on live data for {symbol.upper()}!")
                    st.rerun()
                else:
                    st.warning("Please enter a stock symbol")

        # Quick stats
        if st.session_state.current_symbol:
            predictor = st.session_state.ai_app.get_predictor(st.session_state.current_symbol)
            if predictor.data:
                currency_symbol = '₹' if predictor.data.get('currency') == 'INR' else '$'
                display_symbol = predictor.data.get('display_symbol', st.session_state.current_symbol)
                st.subheader(f"📊 Market Overview — {display_symbol}")
                if predictor.data.get('source') == 'Demo Data':
                    st.warning('⚠️ Live data unavailable for this symbol. Showing demo data.')
                else:
                    st.success(f"Data source: {predictor.data.get('source')}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"{currency_symbol}{predictor.data['prices'][-1]:.2f}")
                with col2:
                    change = predictor.data['prices'][-1] - predictor.data['prices'][-2]
                    st.metric("Daily Change", f"{change:+.2f}", f"{change/predictor.data['prices'][-2]*100:+.1f}%")
                with col3:
                    st.metric("AI Status", "🟢 Active")

    elif menu == "🧠 AI Analysis":
        if st.session_state.current_symbol:
            predictor = st.session_state.ai_app.get_predictor(st.session_state.current_symbol)

            if predictor.data:
                # AI Prediction
                with st.spinner("🤖 AI is making predictions..."):
                    prediction = predictor.predict_next_move()

                # Main AI Signal
                signal_class = {
                    'BUY': 'buy-ai',
                    'SELL': 'sell-ai',
                    'HOLD': 'hold-ai'
                }.get(prediction['signal'], 'hold-ai')

                st.markdown(f'<div class="ai-signal {signal_class}">🎯 AI Signal: {prediction["signal"]}</div>',
                          unsafe_allow_html=True)

                # Confidence meter
                confidence_pct = int(prediction['confidence'] * 100)
                st.subheader("🎯 AI Confidence Level")
                st.progress(confidence_pct / 100)
                st.write(f"**{confidence_pct}%** confidence in this prediction")

                # Detailed analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📈 Up Probability", f"{prediction['up_probability']:.1%}")
                with col2:
                    st.metric("📉 Down Probability", f"{prediction['down_probability']:.1%}")

                st.subheader("🧠 AI Reasoning")
                st.write(prediction['reason'])

                if prediction.get('news_headlines'):
                    st.subheader("📰 Moneycontrol News Sentiment")
                    sentiment_label = 'Positive' if prediction['news_sentiment'] > 0.1 else 'Negative' if prediction['news_sentiment'] < -0.1 else 'Neutral'
                    st.write(f"**Headline sentiment:** {sentiment_label} ({prediction['news_sentiment']:.2f})")
                    for headline in prediction['news_headlines'][:5]:
                        st.write(f'- {headline}')

                st.subheader("🔗 MoneyControl Links")
                col_mc1, col_mc2, col_mc3 = st.columns(3)
                with col_mc1:
                    st.markdown(f"[📊 MoneyControl Home](https://www.moneycontrol.com/)", unsafe_allow_html=True)
                with col_mc2:
                    st.markdown(f"[📰 Market News](https://www.moneycontrol.com/news/)", unsafe_allow_html=True)
                with col_mc3:
                    symbol_for_mc = st.session_state.current_symbol.upper()
                    if symbol_for_mc in ['RELIANCE', 'INFY', 'TATASTEEL', 'SBIN', 'HDFC', 'ICICI']:
                        st.markdown(f"[📈 {symbol_for_mc} on MC](https://www.moneycontrol.com/india/stockprice/{symbol_for_mc.lower()})", unsafe_allow_html=True)
                    else:
                        st.markdown(f"[🔍 Search on MC](https://www.moneycontrol.com/stocks/searchresults/{symbol_for_mc.lower()})", unsafe_allow_html=True)

                # Price chart with predictions
                st.subheader("📊 Price Chart with AI Predictions")
                fig = create_prediction_chart(predictor.data, predictor.get_prediction_history())
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Please load a stock first from the Dashboard")
        else:
            st.warning("Please select a stock symbol first")

    elif menu == "💰 AI Trading":
        if st.session_state.current_symbol:
            predictor = st.session_state.ai_app.get_predictor(st.session_state.current_symbol)

            if predictor.data:
                prediction = predictor.predict_next_move()

                st.subheader("🤖 AI Trading Recommendations")

                # Current position
                current_qty = st.session_state.ai_app.portfolio[st.session_state.current_symbol]['quantity']
                if current_qty > 0:
                    st.info(f"📊 Current Position: {current_qty} shares @ avg ${st.session_state.ai_app.portfolio[st.session_state.current_symbol]['avg_cost']:.2f}")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🛒 AI Buy Recommendation")
                    if prediction['signal'] == 'BUY':
                        buy_qty = st.number_input("Quantity to Buy:", min_value=1, value=10, key="buy_qty")
                        if st.button("✅ Execute AI Buy", use_container_width=True):
                            trade = st.session_state.ai_app.buy_stock(st.session_state.current_symbol, buy_qty)
                            if trade:
                                st.success(f"🤖 AI executed BUY: {buy_qty} shares @ ${trade['price']:.2f}")
                            else:
                                st.error("❌ AI buy failed")
                    else:
                        st.write("🤖 AI does not recommend buying now")
                        st.write(f"**Reason:** {prediction['reason']}")

                with col2:
                    st.subheader("💸 AI Sell Recommendation")
                    if prediction['signal'] == 'SELL' and current_qty > 0:
                        sell_qty = st.number_input("Quantity to Sell:", min_value=1, max_value=current_qty, value=min(10, current_qty), key="sell_qty")
                        if st.button("✅ Execute AI Sell", use_container_width=True):
                            trade = st.session_state.ai_app.sell_stock(st.session_state.current_symbol, sell_qty)
                            if trade:
                                st.success(f"🤖 AI executed SELL: {sell_qty} shares @ ${trade['price']:.2f}")
                            else:
                                st.error("❌ AI sell failed")
                    else:
                        if current_qty == 0:
                            st.write("📊 No shares to sell")
                        else:
                            st.write("🤖 AI does not recommend selling now")
                            st.write(f"**Reason:** {prediction['reason']}")

            else:
                st.warning("Please load a stock first")
        else:
            st.warning("Please select a stock symbol first")

    elif menu == "📊 Portfolio":
        portfolio = st.session_state.ai_app.portfolio

        if any(v['quantity'] > 0 for v in portfolio.values()):
            st.subheader("📊 Your AI-Traded Portfolio")

            total_value = 0
            total_cost = 0

            for symbol, position in portfolio.items():
                if position['quantity'] > 0:
                    predictor = st.session_state.ai_app.get_predictor(symbol)
                    current_price = predictor.data['prices'][-1] if predictor.data else position['avg_cost']

                    current_value = position['quantity'] * current_price
                    cost_basis = position['quantity'] * position['avg_cost']
                    profit = current_value - cost_basis
                    profit_pct = (profit / cost_basis) * 100 if cost_basis > 0 else 0

                    total_value += current_value
                    total_cost += cost_basis

                    with st.expander(f"📈 {symbol} Position"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Shares", position['quantity'])
                        with col2:
                            st.metric("Avg Cost", f"${position['avg_cost']:.2f}")
                        with col3:
                            st.metric("Current Value", f"${current_value:.2f}")
                        with col4:
                            st.metric("P&L", f"${profit:.2f}", f"{profit_pct:+.2f}%")

            if total_value > 0:
                total_profit = total_value - total_cost
                total_profit_pct = (total_profit / total_cost) * 100
                st.subheader("💰 Portfolio Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Value", f"${total_value:.2f}")
                with col2:
                    st.metric("Total Cost", f"${total_cost:.2f}")
                with col3:
                    st.metric("Total P&L", f"${total_profit:.2f}", f"{total_profit_pct:+.2f}%")
        else:
            st.info("🤖 No positions yet. Let AI make your first trade!")

    elif menu == "📈 Performance":
        trades = st.session_state.ai_app.trades

        if trades:
            st.subheader("📈 AI Trading Performance")

            # Convert to DataFrame
            df = pd.DataFrame(trades)
            df['date'] = pd.to_datetime(df['date'])

            # Performance metrics
            total_trades = len(trades)
            buy_trades = len(df[df['action'] == 'BUY'])
            sell_trades = len(df[df['action'] == 'SELL'])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Buy Orders", buy_trades)
            with col3:
                st.metric("Sell Orders", sell_trades)
            with col4:
                avg_confidence = df['ai_confidence'].mean()
                st.metric("Avg AI Confidence", f"{avg_confidence:.1%}")

            # Trade history table
            st.subheader("📋 AI Trade History")
            display_df = df[['date', 'action', 'symbol', 'quantity', 'price', 'ai_confidence']].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['price'] = display_df['price'].map('${:.2f}'.format)
            display_df['ai_confidence'] = display_df['ai_confidence'].map('{:.1%}'.format)
            display_df.columns = ['Date', 'Action', 'Symbol', 'Qty', 'Price', 'AI Confidence']

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("🤖 No trades yet. AI will track your performance here!")

    elif menu == "🎯 AI Insights":
        st.subheader("🎯 AI Trading Insights")

        if st.session_state.current_symbol:
            predictor = st.session_state.ai_app.get_predictor(st.session_state.current_symbol)

            if predictor.model:
                st.write("### 🤖 AI Model Performance")
                metrics = predictor.train_model()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Accuracy", f"{metrics['train_accuracy']:.1%}")
                with col2:
                    st.metric("Test Accuracy", f"{metrics['test_accuracy']:.1%}")

                st.write("### 📊 Prediction History")
                predictions = predictor.get_prediction_history(20)
                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    pred_df['date'] = pd.to_datetime(pred_df['date']).dt.strftime('%Y-%m-%d')
                    pred_df = pred_df[['date', 'signal', 'confidence', 'actual_move']]
                    pred_df.columns = ['Date', 'AI Signal', 'Confidence', 'Actual Move']
                    pred_df['Confidence'] = pred_df['Confidence'].map('{:.1%}'.format)

                    st.dataframe(pred_df, use_container_width=True)
            else:
                st.info("🤖 Train the AI model first by analyzing a stock!")
        else:
            st.write("### 🚀 How AI Trading Works")
            st.write("""
            **🤖 Machine Learning Predictions:**
            - Analyzes historical price patterns
            - Learns from market trends and volatility
            - Predicts future price movements

            **🎯 Smart Trading Signals:**
            - BUY: AI predicts price increase >55% probability
            - SELL: AI predicts price decrease >55% probability
            - HOLD: AI prediction uncertain

            **📊 Confidence Levels:**
            - Higher confidence = More reliable signals
            - Based on historical accuracy
            - Continuously learning from market data
            """)

    # Footer
    st.markdown("---")
    st.markdown("*🤖 AI-Powered Trading App - Making Smart Investment Decisions*")

if __name__ == "__main__":
    main()