import random
from datetime import datetime, timedelta
import os
from collections import defaultdict

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
            print(f"[*] Generating {self.symbol} demo data...")

            # Generate 100 days of demo data
            base_price = random.uniform(50, 500)  # Random starting price
            dates = []
            closes = []
            opens = []
            highs = []
            lows = []
            volumes = []

            current_price = base_price

            for i in range(100):
                # Generate realistic price movements
                change_percent = random.uniform(-0.05, 0.05)  # -5% to +5% daily change
                current_price *= (1 + change_percent)

                # Ensure price doesn't go negative
                current_price = max(current_price, 1.0)

                # Generate OHLC data
                volatility = current_price * 0.02  # 2% daily volatility
                daily_open = current_price
                daily_close = current_price * (1 + random.uniform(-0.02, 0.02))
                daily_high = max(daily_open, daily_close) + random.uniform(0, volatility)
                daily_low = min(daily_open, daily_close) - random.uniform(0, volatility)

                # Generate volume (realistic trading volume)
                volume = random.randint(100000, 10000000)

                dates.append((datetime.now() - timedelta(days=99-i)).strftime('%Y-%m-%d'))
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

            print(f"[OK] Generated {len(self.data['closes'])} days of demo data")
            return True

        except Exception as e:
            print(f"[ERROR] Error generating data: {e}")
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

        # Calculate indicators
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
            signal['reasons'].append("[BULLISH] Price above both SMA20 and SMA50")
            signal['strength'] += 2
        elif current_price < sma20 and sma20 < sma50:
            signal['reasons'].append("[BEARISH] Price below both SMA20 and SMA50")
            signal['strength'] -= 2

        # RSI Strategy
        if rsi < 30:
            signal['reasons'].append(f"[RSI] Oversold ({rsi:.1f}) - Buy Signal")
            signal['strength'] += 2
        elif rsi > 70:
            signal['reasons'].append(f"[RSI] Overbought ({rsi:.1f}) - Sell Signal")
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
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'action': 'BUY',
            'symbol': self.symbol,
            'quantity': quantity,
            'price': price,
            'total': quantity * price
        }
        self.trades.append(trade)
        print(f"\n[BUY] {quantity} shares of {self.symbol} @ ${price:.2f} | Total: ${quantity * price:.2f}")

    def sell(self, quantity):
        """Record a sell trade"""
        if self.portfolio[self.symbol]['quantity'] < quantity:
            print(f"[ERROR] Insufficient shares. You have {self.portfolio[self.symbol]['quantity']}")
            return

        signal = self.generate_signal()
        price = signal['price']

        self.portfolio[self.symbol]['quantity'] -= quantity
        trade = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'action': 'SELL',
            'symbol': self.symbol,
            'quantity': quantity,
            'price': price,
            'total': quantity * price
        }
        self.trades.append(trade)
        print(f"\n[SELL] {quantity} shares of {self.symbol} @ ${price:.2f} | Total: ${quantity * price:.2f}")

    def display_analysis(self):
        """Display comprehensive stock analysis"""
        os.system('cls' if os.name == 'nt' else 'clear')

        signal = self.generate_signal()

        print("\n" + "="*80)
        print(f"{'STOCK ANALYSIS - ' + self.symbol:^80}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}")
        print("="*80)

        # Price Info
        print(f"\n[PRICE]")
        print(f"   Current Price: ${signal['price']:.2f}")

        # Indicators
        print(f"\n[INDICATORS]")
        if signal['rsi']:
            print(f"   RSI (14):    {signal['rsi']:.2f}")
        else:
            print(f"   RSI (14):    N/A")
        if signal['sma20']:
            print(f"   SMA (20):    ${signal['sma20']:.2f}")
        else:
            print(f"   SMA (20):    N/A")
        if signal['sma50']:
            print(f"   SMA (50):    ${signal['sma50']:.2f}")
        else:
            print(f"   SMA (50):    N/A")

        # Signal
        action_icons = {
            'STRONG BUY': '>>>> ',
            'BUY': '>> ',
            'HOLD': '-- ',
            'SELL': '<< ',
            'STRONG SELL': '<<<< '
        }

        print(f"\n[SIGNAL]")
        print(f"   {action_icons.get(signal['action'], '')} {signal['action']} (Strength: {signal['strength']})")

        print(f"\n[ANALYSIS]")
        for reason in signal['reasons']:
            print(f"   {reason}")

        # Portfolio
        if self.portfolio[self.symbol]['quantity'] > 0:
            current_value = self.portfolio[self.symbol]['quantity'] * signal['price']
            cost_basis = self.portfolio[self.symbol]['quantity'] * self.portfolio[self.symbol]['avg_cost']
            profit = current_value - cost_basis
            profit_pct = (profit / cost_basis) * 100 if cost_basis > 0 else 0

            print(f"\n[POSITION]")
            print(f"   Shares Owned:  {self.portfolio[self.symbol]['quantity']}")
            print(f"   Avg Cost:      ${self.portfolio[self.symbol]['avg_cost']:.2f}")
            print(f"   Current Value: ${current_value:.2f}")
            print(f"   Profit/Loss:   ${profit:.2f} ({profit_pct:+.2f}%)")

        # Recent Trades
        if self.trades:
            print(f"\n[RECENT TRADES (Last 5)]")
            print(f"   {'Date':<20} {'Action':<6} {'Qty':<6} {'Price':<10} {'Total':<12}")
            print(f"   {'-'*50}")
            for trade in self.trades[-5:]:
                print(f"   {trade['date']:<20} {trade['action']:<6} {trade['quantity']:<6} ${trade['price']:<9.2f} ${trade['total']:<11.2f}")

        print("\n" + "="*80)

def interactive_mode():
    """Interactive trading session"""
    analyzer = None

    while True:
        if analyzer is None:
            print("\n" + "="*80)
            print(f"{'DEMO STOCK TRADING APP':^80}")
            print("="*80)
            symbol = input("\n[INPUT] Enter stock symbol (e.g., AAPL, GOOGL, MSFT) or 'quit' to exit: ").strip()

            if symbol.lower() == 'quit':
                print("Goodbye!")
                break

            analyzer = StockAnalyzer(symbol)
            if not analyzer.fetch_data():
                analyzer = None
                continue

            analyzer.display_analysis()

        print("\n\nOptions:")
        print("  1. Analyze & View Signals")
        print("  2. Buy Shares")
        print("  3. Sell Shares")
        print("  4. Change Stock")
        print("  5. View Trade History")
        print("  6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == '1':
            analyzer.display_analysis()

        elif choice == '2':
            try:
                qty = int(input(f"\nHow many shares of {analyzer.symbol} to buy? "))
                if qty > 0:
                    analyzer.buy(qty)
                    input("\nPress Enter to continue...")
                else:
                    print("[ERROR] Invalid quantity")
            except ValueError:
                print("[ERROR] Please enter a valid number")

        elif choice == '3':
            try:
                qty = int(input(f"\nHow many shares of {analyzer.symbol} to sell? "))
                if qty > 0:
                    analyzer.sell(qty)
                    input("\nPress Enter to continue...")
                else:
                    print("[ERROR] Invalid quantity")
            except ValueError:
                print("[ERROR] Please enter a valid number")

        elif choice == '4':
            analyzer = None

        elif choice == '5':
            if analyzer.trades:
                print(f"\n{'TRADE HISTORY':^80}")
                print(f"{'Date':<20} {'Action':<6} {'Symbol':<8} {'Qty':<8} {'Price':<12} {'Total':<15}")
                print(f"{'-'*75}")
                for trade in analyzer.trades:
                    print(f"{trade['date']:<20} {trade['action']:<6} {trade['symbol']:<8} {trade['quantity']:<8} ${trade['price']:<11.2f} ${trade['total']:<14.2f}")
            else:
                print("\nNo trades yet.")
            input("\nPress Enter to continue...")

        elif choice == '6':
            print("Goodbye!")
            break

        else:
            print("[ERROR] Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n\nShutdown... Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")