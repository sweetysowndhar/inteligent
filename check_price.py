import yfinance as yf

print("=== Coal India (COALINDIA.NS) ===")
t = yf.Ticker("COALINDIA.NS")
d = t.history(period="5d")
print(d[["Close","Volume"]].tail())
print()
try:
    info = t.fast_info
    print(f"Last Price: {info.last_price}")
    print(f"Previous Close: {info.previous_close}")
except Exception as e:
    print(f"fast_info error: {e}")

print()
print("=== Using yf.download ===")
d2 = yf.download("COALINDIA.NS", period="5d", progress=False)
print(d2[["Close"]].tail())
