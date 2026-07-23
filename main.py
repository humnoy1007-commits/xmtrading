import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict

# ==========================================
# 1. DATA MODELS & TRADING ENGINE
# ==========================================
@dataclass
class Position:
    ticket: int
    symbol: str
    order_type: str        # 'BUY' หรือ 'SELL'
    volume: float
    open_price: float
    current_price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    trailing_stop: Optional[float] = None
    pnl: float = 0.0

@dataclass
class PendingOrder:
    ticket: int
    symbol: str
    order_type: str        # 'BUY_LIMIT', 'SELL_LIMIT', 'BUY_STOP', 'SELL_STOP'
    volume: float
    target_price: float
    sl: Optional[float] = None
    tp: Optional[float] = None

class TradingEngine:
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: Dict[int, Position] = {}
        self.pending_orders: Dict[int, PendingOrder] = {}
        self.ticket_counter = 1000

    def _next_ticket(self) -> int:
        self.ticket_counter += 1
        return self.ticket_counter

    def open_market_order(self, symbol: str, order_type: str, volume: float, price: float, 
                          sl: float = None, tp: float = None, trailing_stop: float = None):
        ticket = self._next_ticket()
        pos = Position(
            ticket=ticket, symbol=symbol, order_type=order_type,
            volume=volume, open_price=price, current_price=price,
            sl=sl, tp=tp, trailing_stop=trailing_stop
        )
        self.positions[ticket] = pos

    def place_pending_order(self, symbol: str, order_type: str, volume: float, target_price: float,
                            sl: float = None, tp: float = None):
        ticket = self._next_ticket()
        order = PendingOrder(ticket, symbol, order_type, volume, target_price, sl, tp)
        self.pending_orders[ticket] = order

    def close_position(self, ticket: int, ratio: float = 1.0):
        if ticket not in self.positions:
            return
        pos = self.positions[ticket]
        close_volume = pos.volume * ratio
        realized_pnl = pos.pnl * ratio
        self.balance += realized_pnl
        pos.volume -= close_volume
        if pos.volume <= 0.001 or ratio >= 1.0:
            del self.positions[ticket]
        else:
            pos.pnl -= realized_pnl

    def reverse_position(self, ticket: int, current_price: float):
        if ticket not in self.positions:
            return
        pos = self.positions[ticket]
        new_type = 'SELL' if pos.order_type == 'BUY' else 'BUY'
        volume, symbol = pos.volume, pos.symbol
        self.close_position(ticket)
        self.open_market_order(symbol, new_type, volume, current_price)

    def delete_pending_order(self, ticket: int):
        if ticket in self.pending_orders:
            del self.pending_orders[ticket]

# ==========================================
# 2. CHART & INDICATORS ENGINE
# ==========================================
class ProfessionalChart:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        close, high, low = df['close'], df['high'], df['low']

        # 1. EMA & 2. SMA
        df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
        df['SMA_50'] = close.rolling(window=50).mean()

        # 3. RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 4. MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # 5. Bollinger Bands
        df['BB_Mid'] = close.rolling(window=20).mean()
        df['BB_Std'] = close.rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

        # 6. ATR
        tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift()).abs(), 'lc': (low - close.shift()).abs()}).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # 7. Stochastic
        low_14, high_14 = low.rolling(14).min(), high.rolling(14).max()
        df['Stoch_K'] = 100 * ((close - low_14) / (high_14 - low_14))

        # 8. VWAP
        df['VWAP'] = (df['volume'] * (high + low + close) / 3).cumsum() / df['volume'].cumsum()

        # 9. Ichimoku
        df['Tenkan'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
        df['Kijun'] = (high.rolling(26).max() + low.rolling(26).min()) / 2

        # 10. ADX (Simple Proxy)
        df['ADX'] = (high - low).rolling(14).mean()

        return df

    @classmethod
    def render_chart(cls, df: pd.DataFrame, active_indicators: list, drawing_tool: str = "None"):
        df = cls.calculate_indicators(df)
        sub_indicators = ['RSI', 'MACD', 'ATR', 'Stochastic', 'ADX']
        has_sub = any(i in active_indicators for i in sub_indicators)
        rows = 2 if has_sub else 1
        row_heights = [0.7, 0.3] if has_sub else [1.0]

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.03)

        # Candlestick Main
        fig.add_trace(go.Candlestick(x=df['datetime'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)

        # Main Overlays
        if 'EMA' in active_indicators: fig.add_trace(go.Scatter(x=df['datetime'], y=df['EMA_20'], name='EMA 20', line=dict(color='orange')), row=1, col=1)
        if 'SMA' in active_indicators: fig.add_trace(go.Scatter(x=df['datetime'], y=df['SMA_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
        if 'Bollinger Bands' in active_indicators:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
        if 'VWAP' in active_indicators: fig.add_trace(go.Scatter(x=df['datetime'], y=df['VWAP'], name='VWAP', line=dict(color='purple')), row=1, col=1)
        if 'Ichimoku' in active_indicators:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['Tenkan'], name='Tenkan', line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['Kijun'], name='Kijun', line=dict(color='blue')), row=1, col=1)

        # Sub Indicators
        if 'RSI' in active_indicators and has_sub: fig.add_trace(go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI', line=dict(color='magenta')), row=2, col=1)
        if 'MACD' in active_indicators and has_sub:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD'], name='MACD', line=dict(color='cyan')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD_Signal'], name='Signal', line=dict(color='yellow')), row=2, col=1)

        drag_mode = 'drawline' if drawing_tool == 'Trendline' else ('drawrect' if drawing_tool == 'Rectangle' else 'pan')
        fig.update_layout(template='plotly_dark', height=550, xaxis_rangeslider_visible=False, dragmode=drag_mode, margin=dict(l=10, r=10, t=30, b=10))
        return fig

# ==========================================
# 3. STREAMLIT UI DASHBOARD
# ==========================================
st.set_page_config(page_title="Pro Trading Terminal", layout="wide", page_icon="📈")

if 'engine' not in st.session_state:
    st.session_state.engine = TradingEngine(initial_balance=10000.0)

engine = st.session_state.engine

# Mock Data OHLC Generator
def generate_mock_ohlc(symbol, tf_minutes=5, count=100):
    now = datetime.now()
    dates = [now - timedelta(minutes=tf_minutes * i) for i in range(count)][::-1]
    price = 100.0 if symbol != 'BTCUSD' else 64000.0
    data = []
    for d in dates:
        open_p = price + np.random.uniform(-0.5, 0.5)
        high_p, low_p = open_p + np.random.uniform(0.1, 1.0), open_p - np.random.uniform(0.1, 1.0)
        close_p = open_p + np.random.uniform(-0.6, 0.6)
        price = close_p
        data.append([d, open_p, high_p, low_p, close_p, int(np.random.uniform(100, 1000))])
    return pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

# Top Metrics Bar
st.title("⚡ Pro Trading Dashboard")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Balance", f"${engine.balance:,.2f}")
m2.metric("Equity", f"${engine.equity:,.2f}")
m3.metric("Free Margin", f"${engine.equity:,.2f}")
m4.metric("Active Positions", len(engine.positions))
st.divider()

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Dynamic Settings")
    timeframe = st.selectbox("⏱️ Timeframe", ["M1", "M5", "M15", "H1", "H4", "D1"], index=1)
    
    st.subheader("📊 Indicators (เปิด/ปิด)")
    indicators = ["EMA", "SMA", "RSI", "MACD", "Bollinger Bands", "ATR", "ADX", "Stochastic", "Ichimoku", "VWAP"]
    selected_indicators = [ind for ind in indicators if st.checkbox(ind, value=(ind in ["EMA", "RSI"]))]
    
    st.subheader("✏️ Drawing Tools")
    drawing_tool = st.selectbox("เครื่องมือวาด", ["None", "Trendline", "Rectangle", "Horizontal Line", "Vertical Line", "Fibonacci"])

# Main Dashboard
c_left, c_right = st.columns([7, 3])

with c_left:
    df = generate_mock_ohlc("EURUSD", tf_minutes=5)
    fig = ProfessionalChart.render_chart(df, selected_indicators, drawing_tool)
    st.plotly_chart(fig, use_container_width=True)

with c_right:
    st.subheader("🛒 Order Engine")
    symbol = st.selectbox("Symbol", ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "XAUUSD"])
    order_mode = st.radio("Order Type", ["Market", "Pending"], horizontal=True)
    volume = st.number_input("Volume (Lots)", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
    
    col_sl, col_tp = st.columns(2)
    sl_val = col_sl.number_input("Stop Loss (SL)", min_value=0.0, value=0.0)
    tp_val = col_tp.number_input("Take Profit (TP)", min_value=0.0, value=0.0)
    trailing_val = st.number_input("Trailing Stop (Pts)", min_value=0.0, value=0.0)
    current_price = df['close'].iloc[-1]

    if order_mode == "Market":
        b1, b2 = st.columns(2)
        if b1.button("🔴 BUY (Market)", use_container_width=True, type="primary"):
            engine.open_market_order(symbol, "BUY", volume, current_price, sl_val or None, tp_val or None, trailing_val or None)
            st.rerun()
        if b2.button("🔵 SELL (Market)", use_container_width=True):
            engine.open_market_order(symbol, "SELL", volume, current_price, sl_val or None, tp_val or None, trailing_val or None)
            st.rerun()
    else:
        pending_type = st.selectbox("Pending Mode", ["BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"])
        target_price = st.number_input("Target Price", value=current_price)
        if st.button("Place Pending Order", use_container_width=True):
            engine.place_pending_order(symbol, pending_type, volume, target_price, sl_val or None, tp_val or None)
            st.rerun()

st.divider()

# Bottom Tabs Section
t1, t2, t3 = st.tabs(["📊 Market Watch", "💼 Open Positions", "⏳ Pending Orders"])

with t1:
    st.subheader("📋 Market Watch")
    sq = st.text_input("🔍 ค้นหาสินทรัพย์...", "").upper()
    assets = [
        {"Symbol": "EURUSD", "Bid": 1.0850, "Ask": 1.0852, "Spread": 0.0002, "Change %": "+0.15%", "High": 1.0890, "Low": 1.0820, "Volume": 12500, "Session": "London/NY"},
        {"Symbol": "GBPUSD", "Bid": 1.2640, "Ask": 1.2643, "Spread": 0.0003, "Change %": "-0.20%", "High": 1.2680, "Low": 1.2600, "Volume": 9800, "Session": "London/NY"},
        {"Symbol": "BTCUSD", "Bid": 64200.0, "Ask": 64210.0, "Spread": 10.0, "Change %": "+1.50%", "High": 65000.0, "Low": 63500.0, "Volume": 45000, "Session": "24/7"}
    ]
    m_df = pd.DataFrame(assets)
    if sq: m_df = m_df[m_df['Symbol'].str.contains(sq)]
    st.dataframe(m_df, use_container_width=True, hide_index=True)

with t2:
    if not engine.positions:
        st.caption("ไม่มี Position ที่เปิดอยู่")
    else:
        for ticket, pos in list(engine.positions.items()):
            p1, p2, p3, p4, p5, p6 = st.columns([1, 2, 2, 2, 2, 3])
            p1.write(f"#{ticket}")
            p2.write(f"**{pos.symbol}** ({pos.order_type})")
            p3.write(f"Vol: {pos.volume:.2f}")
            p4.write(f"Price: {pos.open_price:.4f}")
            p5.write(f"PnL: **${pos.pnl:.2f}**")
            with p6:
                c1, c2, c3 = st.columns(3)
                if c1.button("Close", key=f"close_{ticket}"):
                    engine.close_position(ticket, 1.0)
                    st.rerun()
                if c2.button("50%", key=f"part_{ticket}"):
                    engine.close_position(ticket, 0.5)
                    st.rerun()
                if c3.button("Reverse", key=f"rev_{ticket}"):
                    engine.reverse_position(ticket, current_price)
                    st.rerun()

with t3:
    if not engine.pending_orders:
        st.caption("ไม่มี Pending Orders")
    else:
        for ticket, p_order in list(engine.pending_orders.items()):
            pc1, pc2, pc3, pc4, pc5 = st.columns([1, 2, 2, 2, 2])
            pc1.write(f"#{ticket}")
            pc2.write(f"**{p_order.symbol}**")
            pc3.write(f"Type: {p_order.order_type}")
            pc4.write(f"Target: {p_order.target_price:.4f}")
            if pc5.button("Cancel", key=f"cancel_{ticket}"):
                engine.delete_pending_order(ticket)
                st.rerun()
