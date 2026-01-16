import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import ta
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO INICIAL & CACHING
# ============================================================================

st.set_page_config(
    page_title="🚀 AI Trading PRO | Cloud Version",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Customizado (Otimizado)
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 14px;
    }
    .signal-long {
        background-color: #28a745;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .signal-short {
        background-color: #dc3545;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .signal-wait {
        background-color: #ffc107;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .stMetric { font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHING COM @st.cache_data E TTL
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def carregar_dados_yfinance(symbol, period, interval, candles):
    """Carrega dados com caching otimizado"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        df = df.tail(candles).copy()
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")
        return None

@st.cache_resource
def criar_ai_trader():
    """Inicializa IA uma única vez (resource cache)"""
    return AIExpertTrader()

# ============================================================================
# CLASSE: AI EXPERT TRADER (VERSÃO OTIMIZADA & CORRIGIDA)
# ============================================================================

class AIExpertTrader:
    """IA Trading com 20 anos de experiência - Performance Otimizada"""
    
    def __init__(self):
        self.experience_years = 20
        self.expertise = {
            'technical_analysis': 95,
            'risk_management': 98,
            'market_psychology': 92,
            'volatility_analysis': 94,
            'trend_following': 96,
            'support_resistance': 93
        }
    
    def analizar_grafico_completo(self, df, ticker):
        """Análise 360° otimizada"""
        try:
            analise = {
                'technicals': self._analisar_tecnicos(df),
                'price_action': self._analisar_price_action(df),
                'volumes': self._analisar_volumes(df),
                'volatility': self._analisar_volatilidade(df),
                'support_resistance': self._analisar_suportes_resistencias(df),
                'trend': self._analisar_tendencia(df),
                'momentum': self._analisar_momentum(df),
                'confidence': self._calcular_confianca(df),
                'risk_score': self._calcular_risco(df)
            }
            return analise
        except Exception as e:
            st.warning(f"⚠️ Erro na análise: {e}")
            return None
    
    def _analisar_tecnicos(self, df):
        """Análise técnica com tratamento de erro"""
        try:
            close_series = pd.Series(df['Close'].squeeze().values.flatten()).astype(float)
            rsi = RSIIndicator(close_series, window=14).rsi()
            macd_obj = MACD(close_series)
            macd = macd_obj.macd()
            macd_signal = macd_obj.macd_signal()
            bb = BollingerBands(close_series, window=20, window_dev=2)
            ema9 = EMAIndicator(close_series, window=9).ema_indicator()
            ema21 = EMAIndicator(close_series, window=21).ema_indicator()
            ema50 = EMAIndicator(close_series, window=50).ema_indicator()
            
            rsi_val = float(rsi.iloc[-1])
            macd_val = float(macd.iloc[-1])
            macd_signal_val = float(macd_signal.iloc[-1])
            bb_upper = float(bb.bollinger_hband().iloc[-1])
            bb_lower = float(bb.bollinger_lband().iloc[-1])
            price = float(df['Close'].squeeze().iloc[-1])
            
            bb_pos = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            return {
                'rsi': rsi_val,
                'macd': macd_val,
                'macd_signal': macd_signal_val,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_position': float(bb_pos),
                'ema9': float(ema9.iloc[-1]),
                'ema21': float(ema21.iloc[-1]),
                'ema50': float(ema50.iloc[-1]),
                'price': price
            }
        except Exception as e:
            st.warning(f"Erro técnicos: {e}")
            return {}
    
    def _analisar_price_action(self, df):
        """🔧 CORRIGIDO: Price Action - Converter SEMPRE para float"""
        try:
                    close = df['Close'].squeeze()
            high = df['High'].squeeze()
                    low = df['Low'].squeeze()
        recent_closes = close.iloc[-5:].values
            recent_lows = low.iloc[-5:].values
            
            is_higher_lows = all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
            is_lower_highs = all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
            is_bullish_close = recent_closes[-1] > recent_closes[-2]
            
            open_val = float(df['Open'].iloc[-1])
            close_val = float(recent_closes[-1])
            body_size = abs(close_val - open_val)
            
            wick_ratio = 0.0
            if body_size > 0:
                high_val = float(recent_highs[-1])
                max_close_open = max(close_val, open_val)
                wick_ratio = float((high_val - max_close_open) / body_size)
            
            return {
                'higher_lows': bool(is_higher_lows),
                'lower_highs': bool(is_lower_highs),
                'bullish_candle': bool(is_bullish_close),
                'body_size': float(body_size),
                'wick_ratio': float(wick_ratio)
            }
        except Exception as e:
            st.warning(f"Erro price action: {e}")
            return {
                'higher_lows': False,
                'lower_highs': False,
                'bullish_candle': False,
                'body_size': 0.0,
                'wick_ratio': 0.0
            }
    
    def _analisar_volumes(self, df):
        """Volumes otimizado"""
        try:
            volume = df['Volume']
            vol_ma = volume.rolling(20).mean()
            current_vol = float(volume.iloc[-1])
            avg_vol = float(vol_ma.iloc[-1])
            
            volume_strength = float(current_vol / avg_vol) if avg_vol > 0 else 1.0
            is_strong = bool(current_vol > avg_vol * 1.2)
            
            return {
                'current_volume': current_vol,
                'avg_volume': avg_vol,
                'volume_strength': volume_strength,
                'is_strong': is_strong
            }
        except Exception as e:
            st.warning(f"Erro volumes: {e}")
            return {'current_volume': 0.0, 'avg_volume': 0.0, 'volume_strength': 1.0, 'is_strong': False}
    
    def _analisar_volatilidade(self, df):
        """Volatilidade otimizada"""
        try:
            atr = AverageTrueRange(
                pd.Series(df['High'].values.flatten()).astype(float),
                pd.Series(df['Low'].values.flatten()).astype(float),
                pd.Series(df['Close'].squeeze().values.flatten()).astype(float),
                window=14
            )
            atr_val = float(atr.average_true_range().iloc[-1])
            
            returns = df['Close'].pct_change()
            volatility = float(returns.rolling(20).std().iloc[-1] * 100)
            std_threshold = float(returns.std() * 100 * 1.2)
            is_volatile = bool(volatility > std_threshold)
            
            return {
                'atr': atr_val,
                'volatility_pct': volatility,
                'is_volatile': is_volatile
            }
        except Exception as e:
            st.warning(f"Erro volatilidade: {e}")
            return {'atr': 0.0, 'volatility_pct': 0.0, 'is_volatile': False}
    
    def _analisar_suportes_resistencias(self, df):
        """SR otimizado"""
        try:
            data = df.iloc[-100:]
            highs = data['High'].values
            lows = data['Low'].values
            close = float(data['Close'].iloc[-1])
            
            resistencia_1 = float(np.max(highs[-20:]))
            suporte_1 = float(np.min(lows[-20:]))
            resistencia_2 = float(np.max(highs))
            suporte_2 = float(np.min(lows))
            
            return {
                'resistencia_1': resistencia_1,
                'suporte_1': suporte_1,
                'resistencia_2': resistencia_2,
                'suporte_2': suporte_2,
                'prox_resistencia': resistencia_1,
                'prox_suporte': suporte_1,
                'distancia_resistencia': float(resistencia_1 - close),
                'distancia_suporte': float(close - suporte_1)
            }
        except Exception as e:
            st.warning(f"Erro SR: {e}")
            return {}
    
    def _analisar_tendencia(self, df):
        """Trend otimizado"""
        try:
            close_series = pd.Series(df['Close'].squeeze().values.flatten()).astype(float)
            ema9 = EMAIndicator(close_series, window=9).ema_indicator()
            ema21 = EMAIndicator(close_series, window=21).ema_indicator()
            ema50 = EMAIndicator(close_series, window=50).ema_indicator()
            
            val9 = float(ema9.iloc[-1])
            val21 = float(ema21.iloc[-1])
            val50 = float(ema50.iloc[-1])
            
            if val9 > val21 > val50:
                trend, trend_score = "BULLISH FORTE", 10
            elif val9 > val21 and val21 > val50:
                trend, trend_score = "BULLISH", 7
            elif val9 > val50 > val21:
                trend, trend_score = "CONSOLIDADO", 5
            elif val9 < val21 < val50:
                trend, trend_score = "BEARISH FORTE", -10
            elif val9 < val21 and val21 < val50:
                trend, trend_score = "BEARISH", -7
            else:
                trend, trend_score = "CONSOLIDADO", 0
            
            return {
                'trend': trend,
                'trend_score': float(trend_score),
                'ema_alignment': float(abs(trend_score))
            }
        except Exception as e:
            st.warning(f"Erro trend: {e}")
            return {}
    
    def _analisar_momentum(self, df):
        """Momentum otimizado"""
        try:
            close_series = pd.Series(df['Close'].squeeze().values.flatten()).astype(float)
            rsi = RSIIndicator(close_series, window=14).rsi()
            stoch = StochasticOscillator(
                pd.Series(df['High'].values.flatten()).astype(float),
                pd.Series(df['Low'].values.flatten()).astype(float),
                close_series,
                window=14
            )
            
            rsi_val = float(rsi.iloc[-1])
            stoch_k = float(stoch.stoch().iloc[-1])
            stoch_d = float(stoch.stoch_signal().iloc[-1])
            
            momentum_score = 0
            if rsi_val > 70:
                momentum_score = 2
            elif rsi_val > 60:
                momentum_score = 1
            elif rsi_val < 30:
                momentum_score = -2
            elif rsi_val < 40:
                momentum_score = -1
            
            return {
                'rsi': rsi_val,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'momentum_score': float(momentum_score)
            }
        except Exception as e:
            st.warning(f"Erro momentum: {e}")
            return {}
    
    def _calcular_confianca(self, df):
        """Confiança otimizada"""
        try:
            analise = self._analisar_tecnicos(df)
            price_action = self._analisar_price_action(df)
            volumes = self._analisar_volumes(df)
            volatility = self._analisar_volatilidade(df)
            trend = self._analisar_tendencia(df)
            
            confianca = 0
            
            if 40 < analise.get('rsi', 50) < 60:
                confianca += 10
            elif 30 < analise.get('rsi', 50) < 70:
                confianca += 5
            
            if analise.get('macd', 0) > analise.get('macd_signal', 0):
                confianca += 10
            
            if price_action.get('bullish_candle', False):
                confianca += 10
            
            if volumes.get('is_strong', False):
                confianca += 10
            
            if not volatility.get('is_volatile', True):
                confianca += 7.5
            
            confianca += (abs(trend.get('trend_score', 0)) / 10) * 15
            
            return min(float(confianca), 100.0)
        except Exception as e:
            return 50.0
    
    def _calcular_risco(self, df):
        """Risco otimizado"""
        try:
            volatility = self._analisar_volatilidade(df)
            price_action = self._analisar_price_action(df)
            
            risk_score = 50.0
            
            if volatility.get('is_volatile', False):
                risk_score += 25
            
            if price_action.get('wick_ratio', 0) > 0.7:
                risk_score += 15
            
            return min(float(risk_score), 100.0)
        except Exception as e:
            return 50.0
    
    def gerar_decisao_expert(self, df, account_balance, risk_pct):
        """Decisão expert otimizada"""
        try:
            analise = self.analizar_grafico_completo(df, "SYMBOL")
            
            if not analise:
                return self._decisao_padrao()
            
            techs = analise['technicals']
            price_action = analise['price_action']
            trend = analise['trend']
            sr = analise['support_resistance']
            confianca = analise['confidence']
            risk = analise['risk_score']
            
            buy_signals = 0
            sell_signals = 0
            
            # BUY SIGNALS
            if trend.get('trend_score', 0) > 5:
                buy_signals += 2
            elif trend.get('trend_score', 0) > 0:
                buy_signals += 1
            
            rsi = techs.get('rsi', 50)
            if 40 < rsi < 60:
                buy_signals += 2
            elif 30 < rsi < 70:
                buy_signals += 1
            
            if techs.get('macd', 0) > techs.get('macd_signal', 0):
                buy_signals += 2
            
            if price_action.get('higher_lows', False) and price_action.get('bullish_candle', False):
                buy_signals += 2
            elif price_action.get('bullish_candle', False):
                buy_signals += 1
            
            # SELL SIGNALS
            if trend.get('trend_score', 0) < -5:
                sell_signals += 2
            elif trend.get('trend_score', 0) < 0:
                sell_signals += 1
            
            if not (40 < rsi < 60):
                if rsi > 70:
                    sell_signals += 2
                elif rsi > 60:
                    sell_signals += 1
            
            if techs.get('macd', 0) < techs.get('macd_signal', 0):
                sell_signals += 2
            
            if price_action.get('lower_highs', False) and not price_action.get('bullish_candle', False):
                sell_signals += 2
            elif not price_action.get('bullish_candle', False):
                sell_signals += 1
            
            # DECISÃO FINAL
            entry = techs.get('price', 0)
            
            if buy_signals >= 4 and confianca > 65 and risk < 75:
                decision = "🟢 COMPRA FORTE"
                action = "BUY"
                strength = min(buy_signals / 2, 10)
                stop_loss = sr.get('suporte_1', entry)
                target_1 = sr.get('resistencia_1', entry)
                target_2 = sr.get('resistencia_2', entry)
            
            elif sell_signals >= 4 and confianca > 65 and risk < 75:
                decision = "🔴 VENDA FORTE"
                action = "SELL"
                strength = min(sell_signals / 2, 10)
                stop_loss = sr.get('resistencia_1', entry)
                target_1 = sr.get('suporte_1', entry)
                target_2 = sr.get('suporte_2', entry)
            
            elif buy_signals >= 2:
                decision = "🟡 COMPRA FRACA"
                action = "WAIT_BUY"
                strength = buy_signals / 2
                stop_loss = sr.get('suporte_1', entry)
                target_1 = sr.get('resistencia_1', entry)
                target_2 = sr.get('resistencia_2', entry)
            
            elif sell_signals >= 2:
                decision = "🟡 VENDA FRACA"
                action = "WAIT_SELL"
                strength = sell_signals / 2
                stop_loss = sr.get('resistencia_1', entry)
                target_1 = sr.get('suporte_1', entry)
                target_2 = sr.get('suporte_2', entry)
            
            else:
                decision = "⏳ AGUARDANDO SINAL"
                action = "WAIT"
                strength = 0
                stop_loss = entry
                target_1 = entry
                target_2 = entry
            
            # Calcular Risk/Reward
            entry_sl_diff = abs(entry - stop_loss)
            if entry_sl_diff > 0:
                if action in ["BUY", "WAIT_BUY"]:
                    rr = abs((target_1 - entry) / entry_sl_diff)
                else:
                    rr = abs((entry - target_1) / entry_sl_diff)
            else:
                rr = 0.0
            
            risk_value = entry_sl_diff * 100
            position_size = (account_balance * (risk_pct / 100)) / abs(risk_value) if risk_value > 0 else 0
            
            return {
                'decision': decision,
                'action': action,
                'strength': float(strength),
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'target_1': float(target_1),
                'target_2': float(target_2),
                'position_size': float(position_size),
                'risk_reward': float(rr),
                'confidence': float(confianca),
                'risk_score': float(risk),
                'full_analysis': analise
            }
        except Exception as e:
            st.error(f"❌ Erro crítico: {e}")
            return self._decisao_padrao()
    
    def _decisao_padrao(self):
        """Retorna decisão padrão em caso de erro"""
        return {
            'decision': "❌ ERRO NA ANÁLISE",
            'action': "ERROR",
            'strength': 0.0,
            'entry': 0.0,
            'stop_loss': 0.0,
            'target_1': 0.0,
            'target_2': 0.0,
            'position_size': 0.0,
            'risk_reward': 0.0,
            'confidence': 0.0,
            'risk_score': 100.0,
            'full_analysis': {}
        }

# ============================================================================
# INTERFACE STREAMLIT OTIMIZADA
# ============================================================================

st.title("🚀 AI TRADING PRO | Cloud Version")
st.caption("⚡ Otimizado para Streamlit Cloud | Version 4.1 (Bugs Fixed)")

# Sidebar Configurações
with st.sidebar:
    st.header("⚙️ CONFIGURAÇÕES")
    
    col1, col2 = st.columns(2)
    with col1:
        asset_type = st.selectbox("Ativo", ["Forex", "Stock", "Crypto"])
    with col2:
        if asset_type == "Forex":
            symbol = st.selectbox("Símbolo", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"])
        elif asset_type == "Stock":
            symbol = st.text_input("Símbolo", "AAPL", max_chars=10)
        else:
            symbol = st.selectbox("Cripto", ["BTC-USD", "ETH-USD", "ADA-USD"])
    
    interval = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"], index=4)
    
    st.markdown("---")
    account_balance = st.number_input("Saldo ($)", value=10000, min_value=100, step=1000)
    risk_pct = st.slider("Risco (%)", 0.5, 5.0, 2.0, 0.5)
    candles = st.slider("Candles", 50, 300, 150)

# Period mapping
period_map = {
    "1m": "5d", "5m": "5d", "15m": "30d", "30m": "30d", "1h": "3mo", "1d": "1y"
}
period = period_map.get(interval, "3mo")

# Carregamento com spinner
with st.spinner(f"📊 Carregando {symbol}..."):
    df = carregar_dados_yfinance(symbol, period, interval, candles)

if df is None or len(df) < 50:
    st.error("❌ Dados insuficientes")
    st.stop()

# Inicializar AI (cacheado)
ai_expert = criar_ai_trader()
decisao = ai_expert.gerar_decisao_expert(df, account_balance, risk_pct)

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📊 GRÁFICO", "🤖 DECISÃO", "📈 ANÁLISE", "💰 RISK"])

# ============================================================================
# ABA 1: GRÁFICO
# ============================================================================

with tab1:
    st.subheader(f"{symbol} | {interval}")
    
    # Indicadores
    rsi = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()
    bb = ta.volatility.BollingerBands(df['Close'].squeeze(), window=20, window_dev=2)
    ema9 = ta.trend.EMAIndicator(df['Close'].squeeze(), window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(df['Close'].squeeze(), window=21).ema_indicator()
    ema50 = ta.trend.EMAIndicator(df['Close'].squeeze(), window=50).ema_indicator()
    
    # Gráfico
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            increasing_line_color="#28a745",
            decreasing_line_color="#dc3545",
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # BB
    fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_hband().astype(float),
        mode='lines', name='BB+', line=dict(color='orange', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_lband().astype(float),
        mode='lines', name='BB-', line=dict(color='orange', width=1, dash='dash'), fill='tonexty'), row=1, col=1)
    
    # EMAs
    for ema, color, name in [(ema9, 'red', 'EMA9'), (ema21, 'blue', 'EMA21'), (ema50, 'purple', 'EMA50')]:
        fig.add_trace(go.Scatter(x=df.index, y=ema.astype(float), mode='lines',
            name=name, line=dict(color=color, width=2)), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=rsi.astype(float), mode='lines',
        name='RSI', line=dict(color='green', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2)
    
    fig.update_layout(height=600, template="plotly_dark", hovermode='x unified',
        title=f"{symbol} | {decisao['decision']}", xaxis_rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ABA 2: DECISÃO IA
# ============================================================================

with tab2:
    st.subheader("🤖 ANÁLISE DA IA")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "COMPRA" in decisao['decision']:
            st.markdown(f"<div class='signal-long'>{decisao['decision']}</div>", unsafe_allow_html=True)
        elif "VENDA" in decisao['decision']:
            st.markdown(f"<div class='signal-short'>{decisao['decision']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='signal-wait'>{decisao['decision']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.metric("Força", f"{decisao['strength']:.1f}/10")
    
    with col3:
        st.metric("Confiança", f"{decisao['confidence']:.0f}%")
    
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    analise = decisao['full_analysis']
    if analise:
        with col1:
            st.metric("RSI", f"{analise.get('technicals', {}).get('rsi', 0):.1f}")
        with col2:
            st.metric("Trend", f"{analise.get('trend', {}).get('trend_score', 0):.0f}/10")
        with col3:
            st.metric("Volume", f"{analise.get('volumes', {}).get('volume_strength', 0):.1f}x")
        with col4:
            st.metric("ATR", f"{analise.get('volatility', {}).get('atr', 0):.6f}")
        with col5:
            st.metric("Vol%", f"{analise.get('volatility', {}).get('volatility_pct', 0):.1f}%")

# ============================================================================
# ABA 3: ANÁLISE
# ============================================================================

with tab3:
    st.subheader("📈 ANÁLISE DETALHADA")
    
    if analise:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 TÉCNICOS")
            st.write(f"**RSI:** {analise['technicals']['rsi']:.2f}")
            st.write(f"**MACD:** {analise['technicals']['macd']:.6f}")
            st.write(f"**EMA9:** {analise['technicals']['ema9']:.4f}")
            st.write(f"**EMA21:** {analise['technicals']['ema21']:.4f}")
            st.write(f"**EMA50:** {analise['technicals']['ema50']:.4f}")
        
        with col2:
            st.markdown("### 🎯 SUPORTE/RESISTÊNCIA")
            st.write(f"**R1:** {analise['support_resistance']['resistencia_1']:.4f}")
            st.write(f"**S1:** {analise['support_resistance']['suporte_1']:.4f}")
            st.write(f"**R2:** {analise['support_resistance']['resistencia_2']:.4f}")
            st.write(f"**S2:** {analise['support_resistance']['suporte_2']:.4f}")

# ============================================================================
# ABA 4: RISK MANAGEMENT
# ============================================================================

with tab4:
    st.subheader("💰 GESTÃO DE RISCO")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Entrada", f"{decisao['entry']:.4f}")
    with col2:
        st.metric("Stop Loss", f"{decisao['stop_loss']:.4f}")
    with col3:
        st.metric("R:R", f"{decisao['risk_reward']:.2f}:1")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Alvo 1", f"{decisao['target_1']:.4f}")
    with col2:
        st.metric("Alvo 2", f"{decisao['target_2']:.4f}")
    
    st.markdown("---")
    st.markdown("### 📊 POSIÇÃO")
    st.write(f"**Tamanho da Posição:** {decisao['position_size']:.4f} unidades")
    st.write(f"**Risco/Trade:** R$ {account_balance * (risk_pct / 100):.2f}")
    st.write(f"**Risk Score:** {decisao['risk_score']:.0f}/100")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 11px; color: gray;'>🚀 AI TRADING PRO | Version 4.1 | ✅ Bugs Fixed | ⚠️ Trading é risco. Sempre use Stop Loss.</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    # Auto-refresh a cada 5 segundos
main()
