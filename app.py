import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import time

# Configuração da página
st.set_page_config(
    page_title="Muxibaby - AI Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📈 Muxibaby - Trading com IA")
st.markdown("### Análise técnica multi-timeframe com sinais de IA")

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Seleção do símbolo
    symbol = st.text_input("Símbolo", value="BTC-USD", help="Ex: BTC-USD, ETH-USD, AAPL")
    
    # Seleção de timeframes
    st.subheader("Timeframes")
    timeframes = {
        "5 minutos": "5m",
        "15 minutos": "15m",
        "1 hora": "1h",
        "1 dia": "1d"
    }
    
    selected_tf = st.selectbox("Timeframe principal", list(timeframes.keys()))
    
    # Botão de atualização
    if st.button("🔄 Atualizar dados", use_container_width=True):
        st.rerun()

# Função para buscar dados
@st.cache_data(ttl=300)
def get_data(symbol, period="7d", interval="1h"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df
    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        return None

# Função para calcular indicadores
def calculate_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Médias móveis
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# Função para gerar sinal de IA
def generate_ai_signal(df):
    if df is None or len(df) < 50:
        return "NEUTRO", "Dados insuficientes"
    
    last_row = df.iloc[-1]
    
    # Lógica de sinal baseada em indicadores
    signals = []
    
    # RSI
    if last_row['RSI'] < 30:
        signals.append("COMPRA")
    elif last_row['RSI'] > 70:
        signals.append("VENDA")
    
    # Médias móveis
    if last_row['SMA_20'] > last_row['SMA_50']:
        signals.append("COMPRA")
    else:
        signals.append("VENDA")
    
    # MACD
    if last_row['MACD'] > last_row['Signal']:
        signals.append("COMPRA")
    else:
        signals.append("VENDA")
    
    # Decisão final
    buy_count = signals.count("COMPRA")
    sell_count = signals.count("VENDA")
    
    if buy_count > sell_count:
        return "🟢 COMPRA", f"Força do sinal: {buy_count}/{len(signals)}"
    elif sell_count > buy_count:
        return "🔴 VENDA", f"Força do sinal: {sell_count}/{len(signals)}"
    else:
        return "🟡 NEUTRO", "Sinais mistos"

# Main app
try:
    # Buscar dados
    interval_map = {"5 minutos": "5m", "15 minutos": "15m", "1 hora": "1h", "1 dia": "1d"}
    period_map = {"5m": "1d", "15m": "5d", "1h": "7d", "1d": "3mo"}
    
    interval = interval_map[selected_tf]
    period = period_map[interval]
    
    with st.spinner(f"A carregar dados de {symbol}..."):
        df = get_data(symbol, period=period, interval=interval)
    
    if df is not None and not df.empty:
        # Calcular indicadores
        df = calculate_indicators(df)
        
        # Gerar sinal de IA
        signal, strength = generate_ai_signal(df)
        
        # Exibir métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Preço Atual", f"${df['Close'].iloc[-1]:.2f}")
        
        with col2:
            change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
            pct_change = (change / df['Close'].iloc[-2]) * 100
            st.metric("Variação 24h", f"{pct_change:+.2f}%", f"${change:+.2f}")
        
        with col3:
            st.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
        
        with col4:
            st.metric("Sinal IA", signal)
        
        st.info(strength)
        
        # Gráfico principal
        st.subheader("📊 Gráfico de Preços")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol} - Preço', 'Volume', 'RSI')
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Preço'
            ),
            row=1, col=1
        )
        
        # Médias móveis
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        
        # Linhas de referência RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de dados recentes
        st.subheader("📋 Últimos dados")
        st.dataframe(
            df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].tail(10),
            use_container_width=True
        )
        
    else:
        st.error("Não foi possível carregar os dados. Verifica o símbolo e tenta novamente.")
        
except Exception as e:
    st.error(f"Erro: {e}")

# Footer
st.markdown("---")
st.markdown("**Nota:** Esta ferramenta é apenas para fins educacionais. Não constitui aconselhamento financeiro.")
st.markdown("Desenvolvido com ❤️ por Muxibaby")
