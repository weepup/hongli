import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(page_title="红利低波 对数通道分析", layout="wide")
st.title("🟠 红利/红利低波 对数坐标通道分析系统")
st.caption("数据接口：twelvedata.com | 原理：对数线性回归 + 1.5σ 通道 | 完全复刻你那张图")

# ================== 预设基金列表 ==================
etf_list = {
    "中证红利 (515080)": {"symbol": "515080", "exchange": "XSHG"},
    "红利低波 (512890)": {"symbol": "512890", "exchange": "XSHG"},
    "红利低波ETF易方达 (563020)": {"symbol": "563020", "exchange": "XSHG"},
    "红利低波100 (515100)": {"symbol": "515100", "exchange": "XSHG"},
    "红利低波50 (515450)": {"symbol": "515450", "exchange": "XSHG"},
}

# ================== 侧边栏 ==================
with st.sidebar:
    st.header("配置")
    api_key = st.text_input("Twelve Data API Key", type="password", placeholder="输入你的 API Key")
    selected_etf_name = st.selectbox("选择 ETF", options=list(etf_list.keys()))
    etf = etf_list[selected_etf_name]
    symbol = etf["symbol"]
    exchange = etf["exchange"]
    
    st.markdown("---")
    st.info("💡 免费 API 每天有调用限制，建议每天更新一次即可。")

if not api_key:
    st.warning("👆 请在左侧输入你的 Twelve Data API Key")
    st.stop()

# ================== 获取数据 ==================
@st.cache_data(ttl=3600)  # 缓存1小时
def fetch_data(symbol, exchange, api_key, outputsize=5000):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "exchange": exchange,
        "apikey": api_key,
        "outputsize": outputsize,
        "format": "JSON"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error(f"API 请求失败: {response.text}")
        st.stop()
    data = response.json()
    if "values" not in data:
        st.error(f"未获取到数据，可能符号不支持。请检查符号或去 twelvedata.com 验证。错误：{data}")
        st.stop()
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df

df = fetch_data(symbol, exchange, api_key)

# ================== 计算对数回归通道 ==================
df["log_close"] = np.log(df["close"])
x = np.arange(len(df))                    # 交易日序号（更准确）
y = df["log_close"].values

# 线性回归
coeff = np.polyfit(x, y, 1)               # slope, intercept
slope, intercept = coeff
df["trend_log"] = intercept + slope * x
df["trend"] = np.exp(df["trend_log"])

# 残差 & 通道
residuals = y - df["trend_log"]
sigma = np.std(residuals)
df["upper"] = np.exp(df["trend_log"] + 1.5 * sigma)
df["lower"] = np.exp(df["trend_log"] - 1.5 * sigma)

# 年化收益（基于交易日 252 天）
annualized = (np.exp(slope * 252) - 1) * 100

# 最新关键值
latest = df.iloc[-1]
current_date = latest["datetime"].strftime("%Y-%m-%d")
current_price = latest["close"]
trend_price = latest["trend"]
upper_price = latest["upper"]
lower_price = latest["lower"]

# ================== 显示最新关键值（和原图一模一样） ==================
col1, col2, col3, col4 = st.columns(4)
col1.metric("最新净值", f"{current_price:.3f}", f"{current_date}")
col2.metric("趋势值", f"{trend_price:.3f}")
col3.metric("1.5σ 上轨", f"{upper_price:.3f}")
col4.metric("1.5σ 下轨", f"{lower_price:.3f}")

st.success(f"**{selected_etf_name}**  趋势年化收益 ≈ **{annualized:.1f}%**")

# ================== 画图（完全复刻原图风格） ==================
fig = go.Figure()

fig.add_trace(go.Scatter(x=df["datetime"], y=df["close"], 
                         name="基金净值/收盘价", line=dict(color="#1f77b4"), hovertemplate="%{y:.3f}"))

fig.add_trace(go.Scatter(x=df["datetime"], y=df["trend"], 
                         name="对数回归趋势线 (中轨)", line=dict(color="#d62728", dash="dash")))

fig.add_trace(go.Scatter(x=df["datetime"], y=df["upper"], 
                         name="1.5σ 上轨", line=dict(color="#ff7f0e", dash="dot"), 
                         fill=None))

fig.add_trace(go.Scatter(x=df["datetime"], y=df["lower"], 
                         name="1.5σ 下轨", line=dict(color="#ff7f0e", dash="dot"), 
                         fill="tonexty", fillcolor="rgba(255, 165, 0, 0.15)"))

fig.update_layout(
    title=f"{selected_etf_name} 对数坐标通道分析<br>（年化收益 ≈ {annualized:.1f}%）",
    xaxis_title="时间",
    yaxis_title="净值 (对数坐标)",
    yaxis_type="log",
    template="plotly_white",
    height=650,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ================== 额外功能 ==================
st.download_button("下载历史数据 CSV", df.to_csv(index=False), f"{symbol}_channel_data.csv", "text/csv")

st.markdown("---")
st.caption("原理说明：用对数净值做线性回归 → 计算 1.5σ 平行通道。最新净值越靠近下轨越便宜，越靠近上轨越贵。")
st.caption("数据为 ETF 收盘价（与中国基金净值高度一致），中国市场 EOD 数据。")
