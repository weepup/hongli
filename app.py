import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="红利低波 对数通道分析", layout="wide")
st.title("🟠 红利/红利低波 对数坐标通道分析系统")
st.caption("数据接口：Yahoo Finance | 原理：对数线性回归 + 1.5σ 通道 | 已升级交易信号灯")

# ================== ETF 列表 ==================
etf_list = {
    "中证红利 (515080)": "515080.SS",
    "红利低波 (512890)": "512890.SS",
    "红利低波ETF易方达 (563020)": "563020.SS",
    "红利低波100 (515100)": "515100.SS",
    "红利低波50 (515450)": "515450.SS",
}

# ================== 侧边栏 ==================
with st.sidebar:
    st.header("配置")
    selected_etf_name = st.selectbox("选择 ETF", options=list(etf_list.keys()))
    ticker = etf_list[selected_etf_name]
    
    st.subheader("数据周期")
    period_option = st.selectbox(
        "选择回溯周期",
        ["全历史", "最近 2 年", "最近 3 年", "自定义起始日期"],
        index=3  # 默认自定义，更贴近当前市场
    )
    if period_option == "自定义起始日期":
        start_date_input = st.date_input(
            "起始日期",
            value=pd.Timestamp.now() - pd.DateOffset(years=3),
            min_value=pd.Timestamp("2010-01-01")
        )
    st.info("💡 推荐用「最近 3 年」或自定义，能让趋势线更贴近当前市场")

# ================== 获取数据 ==================
@st.cache_data(ttl=3600)
def fetch_data(ticker):
    df = yf.Ticker(ticker).history(period="max", interval="1d")
    if df.empty:
        st.error(f"❌ {ticker} 暂无数据")
        st.stop()
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "datetime"})
    if "Close" in df.columns:
        df = df.rename(columns={"Close": "close"})
    elif "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "close"})
    df = df[["datetime", "close"]].dropna(subset=["close"]).reset_index(drop=True)
    return df

df_full = fetch_data(ticker)

# ================== 根据周期过滤数据 ==================
if period_option == "全历史":
    df = df_full.copy()
elif period_option == "最近 2 年":
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=2)
    df = df_full[df_full["datetime"] >= cutoff].copy()
elif period_option == "最近 3 年":
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=3)
    df = df_full[df_full["datetime"] >= cutoff].copy()
else:
    cutoff = pd.to_datetime(start_date_input)
    df = df_full[df_full["datetime"] >= cutoff].copy()

if len(df) < 252:
    st.warning("⚠️ 所选周期数据不足 1 年，回归结果可能不稳定，仅供参考。")

# ================== 计算对数回归通道 ==================
df["log_close"] = np.log(df["close"])
x = np.arange(len(df))
y = df["log_close"].values

coeff = np.polyfit(x, y, 1)
slope, intercept = coeff
df["trend_log"] = intercept + slope * x
df["trend"] = np.exp(df["trend_log"])

residuals = y - df["trend_log"]
sigma = np.std(residuals)
df["upper"] = np.exp(df["trend_log"] + 1.5 * sigma)
df["lower"] = np.exp(df["trend_log"] - 1.5 * sigma)

annualized = (np.exp(slope * 252) - 1) * 100

# 最新关键值
latest = df.iloc[-1]
current_date = latest["datetime"].strftime("%Y-%m-%d")
current_price = latest["close"]
trend_price = latest["trend"]
upper_price = latest["upper"]
lower_price = latest["lower"]

# ================== 显示关键值 ==================
col1, col2, col3, col4 = st.columns(4)
col1.metric("最新净值", f"{current_price:.3f}", f"{current_date}")
col2.metric("趋势值", f"{trend_price:.3f}")
col3.metric("1.5σ 上轨", f"{upper_price:.3f}")
col4.metric("1.5σ 下轨", f"{lower_price:.3f}")

st.success(f"**{selected_etf_name}**  趋势年化收益 ≈ **{annualized:.1f}%**")

# ================== 交易信号灯 + 建议 ==================
price_position = (current_price - lower_price) / (upper_price - lower_price)  # 0~1

if price_position <= 0.15:
    signal_emoji = "🟢"
    signal_text = "强买信号"
    suggestion = "最新净值**接近或低于下轨**，安全边际极高！建议**分批加仓**（每次 20-30% 仓位）。"
elif price_position <= 0.45:
    signal_emoji = "🟢"
    signal_text = "买入/轻加"
    suggestion = "净值处于通道**下半部**，性价比好。建议**小批加仓**或维持现有仓位。"
elif price_position <= 0.75:
    signal_emoji = "🟡"
    signal_text = "持有"
    suggestion = "净值在**趋势线附近**，正常波动。**继续持有**观察即可。"
else:
    signal_emoji = "🔴"
    signal_text = "减仓信号"
    suggestion = "最新净值**接近或高于上轨**，估值偏高。建议**分批减仓锁定收益**，等回落再加。"

# 趋势斜率额外提醒
if annualized < 5:
    suggestion += "\n\n⚠️ 注意：当前长期趋势走平/向下，整体仓位请控制在 60% 以内。"

st.subheader("📊 交易信号灯 & 操作建议")
st.markdown(f"### {signal_emoji} **{signal_text}**")
st.info(suggestion)

# ================== 画图 ==================
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["datetime"], y=df["close"], name="基金净值", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=df["datetime"], y=df["trend"], name="对数回归趋势线 (中轨)", line=dict(color="#d62728", dash="dash")))
fig.add_trace(go.Scatter(x=df["datetime"], y=df["upper"], name="1.5σ 上轨", line=dict(color="#ff7f0e", dash="dot")))
fig.add_trace(go.Scatter(x=df["datetime"], y=df["lower"], name="1.5σ 下轨", line=dict(color="#ff7f0e", dash="dot"), fill="tonexty", fillcolor="rgba(255, 165, 0, 0.15)"))

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

st.download_button("下载历史数据 CSV", df.to_csv(index=False), f"{selected_etf_name}_channel_data.csv", "text/csv")

st.caption("信号仅供参考 · 坚持分批买卖 · 非实时交易指令")
