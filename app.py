import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="红利低波 对数通道分析", layout="wide")
st.title("🟠 红利/红利低波 对数坐标通道分析系统")
st.caption("数据接口：Yahoo Finance (yfinance) | 原理：对数线性回归 + 1.5σ 通道 | 已修复 KeyError")

# ================== 预设基金列表 ==================
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
    st.info("💡 数据自动更新，无需 API Key")

# ================== 获取数据（已加固） ==================
@st.cache_data(ttl=3600)
def fetch_data(ticker):
    try:
        # 使用 Ticker.history，更稳定
        df = yf.Ticker(ticker).history(period="max", interval="1d")
        if df.empty:
            st.error(f"❌ {ticker} 暂无数据，请稍后重试或检查网络。")
            st.stop()
        
        # 重置索引并处理各种可能的列名情况
        df = df.reset_index()
        
        # 处理可能的 MultiIndex（极少数情况）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # 统一列名
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "datetime"})
        elif "datetime" in df.columns:
            pass  # 已有
        
        # 优先使用 Close，其次 Adj Close
        if "Close" in df.columns:
            df = df.rename(columns={"Close": "close"})
        elif "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "close"})
        else:
            st.error(f"❌ 数据列异常！当前列名：{list(df.columns)}")
            st.error("请把上面这行列名截图发给我，我马上修复。")
            st.stop()
        
        df = df[["datetime", "close"]].dropna(subset=["close"]).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"数据获取失败: {str(e)}")
        st.stop()

df = fetch_data(ticker)

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

st.markdown("---")
st.caption("原理：对数净值线性回归 + 1.5σ 通道")
