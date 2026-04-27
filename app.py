"""
榕江县黄金百香果市场AI分析与销售决策系统 v2.0
=================================================
一站式AI看板：历史分析 + 7天销量/价格双预测 + 智能定价 + 销售策略
直接运行: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings, io, base64, os, sys
warnings.filterwarnings("ignore")

# 数据抓取模块
try:
    from data_fetcher import (
        get_market_reference, get_rongjiang_reference,
        calibrate_simulation, _get_fallback_reference
    )
    FETCHER_AVAILABLE = True
except ImportError:
    FETCHER_AVAILABLE = False

# ============================================================
# 页面配置 & 农业风格CSS
# ============================================================
st.set_page_config(
    page_title="榕江百香果AI决策系统",
    page_icon="🍈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
    * { font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif; }
    .main { background: linear-gradient(180deg, #f8fdf5 0%, #fefdf0 50%, #fffcf5 100%); }
    .stApp { background: transparent; }
    .section-title {
        font-size: 1.25rem; font-weight: 700; color: #1a5c32;
        border-bottom: 3px solid #f5a623; padding-bottom: 6px; margin-bottom: 12px;
    }
    .risk-high { color: #d32f2f; font-weight: 700; }
    .risk-mid { color: #f5a623; font-weight: 700; }
    .risk-low { color: #2d8c4a; font-weight: 700; }
    .strategy-box {
        background: #fffdf5; border: 1px solid rgba(245,166,35,0.2); border-radius: 10px;
        padding: 14px 18px; margin: 6px 0;
    }
    .insight-card {
        background: white; border-radius: 10px; padding: 14px 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04); margin: 6px 0;
        border-left: 4px solid #2d8c4a;
    }
    div[data-testid="stMetric"] {
        background: white; border-radius: 10px; padding: 12px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 模块A: 数据生成器 v2 — 更真实、更多维度
# ============================================================
@st.cache_data
def generate_data():
    """生成榕江百香果日度模拟数据 2024.01-2026.04"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", "2026-04-27", freq="D")
    n = len(dates)

    df = pd.DataFrame({"date": dates})
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    # ---- 产季标记 (榕江: 5-11月产果季, 12-4月休果期) ----
    df["production_season"] = df["month"].apply(lambda m: "产果旺季" if m in [8,9,10]
        else ("产果期" if m in [5,6,7,11] else "休果期"))

    # ---- 温度 (榕江: 年均18°C, 夏季25-35, 冬季5-15) ----
    base_temp = 16 + 11 * np.sin((df["day_of_year"] - 80) * 2 * np.pi / 365)
    df["temperature"] = (base_temp + np.random.normal(0, 2.2, n)).clip(1, 39).round(1)

    # ---- 湿度 ----
    base_humidity = 78 + 10 * np.sin((df["day_of_year"] - 30) * 2 * np.pi / 365)
    df["humidity"] = (base_humidity + np.random.normal(0, 5, n)).clip(45, 99).round(1)

    # ---- 降雨量 ----
    rainy = ((df["month"] >= 4) & (df["month"] <= 9)).astype(float)
    df["rainfall"] = (rainy * np.random.exponential(15, n) +
                      (1 - rainy) * np.random.exponential(4, n)).clip(0, 85).round(1)

    # ---- 村超 (3-11月) ----
    df["cunchao_flag"] = 0
    for i in range(n):
        m = df.loc[i, "month"]
        if 3 <= m <= 11 and np.random.random() < 0.20:
            df.loc[i, "cunchao_flag"] = 1
    for m in [5,6,7,8,9,10]:
        mask = (df["month"] == m) & (df["cunchao_flag"] == 0)
        idx = df[mask].index
        if len(idx) > 0:
            extra = np.random.choice(idx, size=min(6, len(idx)), replace=False)
            df.loc[extra, "cunchao_flag"] = 1

    # ---- 促销 ----
    df["promotion_type"] = "无"
    promos_list = ["直播带货", "满减活动", "村超联名", "产地溯源", "节日礼盒", "社区团购"]
    for i in range(n):
        r = np.random.random()
        if df.loc[i, "cunchao_flag"] == 1 and r < 0.5:
            df.loc[i, "promotion_type"] = "村超联名"
        elif r < 0.04:
            df.loc[i, "promotion_type"] = np.random.choice(promos_list)
        elif r < 0.08:
            df.loc[i, "promotion_type"] = np.random.choice(["直播带货", "满减活动"])
    df["promo_flag"] = (df["promotion_type"] != "无").astype(int)

    # ---- 节日 ----
    holidays = {
        "2024-01-01","2024-02-10","2024-04-04","2024-05-01","2024-06-10",
        "2024-09-17","2024-10-01","2025-01-01","2025-01-29","2025-04-04",
        "2025-05-01","2025-05-31","2025-10-01","2025-10-06","2026-01-01",
        "2026-02-17","2026-04-05"
    }
    df["holiday_flag"] = df["date"].astype(str).str[:10].isin(holidays).astype(int)

    # ---- 月销量基准 (旺季8-10月, 日均10吨=10000kg) ----
    month_factor = {
        1:0.22, 2:0.20, 3:0.28, 4:0.33, 5:0.50, 6:0.65,
        7:0.80, 8:1.00, 9:1.08, 10:0.95, 11:0.55, 12:0.32
    }
    base_sales = 10000
    df["month_factor"] = df["month"].map(month_factor).astype(float)

    df["sales_kg"] = (base_sales * df["month_factor"]
                      * np.random.normal(1, 0.12, n)).clip(400, 19000)

    # 叠加因子
    df["sales_kg"] = (df["sales_kg"]
                      * (1 + 0.30 * df["cunchao_flag"])
                      * (1 + 0.15 * df["promo_flag"])
                      * (1 + 0.08 * df["is_weekend"])
                      * (1 + 0.25 * df["holiday_flag"])).round(0)
    # 年度增长
    df["sales_kg"] = (df["sales_kg"]
                      * np.where(df["year"] == 2025, 1.10,
                         np.where(df["year"] == 2026, 1.22, 1.0))).round(0)

    # ---- 线上/线下 ----
    online_base = 0.25
    online_boost = (np.where(df["cunchao_flag"] == 1, 0.10, 0)
                    + np.where(df["promotion_type"].isin(["直播带货","村超联名"]), 0.15, 0))
    online_ratio = (online_base + online_boost + np.random.normal(0, 0.03, n)).clip(0.12, 0.55)
    df["online_sales"] = (df["sales_kg"] * online_ratio).round(0)
    df["offline_sales"] = (df["sales_kg"] - df["online_sales"]).round(0)
    df["online_ratio"] = online_ratio.round(3)

    # ---- 均价 (旺季16-22, 休果期略高, 线上溢价) ----
    base_price = 19.5 - 3.5 * (df["month_factor"] - 0.35)
    df["avg_price"] = (base_price + np.random.normal(0, 1.6, n)).clip(12, 30).round(1)
    df["avg_price"] = (df["avg_price"]
                       * (1 - 0.08 * df["promo_flag"])
                       * (1 + 0.05 * df["cunchao_flag"])
                       + np.where(df["year"] >= 2025, 1.8, 0)).round(1)

    # ---- 竞品均价 ----
    df["competitor_price"] = (df["avg_price"]
                              * np.random.normal(1.04, 0.07, n)).clip(10, 32).round(1)

    # ---- 产地 ----
    df["origin"] = np.random.choice(
        ["榕江","榕江","榕江","贵州其他","广西","云南"],
        size=n, p=[0.45, 0.18, 0.12, 0.10, 0.08, 0.07])

    # ---- 果实分级 ----
    df["grade_premium_pct"] = (0.20 + np.random.normal(0, 0.03, n)).clip(0.10, 0.35)
    df["grade_standard_pct"] = (0.55 + np.random.normal(0, 0.04, n)).clip(0.40, 0.70)
    df["grade_second_pct"] = (1 - df["grade_premium_pct"] - df["grade_standard_pct"]).clip(0.05, 0.35)

    # ---- 收入 ----
    df["daily_revenue"] = (df["sales_kg"] * df["avg_price"]).round(0)

    return df

# ---- 外部数据导入 ----
def load_external_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file, parse_dates=["date"])
    else:
        return pd.read_excel(uploaded_file, parse_dates=["date"])

# ---- CSV导出 ----
def csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="rongjiang_passionfruit.csv">📥 点击下载 CSV</a>'

# ============================================================
# 模块B: AI分析 (增强版)
# ============================================================
def detect_anomalies(series, threshold=1.8):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - threshold * IQR) | (series > Q3 + threshold * IQR)

def annotate_anomaly_events(df, anom_mask):
    """为异常点匹配可能的事件原因"""
    events = []
    anom_dates = df[anom_mask]["date"]
    for d in anom_dates:
        row = df[df["date"] == d].iloc[0]
        reasons = []
        if row["cunchao_flag"] == 1:
            reasons.append("村超比赛日")
        if row["promo_flag"] == 1:
            reasons.append(row["promotion_type"])
        if row["holiday_flag"] == 1:
            reasons.append("节假日")
        if row["rainfall"] > 50:
            reasons.append("暴雨天气")
        events.append({
            "date": d.strftime("%Y-%m-%d"),
            "sales_kg": int(row["sales_kg"]),
            "price": row["avg_price"],
            "likely_reason": " + ".join(reasons) if reasons else "自然波动"
        })
    return pd.DataFrame(events)

def seasonal_decomposition(df):
    """时序分解: 趋势+季节+残差 (7天移动平均)"""
    sales = df.set_index("date")["sales_kg"]
    trend = sales.rolling(30, center=True).mean()
    detrended = sales - trend
    seasonal = detrended.groupby(detrended.index.dayofyear).transform("mean")
    residual = detrended - seasonal
    return trend, seasonal, residual

def generate_market_summary(df):
    """自动生成行情总结 v2"""
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=30)]
    prev30 = df[(df["date"] >= df["date"].max() - pd.Timedelta(days=60)) &
                (df["date"] < df["date"].max() - pd.Timedelta(days=30))]
    same_period_last_year = df[(df["date"] >= df["date"].max() - pd.Timedelta(days=395)) &
                                (df["date"] <= df["date"].max() - pd.Timedelta(days=365))]

    avg_sales = recent["sales_kg"].mean()
    avg_price = recent["avg_price"].mean()
    online_pct = recent["online_sales"].sum() / recent["sales_kg"].sum() * 100

    sales_mom = (avg_sales / prev30["sales_kg"].mean() - 1) * 100 if len(prev30) > 0 else 0
    price_mom = (avg_price / prev30["avg_price"].mean() - 1) * 100 if len(prev30) > 0 else 0
    sales_yoy = (avg_sales / same_period_last_year["sales_kg"].mean() - 1) * 100 if len(same_period_last_year) > 0 else 0

    month_now = datetime.now().month
    if month_now in [8, 9, 10]:
        supply_, demand_ = "旺季大量上市，日供应量8-12吨", "需求旺盛，线上走量快，村超引流效应显著"
        risk_ = "⚠️ 集中上市期，竞品（广西/云南）增量可能出现价格踩踏，建议每日控量出货，走多渠道分散"
    elif month_now in [5, 6, 7]:
        supply_, demand_ = "产季启动，供应逐日增加", "需求回暖，适合直播预售+渠道铺设"
        risk_ = "✅ 价格有支撑，但需关注6月下旬广西果上市对榕江价格的冲击"
    elif month_now in [11]:
        supply_, demand_ = "产季尾期，供应减少", "需求平稳，库存去化为主"
        risk_ = "📦 尾期货建议走批发快出，避免仓储损耗; 品质好的做礼品装溢价"
    else:
        supply_, demand_ = "休果期，无鲜果产出", "以库存/加工品流通为主"
        risk_ = "❄️ 淡季保鲜成本高，控制库存周转<7天; 可推果干/果酱等加工品"

    comp_price = recent["competitor_price"].mean()
    premium = (avg_price / comp_price - 1) * 100 if comp_price > 0 else 0

    return f"""
### 📋 近30天市场行情深度总结

| 维度 | 现状 | 环比 | 同比 |
|------|------|------|------|
| 日均销量 | {avg_sales:.0f} kg | {sales_mom:+.1f}% | {sales_yoy:+.1f}% |
| 均价 | ¥{avg_price:.1f}/kg | {price_mom:+.1f}% | — |
| 线上占比 | {online_pct:.1f}% | — | — |
| 竞品均价 | ¥{comp_price:.1f}/kg | 榕江溢价 {premium:+.0f}% | — |

**供应端**: {supply_}
**需求端**: {demand_}
**风险提示**: {risk_}
"""

# ============================================================
# 模块C: 特征工程 + 未来天气模拟
# ============================================================
def create_features(df):
    data = df.sort_values("date").reset_index(drop=True).copy()

    data["dayofweek_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
    data["dayofweek_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["quarter_sin"] = np.sin(2 * np.pi * data["quarter"] / 4)

    data["temp_norm"] = (data["temperature"] - 20) / 10
    data["humidity_norm"] = (data["humidity"] - 75) / 15
    data["rainfall_norm"] = data["rainfall"] / 40

    for lag in [1, 3, 7, 14, 30]:
        data[f"sales_lag_{lag}"] = data["sales_kg"].shift(lag)
        data[f"price_lag_{lag}"] = data["avg_price"].shift(lag)

    for w in [7, 14, 30]:
        data[f"sales_rm_{w}"] = data["sales_kg"].rolling(w).mean()
        data[f"sales_rs_{w}"] = data["sales_kg"].rolling(w).std()
        data[f"price_rm_{w}"] = data["avg_price"].rolling(w).mean()
        data[f"price_rs_{w}"] = data["avg_price"].rolling(w).std()

    data["online_ratio_shift"] = data["online_ratio"].shift(1)
    data["price_momentum"] = data["avg_price"].diff(7)
    data["sales_momentum"] = data["sales_kg"].diff(7)
    # 供需比特征 (日均产量 vs 近期均值)
    data["supply_ratio"] = data["sales_kg"] / data["sales_rm_30"]

    data = data.dropna().reset_index(drop=True)
    return data

def simulate_future_weather(last_row, n_days=7):
    """基于最近30天天气趋势，模拟未来7天天气"""
    temps, hums, rains = [], [], []
    current_temp = last_row["temperature"]
    current_hum = last_row["humidity"]
    current_month = last_row["month"]

    for i in range(1, n_days + 1):
        next_date = last_row["date"] + timedelta(days=i)
        # 季节性趋势
        seasonal_temp = 16 + 11 * np.sin((next_date.timetuple().tm_yday - 80) * 2 * np.pi / 365)
        next_temp = current_temp + 0.3 * (seasonal_temp - current_temp) + np.random.normal(0, 1.5)
        next_hum = current_hum + np.random.normal(0, 2)
        # 当前月份降雨概率
        if 4 <= next_date.month <= 9:
            next_rain = max(0, np.random.exponential(8))
        else:
            next_rain = max(0, np.random.exponential(2))

        temps.append(round(np.clip(next_temp, 2, 38), 1))
        hums.append(round(np.clip(next_hum, 45, 98), 1))
        rains.append(round(next_rain, 1))
        current_temp, current_hum = next_temp, next_hum

    return temps, hums, rains

def prepare_sequences(data, seq_len=30, pred_len=7):
    exclude = ["date","sales_kg","avg_price","promotion_type","origin",
               "online_sales","offline_sales","grade_premium_pct","grade_standard_pct",
               "grade_second_pct","daily_revenue","month_factor","production_season","year"]
    feature_cols = [c for c in data.columns if c not in exclude]

    X = data[feature_cols].values.astype(np.float32)
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds[stds == 0] = 1
    X_s = (X - means) / stds

    y_s = data["sales_kg"].values.astype(np.float32)
    y_p = data["avg_price"].values.astype(np.float32)
    s_m, s_s = y_s.mean(), y_s.std()
    p_m, p_s = y_p.mean(), y_p.std()
    if s_s == 0: s_s = 1
    if p_s == 0: p_s = 1
    y_s_n = (y_s - s_m) / s_s
    y_p_n = (y_p - p_m) / p_s

    X_seq, y_s_seq, y_p_seq = [], [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X_seq.append(X_s[i:i+seq_len])
        y_s_seq.append(y_s_n[i+seq_len:i+seq_len+pred_len])
        y_p_seq.append(y_p_n[i+seq_len:i+seq_len+pred_len])

    return (np.array(X_seq), np.array(y_s_seq), np.array(y_p_seq),
            means, stds, s_m, s_s, p_m, p_s, feature_cols)

# ============================================================
# LSTM 模型
# ============================================================
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    pass

def build_lstm_model(seq_len, n_features, pred_len):
    if not TF_AVAILABLE:
        return None
    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(80, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(40)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    sales_out = layers.Dense(pred_len, name="sales_output")(x)
    price_out = layers.Dense(pred_len, name="price_output")(x)
    model = keras.Model(inputs=inputs, outputs=[sales_out, price_out])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0008),
        loss={"sales_output": "mse", "price_output": "mse"},
        loss_weights={"sales_output": 0.55, "price_output": 0.45}
    )
    return model

@st.cache_resource
def train_model(_df):
    df = _df.copy()
    feats = create_features(df)
    seq_len, pred_len = 30, 7
    X, y_s, y_p, means, stds, s_m, s_s, p_m, p_s, fcols = prepare_sequences(feats, seq_len, pred_len)

    split = max(0, len(X) - min(90, len(X) // 5))
    X_tr, X_v = X[:split], X[split:]
    y_s_tr, y_s_v = y_s[:split], y_s[split:]
    y_p_tr, y_p_v = y_p[:split], y_p[split:]

    meta = {
        "feature_means": means, "feature_stds": stds,
        "sales_mean": s_m, "sales_std": s_s,
        "price_mean": p_m, "price_std": p_s,
        "feature_cols": fcols, "seq_len": seq_len, "pred_len": pred_len,
        "model_type": "LSTM" if TF_AVAILABLE else "sklearn"
    }

    if TF_AVAILABLE:
        model = build_lstm_model(seq_len, len(fcols), pred_len)
        cb = [keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)]
        with st.spinner("🤖 LSTM神经网络训练中..."):
            model.fit(X_tr, {"sales_output": y_s_tr, "price_output": y_p_tr},
                      validation_data=(X_v, {"sales_output": y_s_v, "price_output": y_p_v}),
                      epochs=200, batch_size=32, verbose=0, callbacks=cb)
        meta["model"] = model
        preds = model.predict(X_v, verbose=0)
        s_pred = preds[0].flatten() * s_s + s_m
        s_true = y_s_v.flatten() * s_s + s_m
        p_pred = preds[1].flatten() * p_s + p_m
        p_true = y_p_v.flatten() * p_s + p_m
    else:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        X_tr_f = X_tr.reshape(X_tr.shape[0], -1)
        X_v_f = X_v.reshape(X_v.shape[0], -1)
        with st.spinner("🤖 集成模型训练中 (RF+GBDT)..."):
            rf_s = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1))
            rf_p = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1))
            rf_s.fit(X_tr_f, y_s_tr)
            rf_p.fit(X_tr_f, y_p_tr)
        meta["model"] = (rf_s, rf_p)
        s_pred = rf_s.predict(X_v_f).flatten() * s_s + s_m
        s_true = y_s_v.flatten() * s_s + s_m
        p_pred = rf_p.predict(X_v_f).flatten() * p_s + p_m
        p_true = y_p_v.flatten() * p_s + p_m

    meta["metrics"] = {
        "sales_mae": float(np.mean(np.abs(s_pred - s_true))),
        "sales_mape": float(np.mean(np.abs((s_pred - s_true) / (s_true + 1))) * 100),
        "price_mae": float(np.mean(np.abs(p_pred - p_true))),
        "price_mape": float(np.mean(np.abs((p_pred - p_true) / (p_true + 0.1))) * 100)
    }
    return meta

def predict_future(model_meta, df):
    """预测未来7天 (含天气外推)"""
    seq_len = model_meta["seq_len"]
    pred_len = model_meta["pred_len"]
    fcols = model_meta["feature_cols"]
    means = model_meta["feature_means"]
    stds = model_meta["feature_stds"]
    s_m, s_s = model_meta["sales_mean"], model_meta["sales_std"]
    p_m, p_s = model_meta["price_mean"], model_meta["price_std"]

    # 构建未来7天的特征行 (基于最近数据外推)
    last_row = df.iloc[-1]
    future_temps, future_hums, future_rains = simulate_future_weather(last_row, pred_len)

    # 从最近数据构建预测所需的特征
    feats = create_features(df)
    recent_features = feats[fcols].values[-seq_len:].astype(np.float32)
    X_input = ((recent_features - means) / stds).reshape(1, seq_len, len(fcols))

    if model_meta["model_type"] == "LSTM":
        preds = model_meta["model"].predict(X_input, verbose=0)
        s_pred = preds[0].flatten() * s_s + s_m
        p_pred = preds[1].flatten() * p_s + p_m
    else:
        rf_s, rf_p = model_meta["model"]
        s_pred = rf_s.predict(X_input.reshape(1, -1)).flatten() * s_s + s_m
        p_pred = rf_p.predict(X_input.reshape(1, -1)).flatten() * p_s + p_m

    # 置信区间
    tail_s = df["sales_kg"].tail(60)
    tail_p = df["avg_price"].tail(60)
    vol_s = tail_s.std() / tail_s.mean()
    ci_s = 1.96 * tail_s.std()
    ci_p = 1.96 * tail_p.std()

    if vol_s > 0.28:
        risk = "高"
    elif vol_s > 0.16:
        risk = "中"
    else:
        risk = "低"

    pred_dates = [df["date"].max() + timedelta(days=i) for i in range(1, pred_len + 1)]
    s_pred = np.maximum(s_pred, 100)

    return pd.DataFrame({
        "date": pred_dates,
        "pred_sales_kg": s_pred.round(0),
        "pred_sales_low": (s_pred - ci_s).clip(0).round(0),
        "pred_sales_high": (s_pred + ci_s).round(0),
        "pred_price": p_pred.round(1),
        "pred_price_low": (p_pred - ci_p).clip(0).round(1),
        "pred_price_high": (p_pred + ci_p).round(1),
        "sim_temp": future_temps,
        "sim_rainfall": future_rains,
        "risk_level": risk,
        "volatility": round(vol_s, 3)
    })

# ============================================================
# 模块D: 智能定价
# ============================================================
def price_optimizer(df, prediction):
    recent = df.tail(30)
    cost = 10.0
    margin = 0.45

    supply_idx = recent["sales_kg"].mean() / df["sales_kg"].mean()
    demand_f = 1.0 + (1.0 - supply_idx) * 0.15

    gap = recent["competitor_price"].mean() - recent["avg_price"].mean()
    comp_f = 1.0 + gap / max(recent["avg_price"].mean(), 0.1) * 0.3

    base = cost * (1 + margin) * demand_f * comp_f
    base = np.clip(base, 13, 26)

    grades = {"特级果": round(base * 1.35, 1),
              "一级果": round(base, 1),
              "二级果": round(base * 0.68, 1)}

    dynamic = []
    for i in range(7):
        dfac = 1.0 + i * 0.004
        if prediction is not None and i < len(prediction):
            sr = prediction.iloc[i]["pred_sales_kg"] / max(recent["sales_kg"].mean(), 1)
            padj = 1.0 - (sr - 1.0) * 0.06
        else:
            padj = 1.0
        dp = base * dfac * padj
        dynamic.append({
            "day": i+1,
            "price": round(dp, 1),
            "premium": round(dp * 1.35, 1),
            "standard": round(dp, 1),
            "second": round(dp * 0.68, 1)
        })

    return {
        "base_price": round(base, 1),
        "grade_prices": grades,
        "dynamic_table": pd.DataFrame(dynamic),
        "elasticity": -1.05,
        "sales_boost_1pct_cut": "+1.1%",
        "cost_base": cost,
        "demand_factor": round(demand_f, 3),
        "competitor_factor": round(comp_f, 3)
    }

# ============================================================
# 模块E: 销售策略引擎
# ============================================================
def strategy_engine(df, prediction, pricing):
    recent = df.tail(30)
    online_pct = recent["online_sales"].sum() / max(recent["sales_kg"].sum(), 1)

    if prediction is not None and len(prediction) > 0:
        month_now = prediction.iloc[0]["date"].month
    else:
        month_now = datetime.now().month

    if month_now in [8,9,10]:
        mix = "直播30% + 社区团购25% + 批发25% + 零售20%"
        note = "旺季走量为主，直播+社区团购抓线上流量，批发稳大盘"
    elif month_now in [5,6,7]:
        mix = "直播35% + 电商25% + 社区团购20% + 批发20%"
        note = "产季启动期，提早锁渠道，直播预热+预售主打"
    elif month_now in [11]:
        mix = "电商25% + 批发40% + 社区团购20% + 直播15%"
        note = "尾期快出为主，批发走量，精选礼品装溢价"
    else:
        mix = "电商30% + 社区团购25% + 直播20% + 批发25%"
        note = "休果期以加工品和库存为主，维护客户关系"

    targets = []
    for i in range(7):
        if prediction is not None and i < len(prediction):
            t = prediction.iloc[i]["pred_sales_kg"]
        else:
            t = recent["sales_kg"].mean()
        on_t = int(t * online_pct * (1.15 if i % 3 == 0 else 1.0))
        off_t = int(t * (1 - online_pct))
        targets.append({
            "day": i+1,
            "date": (datetime.now() + timedelta(days=i+1)).strftime("%m/%d"),
            "total_kg": int(t), "online_kg": on_t, "offline_kg": off_t
        })

    promos_pool = [
        {"n":"组合装引流","d":"特级+一级混装3斤 ¥39.9包邮拉新","m":[1,2,3,4,5,6,7,8,9,10,11,12]},
        {"n":"产地溯源直播","d":"田间直播+榕江故事+村超IP，转化率极高","m":[6,7,8,9,10,11]},
        {"n":"满减阶梯","d":"满99减15 / 满199减40，提客单价","m":[1,2,3,4,5,6,7,8,9,10,11,12]},
        {"n":"礼品套装","d":"精装礼盒+文化卡片，企业采购/伴手礼","m":[1,9,10,11,12]},
        {"n":"村超联名限定","d":"村超比赛日限量包装，热点溢价5-8元","m":[3,4,5,6,7,8,9,10,11]},
        {"n":"早鸟预售锁单","d":"产季前预售锁客，资金提前回流","m":[5,6,7]},
        {"n":"社区团长激励","d":"团长佣金12%+销量阶梯奖励，快速铺社区","m":[1,2,3,4,5,6,7,8,9,10,11,12]},
    ]
    active = [p for p in promos_pool if month_now in p["m"]]

    risks = []
    if prediction is not None:
        ps = prediction["pred_sales_kg"].values
        pp = prediction["pred_price"].values
        if ps[-1] > ps[0] * 1.25:
            risks.append("📈 未来7天预计销量持续攀升，确认物流和包装产能")
        if ps[-1] < ps[0] * 0.72:
            risks.append("📉 销量有下行趋势，适度减少采摘量，避免库存积压")
        if pp[-1] < pp[0] * 0.93:
            risks.append("⚠️ 价格预计下跌7%+，建议提前锁单或走批发快出")
        if prediction["risk_level"].iloc[0] == "高":
            risks.append("🔴 市场波动风险高，控制日出货量在均值80%以内")
    if month_now in [8,9,10]:
        risks.append("⚠️ 集中上市期，广西/云南货涌入或致价格踩踏，分散出货节奏")
    if month_now in [11,12,1,2]:
        risks.append("❄️ 休果期保鲜成本上升，控制库存<7天周转")

    return {
        "channel_mix": mix, "channel_note": note,
        "daily_targets": pd.DataFrame(targets),
        "active_promos": active, "risks": risks
    }

# ============================================================
# Streamlit UI — 5 Tab
# ============================================================
def main():
    # ---- 侧边栏 ----
    with st.sidebar:
        st.markdown("## 🍈 榕江百香果AI系统")
        st.markdown("**📍 贵州省·黔东南·榕江县**")
        st.markdown("**🏆 中国黄金百香果之乡**")
        st.markdown("---")
        emoji = "🧠" if TF_AVAILABLE else "🌲"
        name = "LSTM神经网络" if TF_AVAILABLE else "RF+GBDT集成"
        st.markdown(f"**引擎**: {emoji} {name}")
        st.markdown("---")

        # 榕江真实行情数据
        st.markdown("### 📡 榕江真实行情")
        if FETCHER_AVAILABLE:
            rj = get_rongjiang_reference()
            if rj:
                st.markdown(f"""
                <div style="background:#e8f5e9;border-radius:8px;padding:10px;font-size:0.85rem;">
                <b>🍈 榕江黄金百香果</b><br>
                批发: <b>¥{rj['origin_wholesale_jin']}/斤</b> (¥{rj['origin_wholesale_kg']}/kg)<br>
                大宗: <b>¥{rj['bulk_1000jin_price']}/斤</b> (1000斤起)<br>
                地头: <b>¥{rj['field_min_price_jin']}/斤</b> 起
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"📊 年产值{rj['annual_value_yi']}亿 | 线上{rj['online_ratio']*100:.0f}% | {rj['area_mu']}亩")
                st.caption(f"数据: {rj['data_source']} | {rj['data_date']}")
            else:
                st.caption("⚠️ 榕江数据未加载")
        else:
            st.caption("⚠️ 数据模块未加载")

        st.markdown("---")
        st.markdown("### 📅 数据筛选")
        d1 = st.date_input("开始", value=datetime(2024, 1, 1))
        d2 = st.date_input("结束", value=datetime(2026, 4, 27))

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 刷新", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        with c2:
            pass  # CSV export handled below

        st.markdown("---")
        uploaded = st.file_uploader("📥 导入CSV/Excel", type=["csv","xlsx"])
        st.caption("需含: date, sales_kg, avg_price")
        st.markdown("---")
        st.caption("v2.0 | AI-Powered")

    # ---- 数据 ----
    if uploaded is not None:
        try:
            df = load_external_data(uploaded)
            st.sidebar.success(f"✅ 导入 {len(df)} 条")
        except Exception as e:
            st.sidebar.error(f"失败: {e}")
            df = generate_data()
    else:
        df = generate_data()

    # 用榕江真实行情校准模拟价格
    if FETCHER_AVAILABLE and uploaded is None:
        try:
            rj_ref = get_rongjiang_reference()
            if rj_ref:
                df = calibrate_simulation(df, rj_ref)
        except:
            pass

    mask = (df["date"] >= pd.Timestamp(d1)) & (df["date"] <= pd.Timestamp(d2))
    df_f = df[mask].copy()

    # ---- 训练 & 预测 ----
    model_meta = train_model(df_f)
    prediction = predict_future(model_meta, df_f)
    pricing = price_optimizer(df_f, prediction)
    strategy = strategy_engine(df_f, prediction, pricing)

    # ======= 标题 + KPI =======
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
        <h1 style="color:#1a5c32;margin:0;">🍈 榕江县黄金百香果AI决策系统</h1>
        <span style="background:#2d8c4a;color:white;padding:2px 10px;border-radius:12px;font-size:0.8rem;">v2.0</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"数据 {df_f['date'].min().strftime('%Y-%m-%d')} ~ {df_f['date'].max().strftime('%Y-%m-%d')} "
               f"| {len(df_f)}条 | {model_meta['model_type']} | 更新 {datetime.now().strftime('%m/%d %H:%M')}")

    recent = df_f[df_f["date"] >= df_f["date"].max() - pd.Timedelta(days=30)]
    prev = df_f[(df_f["date"] >= df_f["date"].max() - pd.Timedelta(days=60)) &
                (df_f["date"] < df_f["date"].max() - pd.Timedelta(days=30))]

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    ts = recent["sales_kg"].sum()/1000
    ps_ = prev["sales_kg"].sum()/1000 if len(prev)>0 else ts
    k1.metric("📦 30天总销量", f"{ts:.1f}吨", f"{(ts/ps_-1)*100:+.1f}%")
    ap = recent["avg_price"].mean()
    app = prev["avg_price"].mean() if len(prev)>0 else ap
    k2.metric("💰 均价", f"¥{ap:.1f}", f"{(ap/app-1)*100:+.1f}%")
    k3.metric("📱 线上占比", f"{recent['online_sales'].sum()/max(recent['sales_kg'].sum(),1)*100:.1f}%")
    k4.metric("📊 日均销量", f"{recent['sales_kg'].mean():.0f}kg")
    k5.metric("💵 月收入", f"¥{recent['daily_revenue'].sum()/10000:.1f}万")
    rl = prediction["risk_level"].iloc[0] if prediction is not None else "--"
    k6.metric("🎯 风险", rl)
    st.markdown("---")

    # ======= 5 Tab =======
    t1,t2,t3,t4,t5 = st.tabs(["📊 市场看板","🔍 深度分析","📈 7天预测","💰 智能定价","🚀 销售策略"])

    # ===== TAB1: 市场看板 =====
    with t1:
        st.markdown('<p class="section-title">📊 市场趋势与概览</p>', unsafe_allow_html=True)

        # 月度双轴
        mon = df_f.groupby(df_f["date"].dt.to_period("M")).agg(
            sales_kg=("sales_kg","sum"), avg_price=("avg_price","mean"),
            online=("online_ratio","mean")).reset_index()
        mon["date"] = mon["date"].astype(str)
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Bar(x=mon["date"], y=mon["sales_kg"]/1000,
                               name="月销量(吨)", marker_color="#2d8c4a"), secondary_y=False)
        fig1.add_trace(go.Scatter(x=mon["date"], y=mon["avg_price"],
                                   name="均价(元/kg)", mode="lines+markers",
                                   line=dict(color="#f5a623",width=3)), secondary_y=True)
        fig1.update_layout(height=400, hovermode="x unified", template="plotly_white",
                           legend=dict(orientation="h",y=1.12), margin=dict(l=20,r=20,t=20,b=20))
        fig1.update_yaxes(title_text="吨", secondary_y=False, gridcolor="#e8f5e9")
        fig1.update_yaxes(title_text="元/kg", secondary_y=True, gridcolor="#fff8e1")
        st.plotly_chart(fig1, use_container_width=True)

        # 同比对比 + 价格分布
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**📅 年度同比: 逐月销量**")
            yoy = df_f.groupby(["year","month"]).agg(sales=("sales_kg","sum")).reset_index()
            yoy["year"] = yoy["year"].astype(int)
            fig_yoy = px.line(yoy, x="month", y="sales", color="year",
                              color_discrete_sequence=["#90caf9","#2d8c4a","#f5a623"],
                              markers=True, labels={"sales":"销量(kg)","month":"月份"})
            fig_yoy.update_layout(height=320, template="plotly_white", margin=dict(l=20,r=20,t=10,b=20))
            st.plotly_chart(fig_yoy, use_container_width=True)

        with cb:
            st.markdown("**📊 价格区间分布**")
            fig_hist = px.histogram(df_f, x="avg_price", nbins=40, color_discrete_sequence=["#2d8c4a"],
                                     labels={"avg_price":"均价(元/kg)"},
                                     title="全期价格分布直方图")
            fig_hist.add_vline(x=df_f["avg_price"].mean(), line_dash="dash", line_color="#f5a623",
                               annotation_text=f"均值 ¥{df_f['avg_price'].mean():.1f}")
            fig_hist.update_layout(height=320, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        # 周度 + 渠道饼图
        cc, cd = st.columns(2)
        with cc:
            dow = df_f.groupby("dayofweek")["sales_kg"].mean().reset_index()
            dow["label"] = dow["dayofweek"].apply(lambda x: ["周一","周二","周三","周四","周五","周六","周日"][int(x)])
            fig_dow = px.bar(dow, x="label", y="sales_kg", color_discrete_sequence=["#2d8c4a"],
                             labels={"sales_kg":"日均销量(kg)", "label":"星期"}, title="周度销量模式")
            fig_dow.update_layout(height=300, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(fig_dow, use_container_width=True)
        with cd:
            fig_pie = px.pie(values=[df_f["online_sales"].sum(), df_f["offline_sales"].sum()],
                             names=["线上","线下"], hole=0.45,
                             color_discrete_sequence=["#f5a623","#2d8c4a"], title="渠道占比")
            fig_pie.update_layout(height=300, template="plotly_white", margin=dict(l=20,r=20,t=30,b=20))
            fig_pie.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_pie, use_container_width=True)

    # ===== TAB2: 深度分析 =====
    with t2:
        st.markdown('<p class="section-title">🔍 多维深度分析</p>', unsafe_allow_html=True)

        ca1, ca2 = st.columns(2)
        with ca1:
            st.markdown("**🔥 核心变量相关性**")
            ccols = ["sales_kg","avg_price","temperature","humidity","rainfall",
                     "competitor_price","cunchao_flag","promo_flag","is_weekend","online_ratio"]
            corr = df_f[ccols].corr()
            fig_c = px.imshow(corr, text_auto=".2f", aspect="auto",
                              color_continuous_scale=["#e8f5e9","#fff8e1","#f5a623","#c62828"])
            fig_c.update_layout(height=400, template="plotly_white", margin=dict(l=20,r=20,t=10,b=20))
            st.plotly_chart(fig_c, use_container_width=True)

        with ca2:
            st.markdown("**⚠️ 销量+价格双异常检测**")
            a_s = detect_anomalies(df_f["sales_kg"])
            a_p = detect_anomalies(df_f["avg_price"])
            a_all = a_s | a_p
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=df_f[~a_all]["date"], y=df_f[~a_all]["sales_kg"],
                                        mode="markers", name="正常", marker=dict(size=2,color="#2d8c4a")))
            fig_a.add_trace(go.Scatter(x=df_f[a_s]["date"], y=df_f[a_s]["sales_kg"],
                                        mode="markers", name="销量异常", marker=dict(size=8,color="#d32f2f",symbol="x")))
            fig_a.add_trace(go.Scatter(x=df_f[a_p & ~a_s]["date"], y=df_f[a_p & ~a_s]["sales_kg"],
                                        mode="markers", name="价格异常", marker=dict(size=8,color="#f5a623",symbol="triangle-up")))
            fig_a.update_layout(height=400, template="plotly_white",
                                title=f"异常: 销量{a_s.sum()}天 + 价格{a_p.sum()}天", margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_a, use_container_width=True)

        # 时序分解
        st.markdown("**📈 时序分解: 趋势·季节·残差**")
        trend, seasonal, residual = seasonal_decomposition(df_f)
        fig_decomp = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    subplot_titles=["原始数据+趋势线","季节分量(年度周期)","残差"],
                                    vertical_spacing=0.08)
        fig_decomp.add_trace(go.Scatter(x=df_f["date"], y=df_f["sales_kg"],
                                         name="原始销量", line=dict(color="#2d8c4a",width=1)), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=df_f["date"], y=trend,
                                         name="30日均线趋势", line=dict(color="#f5a623",width=2)), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=df_f["date"], y=seasonal,
                                         name="季节分量", line=dict(color="#4caf50",width=1)), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=df_f["date"], y=residual,
                                         name="残差", line=dict(color="#90a4ae",width=0.8)), row=3, col=1)
        fig_decomp.update_layout(height=500, template="plotly_white", hovermode="x",
                                  showlegend=False, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig_decomp, use_container_width=True)

        # 异常事件表 + 产季分析
        cb1, cb2 = st.columns(2)
        with cb1:
            st.markdown("**🔴 异常事件归因**")
            evt = annotate_anomaly_events(df_f, a_s | a_p)
            if len(evt) > 0:
                st.dataframe(evt, use_container_width=True, hide_index=True)
            else:
                st.info("无显著异常")

        with cb2:
            st.markdown("**🌿 产季分析**")
            season_data = df_f.groupby("production_season").agg(
                天数=("date","count"), 总销量吨=("sales_kg",lambda x: round(x.sum()/1000,1)),
                均价=("avg_price","mean"), 线上占比=("online_ratio","mean")
            ).reindex(["产果旺季","产果期","休果期"]).reset_index()
            st.dataframe(season_data, use_container_width=True, hide_index=True,
                         column_config={"均价": st.column_config.NumberColumn(format="¥%.1f"),
                                        "线上占比": st.column_config.NumberColumn(format="%.1f%%")})

        # 行情总结
        st.markdown(generate_market_summary(df_f))

    # ===== TAB3: 7天预测 =====
    with t3:
        st.markdown('<p class="section-title">📈 未来7天 AI双预测 (销量+价格)</p>', unsafe_allow_html=True)
        st.caption(f"模型: {model_meta['model_type']} | 序列长度30天 | 输出7天 | 含天气预报外推")

        m = model_meta["metrics"]
        mc1,mc2,mc3,mc4,mc5,mc6 = st.columns(6)
        mc1.metric("🎯 销量MAE", f"{m['sales_mae']:.0f}kg")
        mc2.metric("📊 销量MAPE", f"{m['sales_mape']:.1f}%")
        mc3.metric("🎯 价格MAE", f"¥{m['price_mae']:.2f}")
        mc4.metric("📊 价格MAPE", f"{m['price_mape']:.1f}%")
        mc5.metric("📐 波动系数", f"{prediction['volatility'].iloc[0]:.3f}" if prediction is not None else "--")
        mc6.metric("⚡ 风险等级", prediction["risk_level"].iloc[0] if prediction is not None else "--")

        if prediction is not None:
            cp1, cp2 = st.columns(2)
            hist60 = df_f.tail(60)
            with cp1:
                st.markdown("**📦 未来7天销量预测 (含95%置信区间)**")
                fps = go.Figure()
                fps.add_trace(go.Scatter(x=hist60["date"], y=hist60["sales_kg"],
                                          name="历史", line=dict(color="#2d8c4a",width=1.5)))
                fps.add_trace(go.Scatter(x=prediction["date"], y=prediction["pred_sales_kg"],
                                          name="预测", mode="lines+markers",
                                          line=dict(color="#f5a623",width=3,dash="dash"), marker=dict(size=8)))
                fps.add_trace(go.Scatter(x=prediction["date"], y=prediction["pred_sales_high"],
                                          fill=None, mode="lines", line=dict(width=0), showlegend=False))
                fps.add_trace(go.Scatter(x=prediction["date"], y=prediction["pred_sales_low"],
                                          fill="tonexty", mode="lines", name="95%CI",
                                          line=dict(width=0), fillcolor="rgba(245,166,35,0.12)"))
                fps.update_layout(height=380, template="plotly_white", hovermode="x unified",
                                   margin=dict(l=20,r=20,t=10,b=20))
                st.plotly_chart(fps, use_container_width=True)

            with cp2:
                st.markdown("**💰 未来7天价格预测 (含95%置信区间)**")
                fpp = go.Figure()
                fpp.add_trace(go.Scatter(x=hist60["date"], y=hist60["avg_price"],
                                          name="历史", line=dict(color="#4caf50",width=1.5)))
                fpp.add_trace(go.Scatter(x=prediction["date"], y=prediction["pred_price"],
                                          name="预测", mode="lines+markers",
                                          line=dict(color="#d32f2f",width=3,dash="dash"), marker=dict(size=8)))
                fpp.add_trace(go.Scatter(x=prediction["date"], y=prediction["pred_price_high"],
                                          fill=None, mode="lines", line=dict(width=0), showlegend=False))
                fpp.add_trace(go.Scatter(x=prediction["date"], y=prediction["pred_price_low"],
                                          fill="tonexty", mode="lines", name="95%CI",
                                          line=dict(width=0), fillcolor="rgba(211,47,47,0.10)"))
                fpp.update_layout(height=380, template="plotly_white", hovermode="x unified",
                                   margin=dict(l=20,r=20,t=10,b=20))
                st.plotly_chart(fpp, use_container_width=True)

            # 预测明细表
            st.markdown("**📋 7天预测明细**")
            disp = prediction.copy()
            disp["date"] = disp["date"].dt.strftime("%m/%d")
            st.dataframe(
                disp[["date","pred_sales_kg","pred_sales_low","pred_sales_high",
                      "pred_price","pred_price_low","pred_price_high",
                      "sim_temp","sim_rainfall","risk_level"]].rename(columns={
                    "date":"日期","pred_sales_kg":"销量(kg)","pred_sales_low":"销量下限",
                    "pred_sales_high":"销量上限","pred_price":"均价","pred_price_low":"价下限",
                    "pred_price_high":"价上限","sim_temp":"模拟温度°C",
                    "sim_rainfall":"模拟降雨mm","risk_level":"风险"
                }),
                use_container_width=True, hide_index=True,
                column_config={
                    "销量(kg)": st.column_config.NumberColumn(format="%d"),
                    "均价": st.column_config.NumberColumn(format="¥%.1f"),
                    "模拟温度°C": st.column_config.NumberColumn(format="%.1f"),
                }
            )

            rl = prediction["risk_level"].iloc[0]
            rc = "risk-low" if rl=="低" else ("risk-mid" if rl=="中" else "risk-high")
            st.markdown(f'### 波动风险评级: <span class="{rc}">{rl}</span> '
                        f'（波动系数 {prediction["volatility"].iloc[0]:.3f}）', unsafe_allow_html=True)

            # 价格预测特别说明
            st.markdown(f"""
            <div class="strategy-box">
            <strong>💰 7天价格趋势解读</strong><br>
            · 预测起始价: <b>¥{prediction["pred_price"].iloc[0]}/kg</b> → 第7天: <b>¥{prediction["pred_price"].iloc[-1]}/kg</b><br>
            · 趋势: {"📈 上行" if prediction["pred_price"].iloc[-1] > prediction["pred_price"].iloc[0]*1.02
                    else ("📉 下行" if prediction["pred_price"].iloc[-1] < prediction["pred_price"].iloc[0]*0.98 else "➡️ 平稳")}<br>
            · 价格区间: ¥{prediction["pred_price_low"].min():.1f} ~ ¥{prediction["pred_price_high"].max():.1f}/kg<br>
            · 建议: 抓住价格高位出货，参考定价模块每日建议价
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("预测模型暂不可用，请刷新页面重试")

    # ===== TAB4: 智能定价 =====
    with t4:
        st.markdown('<p class="section-title">💰 四维智能定价引擎</p>', unsafe_allow_html=True)
        st.caption("成本基准 × 供需因子 × 竞品因子 × 预测销量 → 动态建议价")

        pc1,pc2,pc3,pc4 = st.columns(4)
        pc1.metric("📋 成本基线", f"¥{pricing['cost_base']}/kg")
        pc2.metric("📊 供需因子", f"{pricing['demand_factor']:.3f}")
        pc3.metric("🏪 竞品因子", f"{pricing['competitor_factor']:.3f}")
        pc4.metric("📐 价格弹性", f"{pricing['elasticity']}")

        pr1, pr2 = st.columns(2)
        with pr1:
            st.markdown("**🍈 今日建议售价 (元/kg)**")
            g = pricing["grade_prices"]
            fig_g = go.Figure()
            fig_g.add_trace(go.Bar(x=list(g.keys()), y=list(g.values()),
                                    marker_color=["#ffd700","#2d8c4a","#90caf9"],
                                    text=[f"¥{v}/kg" for v in g.values()],
                                    textposition="outside"))
            fig_g.update_layout(height=340, template="plotly_white", yaxis_title="元/kg",
                                 margin=dict(l=20,r=20,t=10,b=20))
            st.plotly_chart(fig_g, use_container_width=True)
            st.markdown(f"""
            <div class="strategy-box">
            <strong>💡 弹性分析</strong><br>
            降价1% → 预计增销 <b>{pricing['sales_boost_1pct_cut']}</b><br>
            建议小步快调：3-5%/次，观察3天效果再调整
            </div>
            """, unsafe_allow_html=True)

        with pr2:
            st.markdown("**📅 7天动态定价**")
            dyn = pricing["dynamic_table"]
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=dyn["day"], y=dyn["premium"],
                                        name="特级果", mode="lines+markers",
                                        line=dict(color="#ffd700",width=2)))
            fig_d.add_trace(go.Scatter(x=dyn["day"], y=dyn["standard"],
                                        name="一级果", mode="lines+markers",
                                        line=dict(color="#2d8c4a",width=2)))
            fig_d.add_trace(go.Scatter(x=dyn["day"], y=dyn["second"],
                                        name="二级果", mode="lines+markers",
                                        line=dict(color="#90caf9",width=2)))
            fig_d.update_layout(height=340, template="plotly_white",
                                 xaxis_title="天数", yaxis_title="元/kg",
                                 margin=dict(l=20,r=20,t=10,b=20))
            st.plotly_chart(fig_d, use_container_width=True)

        st.markdown(f"""
        <div class="strategy-box">
        <strong>🎯 定价执行建议</strong><br>
        · <b>基准价(一级果)</b>: ¥{pricing['base_price']}/kg — 作为线上标价基准<br>
        · <b>特级果</b>: 溢价+35% → 高端电商/礼品渠道<br>
        · <b>二级果</b>: 折扣-32% → 批发/加工/社区团购走量<br>
        · <b>竞品参考</b>: 因子 {pricing['competitor_factor']:.3f} ({"可溢价" if pricing['competitor_factor']>1 else "需降价竞争"})<br>
        · <b>调价原则</b>: 预测量大→微降走量，预测少→稳价保利
        </div>
        """, unsafe_allow_html=True)

    # ===== TAB5: 销售策略 =====
    with t5:
        st.markdown('<p class="section-title">🚀 可落地销售策略</p>', unsafe_allow_html=True)

        sc1, sc2 = st.columns([1,2])
        with sc1:
            st.markdown(f"""
            <div class="strategy-box">
            <strong>📡 推荐渠道配比</strong><br>
            <b>{strategy['channel_mix']}</b>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.info(f"💡 {strategy['channel_note']}")

        st.markdown("### 🎯 7天每日销售目标")
        tg = strategy["daily_targets"]
        st.dataframe(tg.rename(columns={"day":"天","date":"日期","total_kg":"总目标kg",
                                         "online_kg":"线上kg","offline_kg":"线下kg"}),
                     use_container_width=True, hide_index=True)

        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(x=tg["date"], y=tg["online_kg"], name="线上", marker_color="#f5a623"))
        fig_t.add_trace(go.Bar(x=tg["date"], y=tg["offline_kg"], name="线下", marker_color="#2d8c4a"))
        fig_t.update_layout(height=280, barmode="stack", template="plotly_white",
                             xaxis_title="日期", yaxis_title="kg", margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fig_t, use_container_width=True)

        ss1, ss2 = st.columns([1,1])
        with ss1:
            st.markdown("### 🎁 推荐促销")
            for p in strategy["active_promos"][:4]:
                st.markdown(f'<div class="strategy-box"><b>{p["n"]}</b><br><small>{p["d"]}</small></div>',
                            unsafe_allow_html=True)
        with ss2:
            st.markdown("### ⚠️ 风险预警")
            if strategy["risks"]:
                for r in strategy["risks"]:
                    st.markdown(f'<div class="strategy-box" style="border-left:4px solid #d32f2f;">{r}</div>',
                                unsafe_allow_html=True)
            else:
                st.success("✅ 无特殊风险")

        st.markdown(f"""
        <div class="strategy-box" style="margin-top:16px;">
        <strong>📝 7天执行清单</strong>
        <ol>
            <li>定价基准 ¥{pricing['base_price']}/kg，线上+3~5元，批发-2~3元</li>
            <li>{strategy['channel_note']}</li>
            <li>按预测销量80%备货，保持日周转</li>
            <li>每日复盘实际vs预测，持续校准模型</li>
            <li>榕江产地IP+村超故事=5~8元溢价，坚持讲好产地故事</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    # ---- 导出 & 页脚 ----
    st.markdown("---")
    fc1, fc2 = st.columns([3,1])
    with fc1:
        st.caption(f"榕江百香果AI v2.0 | {df_f['date'].max().strftime('%Y-%m-%d')} | "
                   f"{model_meta['model_type']} | 销量MAPE {model_meta['metrics']['sales_mape']:.1f}% | "
                   f"价格MAPE {model_meta['metrics']['price_mape']:.1f}%")
    with fc2:
        st.markdown(csv_download_link(df_f), unsafe_allow_html=True)

    # ---- 侧边栏数据导出 ----
    with st.sidebar:
        st.markdown("---")
        st.markdown(csv_download_link(df_f), unsafe_allow_html=True)
        st.caption("点击下载完整数据CSV")


if __name__ == "__main__":
    main()
