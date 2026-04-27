# 榕江县黄金百香果市场AI分析与销售决策系统

贵州省黔东南州榕江县 — 黄金百香果专业化AI决策工具

## 功能概览

| 模块 | 功能 |
|------|------|
| 市场看板 | KPI概览、销量/价格双轴趋势、周度模式、渠道占比 |
| 深度分析 | 相关性热力图、异常检测、温度-销量关系、产地流向、行情总结 |
| 7天预测 | LSTM/RandomForest双输出（销量+价格）、置信区间、风险评级 |
| 智能定价 | 成本+供需+竞品+目标四维定价、三级果分级、7天动态定价表 |
| 销售策略 | 渠道组合、每日销售目标、促销方案、风险预警 |

## 快速启动

```bash
# 1. 进入项目目录
cd rongjiang_passionfruit

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动系统
streamlit run app.py

# 4. 浏览器打开
http://localhost:8501
```

## 数据说明

系统启动时自动生成 2024.01 ~ 2026.04 的榕江百香果日度模拟数据，包含以下字段：

- date, sales_kg, avg_price, online_sales, offline_sales
- origin (产地), temperature, humidity, rainfall
- promotion_type (促销类型), cunchao_flag (村超日), holiday_flag
- competitor_price (竞品均价), grade ratios (果实分级)

**模拟数据贴近真实**：旺季8-10月日均10吨、村超效应+30%销量、线上占比约25%、均价16-22元/kg。

### 导入真实数据

点击侧边栏"导入真实数据"，上传CSV/Excel文件，需包含至少以下列：
- date (日期)
- sales_kg (日销量kg)
- avg_price (均价元/kg)

系统会自动匹配其他字段，缺失字段用模拟值填充。

## 模型说明

- **LSTM模式** (需tensorflow): 双层LSTM神经网络，双头输出（销量+价格），适用于GPU环境
- **sklearn降级** (自动): 如tensorflow不可用，自动切换到RandomForest集成模型，CPU友好

## 文件结构

```
rongjiang_passionfruit/
├── app.py              # 主程序（数据生成+分析+预测+定价+策略+UI）
├── requirements.txt    # Python依赖
└── README.md           # 本文件
```

## 榕江百香果背景

- 2025年产值约2亿元
- 旺季：8-10月
- 村超足球联赛（3-11月）期间线上曝光激增
- 线上均价高于线下3-5元/kg
- 特级果占比约20%，一级果55%

---

v1.0 | Powered by AI
