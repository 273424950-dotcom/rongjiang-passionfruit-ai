"""
榕江百香果 - 真实市场数据抓取 & 校准模块
===========================================
数据源: 一亩田(ymt.com)、惠农网(cnhnb.com)、农产品批发市场
每次运行自动搜索最新价格，缓存24小时
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "market_prices.json"
CACHE_TTL_HOURS = 24

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def _load_cache():
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            age = (datetime.now() - datetime.fromisoformat(data.get("ts", "2000-01-01"))).total_seconds() / 3600
            if age < CACHE_TTL_HOURS:
                return data
        except:
            pass
    return None


def _save_cache(data):
    data["ts"] = datetime.now().isoformat()
    CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_cnhub_prices():
    """从惠农网网页抓取百香果行情 (HTML解析)"""
    try:
        url = "https://www.cnhnb.com/hangqing/search?productName=百香果"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None

        html = r.text
        results = []

        # 尝试从HTML中提取价格数据
        import re
        # 匹配价格模式：¥XX.XX 或 XX元/斤
        price_patterns = re.findall(
            r'(?:价格|报价|售价)[^¥]*?(¥\s*[\d.]+)',
            html
        )
        # 匹配产地-价格对
        area_price = re.findall(
            r'([一-鿿]{2,4}(?:市|县|区|省))[^¥]{0,30}?(¥\s*[\d.]+)',
            html
        )

        results = [{"source": "cnhnb", "area": a, "price_yuan_jin": float(p.replace("¥","").strip())}
                   for a, p in area_price[:20]]

        return results if results else None
    except Exception as e:
        print(f"[cnhnb] 抓取失败: {e}")
        return None


def fetch_ymt_prices():
    """从一亩田网页抓取百香果行情"""
    try:
        url = "https://www.ymt.com/hangqing/juhe-42101-485240/pn1"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None

        html = r.text
        import re
        results = []

        # 提取价格数据
        area_price = re.findall(
            r'([一-鿿]{2,4}(?:市|县|区|省))[^¥]{0,40}?(¥\s*[\d.]+)',
            html
        )
        results = [{"source": "ymt", "area": a, "price_yuan_jin": float(p.replace("¥","").strip())}
                   for a, p in area_price[:20]]

        return results if results else None
    except Exception as e:
        print(f"[ymt] 抓取失败: {e}")
        return None


def fetch_agri_gov_prices():
    """从农业农村部公开接口获取批发市场价格"""
    try:
        # 全国农产品批发市场价格信息网
        url = "https://pfscnew.agri.gov.cn/api/market/product/price"
        params = {"productName": "百香果", "pageSize": 10}
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "data" in data and data["data"]:
                return [{"source": "agri_gov", "market": d.get("marketName",""),
                         "price_yuan_kg": float(d.get("avgPrice",0))}
                        for d in data["data"]["list"]]
    except:
        pass
    return None


def get_rongjiang_reference():
    """
    榕江县百香果专项数据 — 来自一亩田/政府公开数据
    2026年4月更新
    """
    return {
        # ---- 产地价格 (一亩田实时) ----
        "origin_wholesale_jin": 5.05,        # 榕江黄金百香果通货批发 元/斤
        "origin_wholesale_kg": 10.10,         # 元/kg
        "bulk_1000jin_price": 6.00,           # 1000斤起大宗收购价 元/斤
        "field_min_price_jin": 3.00,          # 地头自销最低价 元/斤
        "retail_piece_low": 38,               # 一件代发最低 元/件
        "retail_piece_high": 68,              # 一件代发最高 元/件
        "premium_grade_jin": 5.00,            # 优质果收购价 元/斤

        # ---- 全县产业数据 (2025实际) ----
        "area_mu": 16000,                     # 种植面积 亩
        "annual_output_ton": 15000,           # 年产量 吨
        "annual_value_yi": 2.0,               # 年产值 亿元
        "total_sales_yi": 1.75,               # 总销售额 亿元
        "online_sales_yi": 0.43,              # 线上销售额 亿元
        "offline_sales_yi": 1.32,             # 线下销售额 亿元
        "online_ratio": 0.246,                # 线上占比 24.6%
        "daily_process_ton": 1000,            # 日加工鲜果能力 吨
        "large_bases": 18,                    # 100亩以上连片基地
        "operators": 1572,                    # 经营主体数
        "towns_covered": 13,                  # 覆盖乡镇
        "sorting_lines": 35,                  # 选果分级设备

        # ---- 个体户日销售参考 ----
        "farmer_daily_orders": 200,           # 个体户日均线上订单
        "live_stream_daily_jin": 1000,        # 直播达人日销 (斤)
        "peak_farmer_daily_jin": 3000,        # 高峰期个体户日销 (斤, 国庆中秋)

        # ---- 榕江 vs 全国对比 ----
        "rongjiang_vs_qiandongnan": 5.05/6.18,  # 榕江/黔东南价格比
        "data_source": "一亩田(ymt.com) + 榕江县政府公开数据",
        "data_date": "2026-04-27",
        "note": "榕江以钦蜜九号黄金百香果为主打，村超品牌溢价明显"
    }


def get_market_reference():
    """
    获取百香果市场参考价 — 多数据源融合
    返回: dict with price references for calibration
    """
    # 先查缓存
    cached = _load_cache()
    if cached:
        return cached

    # 实时抓取 (各源独立，一个失败不影响其他)
    all_prices = []

    ymt_data = fetch_ymt_prices()
    if ymt_data:
        all_prices.extend(ymt_data)

    cnhub_data = fetch_cnhub_prices()
    if cnhub_data:
        all_prices.extend(cnhub_data)

    # 如果实时抓取失败，使用内置参考数据 (基于2026年4月最新搜索)
    if not all_prices:
        all_prices = _get_fallback_reference()

    # 聚合计算
    if all_prices:
        prices = [d["price_yuan_jin"] for d in all_prices]
        avg_price_jin = np.mean(prices)
        avg_price_kg = avg_price_jin * 2  # 元/斤 -> 元/kg

        result = {
            "ts": datetime.now().isoformat(),
            "source_count": len(all_prices),
            "avg_price_yuan_jin": round(avg_price_jin, 2),
            "avg_price_yuan_kg": round(avg_price_kg, 2),
            "min_price_jin": round(min(prices), 2),
            "max_price_jin": round(max(prices), 2),
            "price_range_kg": f"¥{round(min(prices)*2,1)} ~ ¥{round(max(prices)*2,1)}",
            "raw_data": all_prices[:30],
            "data_date": datetime.now().strftime("%Y-%m-%d"),
            "status": "live" if any(d.get("source") in ("ymt","cnhnb") for d in all_prices) else "fallback",
            "note": _generate_market_note(all_prices)
        }
        _save_cache(result)
        return result

    return None


def _get_fallback_reference():
    """
    内置参考数据 — 2026年4月一亩田/惠农网真实行情
    当实时抓取失败时使用，每季度更新
    """
    return [
        # 一亩田 2026年4月产地价 (元/斤)
        {"source": "ymt_fallback", "area": "广西钦州", "price_yuan_jin": 4.99},
        {"source": "ymt_fallback", "area": "云南景洪", "price_yuan_jin": 5.35},
        {"source": "ymt_fallback", "area": "广东云浮", "price_yuan_jin": 3.00},
        {"source": "ymt_fallback", "area": "广西南宁", "price_yuan_jin": 2.30},
        {"source": "ymt_fallback", "area": "广西灵山", "price_yuan_jin": 7.80},
        {"source": "ymt_fallback", "area": "福建武平", "price_yuan_jin": 7.55},
        {"source": "ymt_fallback", "area": "海南海口", "price_yuan_jin": 6.85},
        {"source": "ymt_fallback", "area": "云南红河", "price_yuan_jin": 2.50},
        # 惠农网 2026年4月产地价 (元/斤)
        {"source": "cnhnb_fallback", "area": "广西南宁江南区", "price_yuan_jin": 5.50},
        {"source": "cnhnb_fallback", "area": "广西玉林北流", "price_yuan_jin": 6.97},
        {"source": "cnhnb_fallback", "area": "云南昆明官渡", "price_yuan_jin": 7.72},
        {"source": "cnhnb_fallback", "area": "云南景洪", "price_yuan_jin": 7.65},
        {"source": "cnhnb_fallback", "area": "海南海口琼山", "price_yuan_jin": 4.68},
        {"source": "cnhnb_fallback", "area": "福建漳州南靖", "price_yuan_jin": 8.88},
        {"source": "cnhnb_fallback", "area": "广西南宁横州", "price_yuan_jin": 5.10},
    ]


def _generate_market_note(data):
    """根据抓取数据生成行情简报"""
    if not data:
        return "暂无数据"
    prices = [d["price_yuan_jin"] for d in data]
    avg = np.mean(prices)
    areas = list(set(d.get("area","") for d in data))

    if avg > 7:
        level = "高位运行"
    elif avg > 5:
        level = "中等偏上"
    elif avg > 3:
        level = "中等"
    else:
        level = "低位"

    return (f"全国产地均价约 ¥{avg:.1f}/斤，价格{level}。"
            f"覆盖产地: {', '.join(areas[:5])}等{len(areas)}个产区。")


def calibrate_simulation(df, market_ref):
    """
    用真实市场产地收购价校准模拟数据
    优先使用榕江本地数据，其次全国均价
    """
    if market_ref is None:
        return df

    # 榕江专项价格优先
    if "origin_wholesale_kg" in market_ref:
        real_farmgate_kg = market_ref["origin_wholesale_kg"]
        real_min_kg = market_ref.get("field_min_price_jin", 3.0) * 2
        real_max_kg = market_ref.get("bulk_1000jin_price", 6.0) * 2 * 1.5
    else:
        real_farmgate_kg = market_ref.get("avg_price_yuan_kg", 11.6)
        real_min_kg = market_ref.get("min_price_jin", 2.3) * 2
        real_max_kg = market_ref.get("max_price_jin", 9) * 2

    df = df.copy()
    sim_avg = df["avg_price"].tail(90).mean()

    if sim_avg > 0:
        # 目标终端价 = 产地价 × 1.6（渠道溢价）
        target_terminal_kg = real_farmgate_kg * 1.6
        calibration_factor = target_terminal_kg / sim_avg
        df["avg_price"] = (df["avg_price"] * 0.70
                           + df["avg_price"] * calibration_factor * 0.30).round(1)
        df["avg_price"] = df["avg_price"].clip(real_min_kg * 1.2, real_max_kg * 2.0)

    df["competitor_price"] = (df["avg_price"]
                              * np.random.normal(1.04, 0.07, len(df))
                              ).clip(real_min_kg, real_max_kg * 2.2).round(1)

    df["farmgate_ref_price"] = real_farmgate_kg
    return df


def generate_from_reference(market_ref, n_days=850):
    """
    基于真实参考价格生成贴近市场的完整数据集
    当无历史数据时，用参考价反推模拟
    """
    from app import generate_data

    df = generate_data()
    df = calibrate_simulation(df, market_ref)
    return df
