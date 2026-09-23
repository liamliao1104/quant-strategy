#!/usr/bin/env python3
"""
七星ETF轮动策略 - 每日市场分析报告
基于 seven_star.py 策略逻辑，使用 akshare 获取行情数据，生成 HTML 分析报告。
每个交易日 14:10 自动执行，输出 ETF 动量排名与交易建议。
"""

import sys
import os
import math
import json
import time
import datetime
import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:
    print("akshare 未安装，请运行: pip install -r requirements.txt")
    sys.exit(1)

IS_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes")

# ==================== 策略参数（与 seven_star.py 一致） ====================
LOOKBACK_DAYS = 25
HOLDINGS_NUM = 1
DEFENSIVE_ETF = ("511880", "银华日利货币基金")
MIN_MONEY = 5000

PROFIT_PROTECTION_LOOKBACK = 1
PROFIT_PROTECTION_THRESHOLD = 0.05

LOSS_THRESHOLD = 0.97

SHORT_LOOKBACK_DAYS = 10
SHORT_MOMENTUM_THRESHOLD = 0.0

PREMIUM_THRESHOLD = 0.20

VOLUME_LOOKBACK = 5
VOLUME_THRESHOLD = 2.0
VOLUME_RETURN_LIMIT = 1.0

MIN_SCORE = 0
MAX_SCORE = 100.0

LOOKBACK_HIGH_LOW_DAYS = 20
RISK_BENCHMARK = ("510300", "沪深300ETF")
MA_PERIOD = 20
LAPLACE_S_PARAM = 0.05
LAPLACE_MIN_SLOPE = 0.001
GAUSSIAN_SIGMA = 1.2
GAUSSIAN_MIN_SLOPE = 0.002
BIAS_THRESHOLD = 0.10
RSI_OVERBOUGHT = 75
RSI_PULLBACK = 60
LOW_POINT_RISE_THRESHOLD = 0.03
DRAWDOWN_RECOVERY = 0.03
MAX_RANGE_BOUND_DAYS = 15

ETF_POOL = [
    ("518880", "黄金ETF"), ("159980", "有色ETF"), ("159985", "豆粕ETF"),
    ("501018", "南方原油"), ("161226", "白银LOF"), ("159981", "能源化工ETF"),
    ("513100", "纳指ETF"), ("159509", "纳指科技ETF"), ("513290", "纳指生物ETF"),
    ("513500", "标普500ETF"), ("159529", "标普消费"), ("513400", "道琼斯ETF"),
    ("513520", "日经225ETF"), ("513030", "德国30ETF"), ("513080", "法国ETF"),
    ("513310", "中韩半导体ETF"), ("513730", "东南亚ETF"),
    ("159792", "港股互联ETF"), ("513130", "恒生科技"), ("513050", "中概互联网ETF"),
    ("159920", "恒生ETF"), ("513690", "港股红利"),
    ("510300", "沪深300ETF"), ("510500", "中证500ETF"), ("510050", "上证50ETF"),
    ("510210", "上证ETF"), ("159915", "创业板ETF"), ("588080", "科创50"),
    ("512100", "中证1000ETF"), ("563360", "A500-ETF"), ("563300", "中证2000ETF"),
    ("512890", "红利低波ETF"), ("159967", "创业板成长ETF"), ("512040", "价值ETF"),
    ("159201", "自由现金流ETF"),
    ("511380", "可转债ETF"), ("511010", "国债ETF"), ("511220", "城投债ETF"),
]


# ==================== 数据获取 ====================
def _to_sina_symbol(code):
    if code.startswith("5"):
        return f"sh{code}"
    return f"sz{code}"


def get_etf_daily_data(code, days=80, max_retries=3):
    sina_symbol = _to_sina_symbol(code)
    for attempt in range(max_retries):
        try:
            df = ak.fund_etf_hist_sina(sina_symbol)
            if df is None or df.empty:
                return None
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            cutoff = df["date"].iloc[-1] - pd.Timedelta(days=int(days * 1.8))
            df = df[df["date"] >= cutoff].reset_index(drop=True)
            for c in ["open", "close", "high", "low", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                time.sleep(wait)
            else:
                print(f"  [数据获取失败] {code}: {e}")
                return None


def is_trading_day():
    try:
        dates = ak.tool_trade_date_hist_sina()
        today = datetime.date.today()
        dates["trade_date"] = pd.to_datetime(dates["trade_date"]).dt.date
        return today in set(dates["trade_date"].values)
    except Exception:
        return True


# ==================== 技术指标 ====================
def calculate_rsi(close, period=14):
    if len(close) < period + 1:
        return None
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def laplace_filter(price, s=0.05):
    alpha = 1 - np.exp(-s)
    L = np.zeros(len(price))
    L[0] = price[0]
    for t in range(1, len(price)):
        L[t] = alpha * price[t] + (1 - alpha) * L[t - 1]
    return L


def gaussian_filter_last_two(price, sigma=1.2):
    n = len(price)
    if n < 2:
        return 0, 0
    idx = np.arange(n)
    w1 = np.exp(-((idx + 1) ** 2) / (2 * sigma ** 2))[::-1]
    w1 /= np.sum(w1)
    g1 = np.sum(price * w1)
    price2 = price[:-1]
    idx2 = np.arange(n - 1)
    w2 = np.exp(-((idx2 + 1) ** 2) / (2 * sigma ** 2))[::-1]
    w2 /= np.sum(w2)
    g2 = np.sum(price2 * w2)
    return g1, g2


# ==================== 动量计算与过滤 ====================
def calculate_etf_metrics(code, name, df, market_state):
    if df is None or len(df) < LOOKBACK_DAYS + 5:
        return None, "数据不足"

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float) if "high" in df.columns else close
    volume = df["volume"].values.astype(float) if "volume" in df.columns else np.zeros(len(close))
    current_price = close[-1]

    # 1. 盈利保护
    if len(high) >= PROFIT_PROTECTION_LOOKBACK:
        max_high = np.max(high[-PROFIT_PROTECTION_LOOKBACK:])
        if current_price <= max_high * (1 - PROFIT_PROTECTION_THRESHOLD):
            drawdown = (1 - current_price / max_high) * 100
            return None, f"盈利保护(回撤{drawdown:.1f}%>{PROFIT_PROTECTION_THRESHOLD*100:.0f}%)"

    # 2. 短期动量
    if len(close) >= SHORT_LOOKBACK_DAYS + 1:
        short_ret = close[-1] / close[-(SHORT_LOOKBACK_DAYS + 1)] - 1
        short_ann = (1 + short_ret) ** (250 / SHORT_LOOKBACK_DAYS) - 1
    else:
        short_ann = 0.0
    if short_ann < SHORT_MOMENTUM_THRESHOLD:
        return None, f"短期动量为负({short_ann*100:.1f}%)"

    # 3. 长期动量得分
    recent = close[-(LOOKBACK_DAYS + 1):]
    y = np.log(recent)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    annualized = math.exp(slope * 250) - 1

    ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
    ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    score = annualized * r_squared

    # 4. 近3日单日跌幅
    if len(close) >= 4:
        d1 = close[-1] / close[-2]
        d2 = close[-2] / close[-3]
        d3 = close[-3] / close[-4]
        if min(d1, d2, d3) < LOSS_THRESHOLD:
            return None, f"近3日单日跌幅>{(1-LOSS_THRESHOLD)*100:.0f}%"

    # 5. 得分范围
    if not (MIN_SCORE < score < MAX_SCORE):
        return None, f"得分{score:.2f}超出范围"

    # 6. 成交量过滤
    if len(volume) >= VOLUME_LOOKBACK + 1 and volume[-1] > 0:
        avg_vol = np.mean(volume[-(VOLUME_LOOKBACK + 1):-1])
        if avg_vol > 0:
            vol_ratio = volume[-1] / avg_vol
            if vol_ratio > VOLUME_THRESHOLD and annualized > VOLUME_RETURN_LIMIT:
                return None, f"成交量放量{vol_ratio:.1f}倍且年化>{VOLUME_RETURN_LIMIT*100:.0f}%"

    # 7. 动态滤波器
    if len(close) >= 10:
        try:
            laplace_vals = laplace_filter(close, s=LAPLACE_S_PARAM)
            laplace_slope = laplace_vals[-1] - laplace_vals[-2] if len(laplace_vals) >= 2 else 0
            passed_laplace = (current_price > laplace_vals[-1] and laplace_slope > LAPLACE_MIN_SLOPE)
            g1, g2 = gaussian_filter_last_two(close, sigma=GAUSSIAN_SIGMA)
            gaussian_slope = g1 - g2
            passed_gaussian = (current_price > g1 and gaussian_slope > GAUSSIAN_MIN_SLOPE)
            filter_name = "拉普拉斯" if market_state == "正常期" else "高斯"
            passed = passed_laplace if market_state == "正常期" else passed_gaussian
            if not passed:
                return None, f"未通过{filter_name}滤波器"
        except Exception:
            pass

    return {
        "code": code, "name": name, "score": score,
        "annualized": annualized, "r_squared": r_squared,
        "price": current_price, "short_annualized": short_ann,
    }, None


# ==================== 市场状态检测 ====================
def detect_market_state(benchmark_df):
    if benchmark_df is None or len(benchmark_df) < 25:
        return "正常期", {}
    close = benchmark_df["close"].values.astype(float)
    high = benchmark_df["high"].values.astype(float) if "high" in benchmark_df.columns else close
    low = benchmark_df["low"].values.astype(float) if "low" in benchmark_df.columns else close

    current_price = close[-1]
    recent_high = np.max(high[-LOOKBACK_HIGH_LOW_DAYS:])
    recent_low = np.min(low[-LOOKBACK_HIGH_LOW_DAYS:])
    ma = np.mean(close[-MA_PERIOD:])
    bias = (current_price - ma) / ma if ma > 0 else 0
    current_rsi = calculate_rsi(close, period=14)
    prev_rsi = calculate_rsi(close[:-1], period=14) if len(close) > 15 else None
    drawdown = (recent_high - current_price) / recent_high if recent_high > 0 else 0

    signals = []
    should_enter = False
    if bias > BIAS_THRESHOLD:
        should_enter = True
        signals.append(f"乖离率 {bias:.2%} > {BIAS_THRESHOLD:.0%}")
    if (current_rsi is not None and prev_rsi is not None and
            prev_rsi > RSI_OVERBOUGHT and current_rsi < RSI_PULLBACK):
        should_enter = True
        signals.append(f"RSI 超买回落 {prev_rsi:.1f}->{current_rsi:.1f}")

    state = "震荡期" if should_enter else "正常期"
    return state, {
        "bias": bias, "ma": ma, "current_rsi": current_rsi, "prev_rsi": prev_rsi,
        "recent_high": recent_high, "recent_low": recent_low,
        "drawdown": drawdown, "signals": signals,
        "filter": "高斯滤波器" if should_enter else "拉普拉斯滤波器",
    }


# ==================== 报告生成 ====================
def generate_html_report(ranked, market_state, market_info, filtered_list, output_path):
    today = datetime.date.today().strftime("%Y-%m-%d")
    top_pick = ranked[0] if ranked else None
    state_color = "#e74c3c" if market_state == "震荡期" else "#27ae60"
    state_bg = "rgba(231,76,60,0.08)" if market_state == "震荡期" else "rgba(39,174,96,0.08)"
    rsi_val = market_info.get("current_rsi")
    rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "N/A"
    bias_val = market_info.get("bias")
    bias_str = f"{bias_val:.2%}" if bias_val is not None else "N/A"
    signals_str = "; ".join(market_info.get("signals", [])) or "无风险信号"

    rows_html = ""
    for i, m in enumerate(ranked[:20]):
        cls = ' class="rank-top"' if i == 0 else ""
        rows_html += f"""
        <tr{cls}>
            <td>{i+1}</td><td>{m['code']}</td><td>{m['name']}</td>
            <td class="score">{m['score']:.4f}</td>
            <td>{m['annualized']*100:.2f}%</td>
            <td>{m['r_squared']:.4f}</td>
            <td>{m['short_annualized']*100:.2f}%</td>
            <td>{m['price']:.3f}</td>
        </tr>"""

    filtered_html = ""
    for code, name, reason in filtered_list:
        filtered_html += f'<div class="filtered-item">{code} {name} — {reason}</div>'

    if top_pick:
        advice = f"""
        <div class="advice-card buy">
            <div class="advice-icon">买入</div>
            <div class="advice-content">
                <div class="advice-title">{top_pick['code']} {top_pick['name']}</div>
                <div class="advice-detail">动量得分 {top_pick['score']:.4f} | 年化 {top_pick['annualized']*100:.2f}% | R2 {top_pick['r_squared']:.4f}</div>
                <div class="advice-detail">滤波器: {market_info.get('filter', '拉普拉斯滤波器')} | 市场状态: {market_state}</div>
            </div>
        </div>"""
    else:
        advice = """
        <div class="advice-card defensive">
            <div class="advice-icon">防御</div>
            <div class="advice-content">
                <div class="advice-title">无ETF通过全部过滤条件</div>
                <div class="advice-detail">建议持有防御ETF(货币基金 511880)或保持空仓</div>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ETF动量轮动分析报告 - {today}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#f5f7fa;color:#2c3e50;line-height:1.6;padding:20px}}
.container{{max-width:960px;margin:0 auto}}
.header{{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:#fff;padding:30px;border-radius:12px;margin-bottom:20px}}
.header h1{{font-size:24px;margin-bottom:8px}}
.header .date{{font-size:14px;opacity:.7}}
.market-state{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600;margin-top:12px;background:{state_bg};color:{state_color};border:1px solid {state_color}}}
.section{{background:#fff;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.section h2{{font-size:18px;margin-bottom:16px;border-left:4px solid #3498db;padding-left:12px}}
.advice-card{{display:flex;align-items:center;gap:16px;padding:20px;border-radius:10px}}
.advice-card.buy{{background:rgba(39,174,96,.08);border:1px solid rgba(39,174,96,.3)}}
.advice-card.defensive{{background:rgba(243,156,18,.08);border:1px solid rgba(243,156,18,.3)}}
.advice-icon{{font-size:14px;font-weight:700;color:#fff;padding:6px 14px;border-radius:6px;white-space:nowrap}}
.advice-card.buy .advice-icon{{background:#27ae60}}
.advice-card.defensive .advice-icon{{background:#f39c12}}
.advice-title{{font-size:18px;font-weight:600}}
.advice-detail{{font-size:13px;color:#7f8c8d;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{background:#f8f9fa;padding:10px 12px;text-align:left;font-weight:600;color:#6c757d;border-bottom:2px solid #dee2e6;white-space:nowrap}}
td{{padding:10px 12px;border-bottom:1px solid #e9ecef}}
tr.rank-top{{background:rgba(39,174,96,.05)}}
tr.rank-top td{{font-weight:600}}
td.score{{font-family:"SF Mono",monospace;font-weight:600;color:#2980b9}}
.market-info{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:12px}}
.info-item{{background:#f8f9fa;padding:12px;border-radius:8px}}
.info-label{{font-size:12px;color:#6c757d;margin-bottom:4px}}
.info-value{{font-size:16px;font-weight:600}}
.filtered-item{{padding:6px 0;font-size:13px;color:#95a5a6;border-bottom:1px solid #ecf0f1}}
.footer{{text-align:center;font-size:12px;color:#bdc3c7;margin-top:20px;padding:20px}}
.signals{{margin-top:12px;padding:10px;background:{state_bg};border-radius:8px;font-size:13px;color:{state_color}}}
</style>
</head>
<body>
<div class="container">
<div class="header">
    <h1>ETF 七星ETF本地轮动策略 - 每日分析报告</h1>
    <div class="date">{today} | 基于沪深300ETF风险基准</div>
    <div class="market-state">市场状态: {market_state}({market_info.get('filter','拉普拉斯滤波器')})</div>
</div>
<div class="section"><h2>交易建议</h2>{advice}</div>
<div class="section">
    <h2>市场状态分析</h2>
    <div class="market-info">
        <div class="info-item"><div class="info-label">乖离率</div><div class="info-value">{bias_str}</div></div>
        <div class="info-item"><div class="info-label">RSI(14)</div><div class="info-value">{rsi_str}</div></div>
        <div class="info-item"><div class="info-label">20日均线</div><div class="info-value">{market_info.get('ma',0):.3f}</div></div>
        <div class="info-item"><div class="info-label">近20日高点</div><div class="info-value">{market_info.get('recent_high',0):.3f}</div></div>
        <div class="info-item"><div class="info-label">近20日低点</div><div class="info-value">{market_info.get('recent_low',0):.3f}</div></div>
        <div class="info-item"><div class="info-label">当前回撤</div><div class="info-value">{market_info.get('drawdown',0):.2%}</div></div>
    </div>
    <div class="signals"><strong>风险信号:</strong> {signals_str}</div>
</div>
<div class="section">
    <h2>ETF 动量排名(通过全部过滤)</h2>
    <table>
        <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>得分</th><th>年化收益</th><th>R2</th><th>短期年化</th><th>现价</th></tr></thead>
        <tbody>{rows_html or '<tr><td colspan="8" style="text-align:center;color:#999;padding:20px;">无ETF通过全部过滤条件</td></tr>'}</tbody>
    </table>
</div>
<div class="section">
    <h2>被过滤的ETF(共{len(filtered_list)}只)</h2>
    {filtered_html or '<div style="color:#999;padding:10px;">无被过滤的ETF</div>'}
</div>
<div class="footer">
    本报告由 ETF 动量轮动策略自动生成 | 仅供参考,不构成投资建议<br>
    策略参数: 动量周期{LOOKBACK_DAYS}天 | 持仓{HOLDINGS_NUM}只 | 盈利保护{PROFIT_PROTECTION_THRESHOLD*100:.0f}% | 溢价率阈值{PREMIUM_THRESHOLD*100:.0f}%
</div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML报告已生成: {output_path}")


# ==================== 主函数 ====================
def main():
    print(f"===== ETF 动量分析开始 {datetime.datetime.now()} =====")

    if not is_trading_day():
        print("今天非交易日,跳过分析。")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)

    today_str = datetime.date.today().strftime("%Y%m%d")
    report_path = os.path.join(output_dir, f"etf_analysis_{today_str}.html")
    json_path = os.path.join(output_dir, f"etf_analysis_{today_str}.json")

    # 1. 基准数据 & 市场状态
    print("正在获取风险基准数据(沪深300ETF)...")
    benchmark_df = get_etf_daily_data(RISK_BENCHMARK[0], days=80)
    market_state, market_info = detect_market_state(benchmark_df)
    print(f"市场状态: {market_state} ({market_info.get('filter', '')})")
    if market_info.get("signals"):
        print(f"  风险信号: {'; '.join(market_info['signals'])}")

    # 2. 全ETF扫描
    print(f"正在分析 {len(ETF_POOL)} 只ETF...")
    ranked = []
    filtered_list = []
    for i, (code, name) in enumerate(ETF_POOL):
        print(f"  [{i+1}/{len(ETF_POOL)}] {code} {name}...", end=" ")
        df = get_etf_daily_data(code, days=80)
        time.sleep(0.3 if IS_CI else 0.5)
        metrics, reason = calculate_etf_metrics(code, name, df, market_state)
        if metrics is not None:
            print(f"得分={metrics['score']:.4f}")
            ranked.append(metrics)
        else:
            print(f"过滤({reason})")
            filtered_list.append((code, name, reason or "未知"))

    ranked.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n通过过滤: {len(ranked)}只, 被过滤: {len(filtered_list)}只")
    if ranked:
        print(f"  Top1: {ranked[0]['code']} {ranked[0]['name']} 得分={ranked[0]['score']:.4f}")

    # 3. 生成报告
    generate_html_report(ranked, market_state, market_info, filtered_list, report_path)

    # 4. JSON摘要
    def safe_float(v):
        if v is None: return None
        if isinstance(v, (np.floating, np.integer)): return float(v)
        if isinstance(v, (int, float)): return float(v)
        return None

    summary = {
        "date": today_str,
        "market_state": market_state,
        "market_info": {
            "bias": safe_float(market_info.get("bias")),
            "current_rsi": safe_float(market_info.get("current_rsi")),
            "ma": safe_float(market_info.get("ma")),
            "recent_high": safe_float(market_info.get("recent_high")),
            "recent_low": safe_float(market_info.get("recent_low")),
            "drawdown": safe_float(market_info.get("drawdown")),
            "filter": market_info.get("filter", ""),
        },
        "market_signals": market_info.get("signals", []),
        "ranked": [{k: safe_float(v) if isinstance(v, (int, float, np.floating)) else v
                      for k, v in m.items()} for m in ranked[:10]],
        "filtered_count": len(filtered_list),
        "filtered": [{"code": c, "name": n, "reason": r} for c, n, r in filtered_list],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"JSON摘要已生成: {json_path}")
    print("===== 分析完成 =====")


if __name__ == "__main__":
    main()
