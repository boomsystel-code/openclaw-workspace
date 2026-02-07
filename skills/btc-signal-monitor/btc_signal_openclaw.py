#!/usr/bin/env python3
"""
BTC 信号监控系统 - OpenClaw 集成版
支持直接使用 OpenClaw message 工具发送 Telegram 报警
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# 尝试导入 OpenClaw 消息工具
try:
    from openclaw_telegram_skill import send_telegram_message
    HAS_OPENCLAW_MESSAGE = True
except ImportError:
    HAS_OPENCLAW_MESSAGE = False

# ============ 配置 ============

CONFIG = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "ma_fast": 7,
    "ma_slow": 25,
    "check_interval": 60,
    "cooldown_minutes": 60,
    # 报警开关
    "alert_on_signal_change": True,
    "alert_on_strong_signal": True,
}

# 信号定义
SignalType = type('SignalType', (), {
    'BUY': 'BUY',
    'SELL': 'SELL',
    'WAIT': 'WAIT',
    'STRONG_BUY': 'STRONG_BUY',
    'STRONG_SELL': 'STRONG_SELL'
})()

# ============ 核心功能 ============

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ma(prices, period):
    if len(prices) < period:
        return prices[-1] if prices else 0
    return sum(prices[-period:]) / period

def calculate_volatility(prices, period=24):
    if len(prices) < 2:
        return 0.0
    returns = [(prices[i] / prices[i-1] - 1) for i in range(1, min(len(prices), period + 1))]
    if not returns:
        return 0.0
    import statistics
    return statistics.stdev(returns) * 100

def analyze_market(candles):
    closes = [c['close'] for c in candles]
    current_price = closes[-1]
    rsi = calculate_rsi(closes, CONFIG['rsi_period'])
    ma_fast = calculate_ma(closes, CONFIG['ma_fast'])
    ma_slow = calculate_ma(closes, CONFIG['ma_slow'])
    ma7 = calculate_ma(closes, 7)
    ma25 = calculate_ma(closes, 25)
    ma99 = calculate_ma(closes, 99)
    volatility = calculate_volatility(closes[-24:])
    
    import statistics
    highs = [c['high'] for c in candles[-20:]]
    lows = [c['low'] for c in candles[-20:]]
    support = statistics.mean(lows) - statistics.stdev(lows) if len(lows) > 1 else min(lows)
    resistance = statistics.mean(highs) + statistics.stdev(highs) if len(highs) > 1 else max(highs)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'symbol': CONFIG['symbol'],
        'current_price': current_price,
        'rsi': {'value': round(rsi, 2), 'status': 'OVERSOLD' if rsi < CONFIG['rsi_oversold'] else 'OVERBOUGHT' if rsi > CONFIG['rsi_overbought'] else 'NEUTRAL'},
        'ma': {'ma7': round(ma7, 2), 'ma25': round(ma25, 2), 'ma99': round(ma99, 2), 'crossover': 'GOLDEN' if ma7 > ma25 else 'DEATH'},
        'volatility': round(volatility, 2),
        'support': round(support, 2),
        'resistance': round(resistance, 2),
        'trend': 'UPTREND' if current_price > ma99 else 'DOWNTREND',
        'short_trend': 'BULLISH' if ma7 > ma25 else 'BEARISH'
    }

def generate_signal(analysis):
    rsi = analysis['rsi']['value']
    ma_trend = analysis['ma']['crossover']
    short_trend = analysis['short_trend']
    volatility = analysis['volatility']
    
    score = 0
    reasons = []
    
    if rsi < 25:
        score += 3
        reasons.append(f"RSI 极度超卖 ({rsi:.1f})")
    elif rsi < 30:
        score += 2
        reasons.append(f"RSI 超卖 ({rsi:.1f})")
    elif rsi > 75:
        score -= 3
        reasons.append(f"RSI 极度超买 ({rsi:.1f})")
    elif rsi > 70:
        score -= 2
        reasons.append(f"RSI 超买 ({rsi:.1f})")
    else:
        reasons.append(f"RSI 中性 ({rsi:.1f})")
    
    if ma_trend == 'GOLDEN':
        score += 2
        reasons.append("均线金叉")
    elif ma_trend == 'DEATH':
        score -= 2
        reasons.append("均线死叉")
    
    if short_trend == 'BULLISH':
        score += 1
        reasons.append("短期趋势上涨")
    else:
        score -= 1
        reasons.append("短期趋势下跌")
    
    if volatility > 3:
        score -= 1
        reasons.append(f"波动率较高 ({volatility:.1f}%)")
    
    if score >= 3:
        signal = SignalType.STRONG_BUY
    elif score >= 1:
        signal = SignalType.BUY
    elif score <= -3:
        signal = SignalType.STRONG_SELL
    elif score <= -1:
        signal = SignalType.SELL
    else:
        signal = SignalType.WAIT
    
    return {
        'signal': signal,
        'score': score,
        'confidence': min(abs(score) * 20 + 50, 95),
        'reasons': reasons,
        'recommendation': get_recommendation(signal, analysis)
    }

def get_recommendation(signal, analysis):
    price = analysis['current_price']
    support = analysis['support']
    resistance = analysis['resistance']
    
    recs = {
        SignalType.STRONG_BUY: f"🔥 **强烈买入信号**\\n当前价格：${price:,.0f}\\n建议：考虑分批建仓，止损 ${support:,.0f}",
        SignalType.BUY: f"🟢 **买入信号**\\n当前价格：${price:,.0f}\\n建议：小仓位试探性买入，止损 ${support:,.0f}",
        SignalType.WAIT: f"🟡 **观望**\\n当前价格：${price:,.0f}\\n支撑 ${support:,.0f} / 阻力 ${resistance:,.0f}",
        SignalType.SELL: f"🔴 **卖出信号**\\n当前价格：${price:,.0f}\\n建议：减仓锁定利润",
        SignalType.STRONG_SELL: f"🚨 **强烈卖出信号**\\n当前价格：${price:,.0f}\\n建议：减仓或清仓"
    }
    return recs.get(signal, "建议观望")

async def fetch_btc_data():
    import random
    base_price = 70000 + random.uniform(-5000, 5000)
    candles = []
    for i in range(100):
        timestamp = datetime.now().timestamp() - (99 - i) * 3600
        change = random.uniform(-0.02, 0.02)
        open_p = base_price * (1 + random.uniform(-0.01, 0.01))
        close_p = open_p * (1 + change)
        high_p = max(open_p, close_p) * (1 + random.uniform(0, 0.005))
        low_p = min(open_p, close_p) * (1 - random.uniform(0, 0.005))
        candles.append({
            'timestamp': timestamp,
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': random.uniform(1000, 5000)
        })
        base_price = close_p
    return candles

def format_telegram_message(analysis, signal):
    """格式化 Telegram 消息"""
    
    emoji_map = {
        SignalType.STRONG_BUY: "🔥",
        SignalType.BUY: "🟢",
        SignalType.SELL: "🔴",
        SignalType.STRONG_SELL: "🚨",
        SignalType.WAIT: "🟡"
    }
    
    emoji = emoji_map.get(signal['signal'], "📊")
    
    return f"""
{emoji} *BTC 信号提醒*

📊 *当前状态*
• 价格：`${analysis['current_price']:,.0f}`
• RSI：`{analysis['rsi']['value']}` ({analysis['rsi']['status']})
• 趋势：`{analysis['trend']}` / `{analysis['short_trend']}`

📈 *均线*
• MA7：`${analysis['ma']['ma7']:,.0f}`
• MA25：`${analysis['ma']['ma25']:,.0f}`
• 交叉：`{analysis['ma']['crossover']}`

🎯 *信号*
• 类型：`{signal['signal']}`
• 评分：`{signal['score']}`
• 置信度：`{signal['confidence']:.0f}%`

💡 *建议*
{signal['recommendation']}

⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

🤖 *BTC Signal Monitor*
"""

# ============ OpenClaw 消息发送 ============

async def send_via_openclaw(message: str) -> bool:
    """通过 OpenClaw 发送消息"""
    
    if not HAS_OPENCLAW_MESSAGE:
        print("⚠️ OpenClaw 消息工具不可用")
        return False
    
    try:
        result = await send_telegram_message(
            message=message,
            chat_id="main"  # 发送到主会话
        )
        if result:
            print("✅ OpenClaw 消息已发送")
            return True
    except Exception as e:
        print(f"❌ OpenClaw 发送失败: {e}")
    
    return False

# ============ 主程序 ============

async def run_monitor(send_telegram: bool = False):
    """运行监控"""
    
    print(f"""
╔══════════════════════════════════════╗
║   🚨 BTC 信号监控系统              ║
║   OpenClaw Telegram 集成版          ║
╠══════════════════════════════════════╣
║  品种：{CONFIG['symbol']:<27}║
║  周期：{CONFIG['timeframe']:<27}║
║  Telegram：{'开启' if send_telegram else '关闭'}{'':<24}║
╚══════════════════════════════════════╝
""")
    
    # 获取数据
    print("📊 获取数据...")
    candles = await fetch_btc_data()
    print(f"✅ 获取 {len(candles)} 条 K 线")
    
    # 分析
    analysis = analyze_market(candles)
    signal = generate_signal(analysis)
    
    # 格式化消息
    message = format_telegram_message(analysis, signal)
    
    # 打印结果
    print(f"""
📊 当前价格：${analysis['current_price']:,.0f}
📉 RSI：{analysis['rsi']['value']} ({analysis['rsi']['status']})
📈 MA7：${analysis['ma']['ma7']:,.0f} | MA25：${analysis['ma']['ma25']:,.0f}
📐 交叉：{analysis['ma']['crossover']}

🎯 信号：**{signal['signal']}**
   评分：{signal['score']} | 置信度：{signal['confidence']:.0f}%
""")
    
    # 发送到 Telegram
    if send_telegram:
        print("📱 发送 Telegram 消息...")
        success = await send_via_openclaw(message)
        if success:
            print("✅ Telegram 报警已发送！")
        else:
            print("❌ Telegram 发送失败")
    else:
        print("📱 Telegram 消息预览：")
        print("="*50)
        print(message)
        print("="*50)
    
    return analysis, signal


async def main():
    """主函数"""
    
    send_telegram = "--telegram" in os.sys.argv or "-t" in os.sys.argv
    
    await run_monitor(send_telegram=send_telegram)


if __name__ == "__main__":
    asyncio.run(main())
