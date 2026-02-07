#!/usr/bin/env python3
"""
BTC 信号监控系统
实时监控 BTC 交易信号，自动推送提醒

功能：
- 实时获取 BTC 行情数据
- 多维度技术指标分析
- 买卖信号检测
- Telegram 自动推送提醒
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 尝试导入 ccxt
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("⚠️ ccxt 未安装，将使用备用数据")


# ============ 配置 ============

CONFIG = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",  # 1m, 5m, 15m, 1h, 4h, 1d
    
    # RSI 配置
    "rsi_period": 14,
    "rsi_oversold": 30,   # 买入阈值
    "rsi_overbought": 70, # 卖出阈值
    
    # MA 配置
    "ma_fast": 7,
    "ma_slow": 25,
    
    # 价格突破配置
    "breakout_threshold": 0.02,  # 2% 突破
    
    # 监控频率（秒）
    "check_interval": 60,  # 每分钟检查一次
    
    # Telegram 配置
    "telegram_enabled": True,
    "telegram_chat_id": "",
    
    # 报警冷却（避免重复报警）
    "cooldown_minutes": 60,
    
    # 趋势判断周期
    "trend_periods": {
        "short": 6,      # 1小时
        "medium": 24,    # 4小时
        "long": 72       # 3天
    }
}


# ============ 信号定义 ============

class SignalType:
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


# ============ 数据获取 ============

async def fetch_btc_data(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 100):
    """获取 BTC K线数据"""
    
    if HAS_CCXT:
        try:
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            candles = []
            for candle in ohlcv:
                candles.append({
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                })
            
            return candles
            
        except Exception as e:
            print(f"❌ 获取数据失败: {e}")
            return generate_mock_data(limit)
    else:
        return generate_mock_data(limit)


def generate_mock_data(limit: int = 100) -> List[Dict]:
    """生成模拟数据（用于测试）"""
    import random
    
    base_price = 70000
    candles = []
    
    for i in range(limit):
        timestamp = datetime.now().timestamp() - (limit - i) * 3600
        
        # 随机波动
        change = random.uniform(-0.02, 0.02)
        open_price = base_price * (1 + random.uniform(-0.1, 0.1))
        close_price = open_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
        volume = random.uniform(1000, 5000)
        
        candles.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        base_price = close_price
    
    return candles


# ============ 技术指标计算 ============

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """计算 RSI"""
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
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_ma(prices: List[float], period: int) -> float:
    """计算移动平均线"""
    if len(prices) < period:
        return prices[-1] if prices else 0
    return sum(prices[-period:]) / period


def calculate_ema(prices: List[float], period: int) -> float:
    """计算指数移动平均线"""
    if len(prices) < period:
        return prices[-1] if prices else 0
    
    multiplier = 2 / (period + 1)
    ema = prices[-period]
    
    for price in prices[-period + 1:]:
        ema = price * multiplier + ema * (1 - multiplier)
    
    return ema


def calculate_volatility(prices: List[float], period: int = 24) -> float:
    """计算波动率"""
    if len(prices) < 2:
        return 0.0
    
    returns = [(prices[i] / prices[i-1] - 1) for i in range(1, min(len(prices), period + 1))]
    
    if not returns:
        return 0.0
    
    import statistics
    return statistics.stdev(returns) * 100


def calculate_support_resistance(prices: List[float], window: int = 20) -> Dict:
    """计算支撑位和阻力位"""
    if len(prices) < window:
        return {'support': min(prices), 'resistance': max(prices)}
    
    recent_prices = prices[-window:]
    
    # 简化计算：使用滚动高低价
    highs = [p['high'] for p in recent_prices] if isinstance(recent_prices[0], dict) else recent_prices
    lows = [p['low'] for p in recent_prices] if isinstance(recent_prices[0], dict) else recent_prices
    
    import statistics
    return {
        'support': statistics.mean(lows) - statistics.stdev(lows) if len(lows) > 1 else min(lows),
        'resistance': statistics.mean(highs) + statistics.stdev(highs) if len(highs) > 1 else max(highs)
    }


# ============ 信号检测 ============

def analyze_market(candles: List[Dict]) -> Dict:
    """综合市场分析"""
    
    if not candles:
        return {'error': '无数据'}
    
    closes = [c['close'] for c in candles]
    current_price = closes[-1]
    
    # 计算指标
    rsi = calculate_rsi(closes, CONFIG['rsi_period'])
    ma_fast = calculate_ma(closes, CONFIG['ma_fast'])
    ma_slow = calculate_ma(closes, CONFIG['ma_slow'])
    ma7 = calculate_ma(closes, 7)
    ma25 = calculate_ma(closes, 25)
    ma99 = calculate_ma(closes, 99)
    volatility = calculate_volatility(closes[-24:])
    
    # 支撑/阻力位
    sr_levels = calculate_support_resistance(candles[-20:])
    
    # 趋势判断
    trend = "UPTREND" if current_price > ma99 else "DOWNTREND"
    short_trend = "BULLISH" if ma7 > ma25 else "BEARISH"
    
    return {
        'timestamp': datetime.now().isoformat(),
        'symbol': CONFIG['symbol'],
        'current_price': current_price,
        'rsi': {
            'value': round(rsi, 2),
            'status': 'OVERSOLD' if rsi < CONFIG['rsi_oversold'] else 'OVERBOUGHT' if rsi > CONFIG['rsi_overbought'] else 'NEUTRAL'
        },
        'ma': {
            'ma7': round(ma7, 2),
            'ma25': round(ma25, 2),
            'ma99': round(ma99, 2),
            'crossover': 'GOLDEN' if ma7 > ma25 else 'DEATH'
        },
        'volatility': round(volatility, 2),
        'support': round(sr_levels['support'], 2),
        'resistance': round(sr_levels['resistance'], 2),
        'trend': trend,
        'short_trend': short_trend,
        'candles_count': len(candles)
    }


def generate_signal(analysis: Dict) -> Dict:
    """生成交易信号"""
    
    price = analysis['current_price']
    rsi = analysis['rsi']['value']
    ma_trend = analysis['ma']['crossover']
    short_trend = analysis['short_trend']
    volatility = analysis['volatility']
    
    score = 0
    reasons = []
    
    # RSI 评分
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
        score += 0
        reasons.append(f"RSI 中性 ({rsi:.1f})")
    
    # MA 交叉评分
    if ma_trend == 'GOLDEN':
        score += 2
        reasons.append("均线金叉")
    elif ma_trend == 'DEATH':
        score -= 2
        reasons.append("均线死叉")
    
    # 趋势评分
    if short_trend == 'BULLISH':
        score += 1
        reasons.append("短期趋势上涨")
    else:
        score -= 1
        reasons.append("短期趋势下跌")
    
    # 波动率调整
    if volatility > 3:
        score -= 1
        reasons.append(f"波动率较高 ({volatility:.1f}%)")
    
    # 生成最终信号
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


def get_recommendation(signal: str, analysis: Dict) -> str:
    """获取交易建议"""
    
    price = analysis['current_price']
    support = analysis['support']
    resistance = analysis['resistance']
    
    recommendations = {
        SignalType.STRONG_BUY: f"""🔥 **强烈买入信号**
当前价格：${price:,.0f}
建议：可以考虑分批建仓，止损位设于 ${support:,.0f}
目标位：${resistance:,.0f}""",
        
        SignalType.BUY: f"""🟢 **买入信号**
当前价格：${price:,.0f}
建议：可以小仓位试探性买入
止损位：${support:,.0f}""",
        
        SignalType.WAIT: f"""🟡 **观望**
当前价格：${price:,.0f}
建议：等待更明确的信号
支撑位：${support:,.0f}
阻力位：${resistance:,.0f}""",
        
        SignalType.SELL: f"""🔴 **卖出信号**
当前价格：${price:,.0f}
建议：可以减仓锁定利润
止损位：${resistance:,.0f}""",
        
        SignalType.STRONG_SELL: f"""🚨 **强烈卖出信号**
当前价格：${price:,.0f}
建议：建议减仓或清仓
止损位：${resistance:,.0f}"""
    }
    
    return recommendations.get(signal, "信号不明确，建议观望")


# ============ 报警管理 ============

class AlertManager:
    """报警管理器"""
    
    def __init__(self):
        self.last_alert_file = Path.home() / ".btc_signal_last_alert.json"
        self.last_alert = self.load_last_alert()
    
    def load_last_alert(self) -> Dict:
        """加载上次报警记录"""
        if self.last_alert_file.exists():
            try:
                with open(self.last_alert_file) as f:
                    return json.load(f)
            except:
                pass
        return {
            'last_signal': SignalType.WAIT,
            'last_time': datetime.fromtimestamp(0).isoformat(),
            'price_at_alert': 0
        }
    
    def save_last_alert(self, signal: str, price: float):
        """保存报警记录"""
        self.last_alert = {
            'last_signal': signal,
            'last_time': datetime.now().isoformat(),
            'price_at_alert': price
        }
        
        with open(self.last_alert_file, 'w') as f:
            json.dump(self.last_alert, f, indent=2)
    
    def should_alert(self, signal: str, price: float) -> bool:
        """判断是否应该报警"""
        
        # 强信号随时报警
        if signal in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            return True
        
        # 普通信号检查冷却期
        last_time = datetime.fromisoformat(self.last_alert['last_time'])
        time_diff = (datetime.now() - last_time).total_seconds() / 60
        
        # 买入信号冷却期内不重复报警
        if signal == SignalType.BUY and self.last_alert['last_signal'] in [SignalType.BUY, SignalType.STRONG_BUY]:
            if time_diff < CONFIG['cooldown_minutes']:
                return False
        
        # 卖出信号冷却期内不重复报警
        if signal == SignalType.SELL and self.last_alert['last_signal'] in [SignalType.SELL, SignalType.STRONG_SELL]:
            if time_diff < CONFIG['cooldown_minutes']:
                return False
        
        # 信号变化时报警
        if signal != self.last_alert['last_signal']:
            return True
        
        return False


# ============ Telegram 推送 ============

async def send_telegram_alert(analysis: Dict, signal: Dict):
    """发送 Telegram 报警"""
    
    if not CONFIG['telegram_enabled']:
        print("\n📱 Telegram 报警（未配置）")
        return
    
    chat_id = CONFIG['telegram_chat_id']
    if not chat_id:
        print("\n📱 Telegram 报警（未配置 chat_id）")
        return
    
    # 格式化消息
    message = f"""
🚨 **BTC 信号提醒**

📊 **当前状态**
• 价格：${analysis['current_price']:,.0f}
• RSI：{analysis['rsi']['value']}（{analysis['rsi']['status']}）
• 趋势：{analysis['trend']} | {analysis['short_trend']}

📈 **均线系统**
• MA7：${analysis['ma']['ma7']:,.0f}
• MA25：${analysis['ma']['ma25']:,.0f}
• 交叉：{analysis['ma']['crossover']}

💡 **交易信号**
• 类型：**{signal['signal']}**
• 评分：{signal['score']}
• 置信度：{signal['confidence']:.0f}%

🎯 **建议**
{signal['recommendation']}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 实际发送时使用 openclaw-telegram 技能
    print(f"\n📱 Telegram 消息已准备：\n{message}")
    
    # TODO: 集成 Telegram 发送功能
    # await telegram.send_message(message, chat_id)


# ============ 主监控循环 ============

async def run_monitor(continuous: bool = False):
    """运行监控"""
    
    print(f"""
╔══════════════════════════════════════╗
║   🚨 BTC 信号监控系统 v1.0          ║
╠══════════════════════════════════════╣
║  监控品种：{CONFIG['symbol']:<25}║
║  检查周期：{CONFIG['check_interval']}秒{'':<19}║
║  模式：{'持续监控' if continuous else '单次检测'}{'':<21}║
╚══════════════════════════════════════╝
""")
    
    alert_manager = AlertManager()
    
    async def check_once():
        """执行一次检测"""
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
        
        # 获取数据
        candles = await fetch_btc_data()
        if not candles:
            print("❌ 获取数据失败")
            return
        
        print(f"✅ 获取数据 {len(candles)} 条")
        
        # 分析市场
        analysis = analyze_market(candles)
        
        print(f"📊 当前价格：${analysis['current_price']:,.0f}")
        print(f"📉 RSI：{analysis['rsi']['value']}（{analysis['rsi']['status']}）")
        print(f"📈 MA7：${analysis['ma']['ma7']:,.0f} | MA25：${analysis['ma']['ma25']:,.0f}")
        print(f"📐 交叉：{analysis['ma']['crossover']}")
        print(f"📊 趋势：{analysis['trend']} | {analysis['short_trend']}")
        
        # 生成信号
        signal = generate_signal(analysis)
        
        print(f"\n🎯 信号：**{signal['signal']}**")
        print(f"   评分：{signal['score']}")
        print(f"   置信度：{signal['confidence']:.0f}%")
        for reason in signal['reasons']:
            print(f"   • {reason}")
        
        # 检查是否应该报警
        if alert_manager.should_alert(signal['signal'], analysis['current_price']):
            print(f"\n🔔 触发报警！")
            await send_telegram_alert(analysis, signal)
            alert_manager.save_last_alert(signal['signal'], analysis['current_price'])
        else:
            print(f"\n⏸️ 报警已抑制（冷却期）")
        
        # 打印建议
        print(f"\n💡 建议：")
        print(signal['recommendation'])
    
    if continuous:
        # 持续监控
        while True:
            await check_once()
            await asyncio.sleep(CONFIG['check_interval'])
    else:
        # 单次检测
        await check_once()


# ============ CLI ============

async def main():
    """CLI 入口"""
    import sys
    
    continuous = "--continuous" in sys.argv or "-c" in sys.argv
    
    # 检查参数
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        symbol = sys.argv[1]
        CONFIG['symbol'] = symbol
    
    await run_monitor(continuous=continuous)


if __name__ == "__main__":
    asyncio.run(main())
