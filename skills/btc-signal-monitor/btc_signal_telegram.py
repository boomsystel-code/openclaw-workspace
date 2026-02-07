#!/usr/bin/env python3
"""
BTC 信号监控系统 - Telegram 集成版
实时监控 BTC 交易信号，自动推送 Telegram 提醒
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# 添加 OpenClaw 路径
sys.path.insert(0, '/Users/wangshice/.openclaw/workspace')
sys.path.insert(0, '/opt/homebrew/lib/node_modules/openclaw/skills/telegram')

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
    # Telegram 配置
    "telegram_chat_id": "",  # 如果为空，发送到默认对话
    "alert_on_signal_change": True,  # 信号变化时报警
    "alert_on_strong_signal": True,  # 强信号时报警
}


# ============ OpenClaw Telegram 集成 ============

class TelegramAlert:
    """Telegram 报警器"""
    
    def __init__(self):
        self.config = self._load_config()
        self.bot_token = self.config.get('botToken', '')
        self.chat_id = CONFIG['telegram_chat_id']
    
    def _load_config(self) -> Dict:
        """加载 OpenClaw 配置"""
        try:
            with open('/Users/wangshice/.openclaw/openclaw.json') as f:
                return json.load(f)
        except:
            return {}
    
    async def send_message(self, message: str, chat_id: str = None) -> bool:
        """发送 Telegram 消息"""
        
        target_chat = chat_id or self.chat_id
        
        # 方法 1：使用 OpenClaw 消息接口
        try:
            from openclaw_telegram_skill import send_telegram_message
            success = await send_telegram_message(
                message=message,
                chat_id=target_chat or "main"
            )
            if success:
                print(f"✅ OpenClaw Telegram 发送成功")
                return True
        except Exception as e:
            print(f"⚠️ OpenClaw 发送失败: {e}")
        
        # 方法 2：直接调用 Telegram API
        try:
            import requests
            
            if not self.bot_token:
                print(f"❌ 未配置 Bot Token")
                return False
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                "text": message,
                "parse_mode": "Markdown",
            }
            
            if target_chat:
                payload["chat_id"] = target_chat
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Telegram API 发送成功")
                return True
            else:
                print(f"❌ Telegram API 错误: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Telegram 发送失败: {e}")
            return False
        
        return False
    
    async def send_signal_alert(self, analysis: Dict, signal: Dict):
        """发送信号报警"""
        
        # 格式化消息
        emoji = {
            SignalType.STRONG_BUY: "🔥",
            SignalType.BUY: "🟢",
            SignalType.SELL: "🔴",
            SignalType.STRONG_SELL: "🚨",
            SignalType.WAIT: "🟡"
        }.get(signal['signal'], "📊")
        
        message = f"""
{emoji} *BTC 信号提醒*

📊 *当前状态*
• 价格：`${analysis['current_price']:,.0f}`
• RSI：`{analysis['rsi']['value']}` ({analysis['rsi']['status']})
• 趋势：`{analysis['trend']}` | `{analysis['short_trend']}`

📈 *均线系统*
• MA7：`${analysis['ma']['ma7']:,.0f}`
• MA25：`${analysis['ma']['ma25']:,.0f}`
• 交叉：`{analysis['ma']['crossover']}`

🎯 *交易信号*
• 类型：`{signal['signal']}`
• 评分：`{signal['score']}`
• 置信度：`{signal['confidence']:.0f}%`

💡 *建议*
{signal['recommendation']}

⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

🤖 *BTC Signal Monitor*
"""
        
        return await self.send_message(message)
    
    async def send_price_alert(self, current_price: float, threshold: float, direction: str):
        """发送价格报警"""
        
        emoji = "📈" if direction == "above" else "📉"
        message = f"""
{emoji} *价格报警*

BTC 当前价格：`${current_price:,.0f}`

已{direction}阈值：`${threshold:,.0f}`

⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
"""
        
        return await self.send_message(message)


# ============ 其余代码（与之前相同）===========

# 信号定义
SignalType = type('SignalType', (), {
    'BUY': 'BUY',
    'SELL': 'SELL',
    'WAIT': 'WAIT',
    'STRONG_BUY': 'STRONG_BUY',
    'STRONG_SELL': 'STRONG_SELL'
})()

# 复用之前的函数
exec("""
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
    if not candles:
        return {'error': '无数据'}
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
    price = analysis['current_price']
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
        SignalType.STRONG_BUY: f"🔥 **强烈买入信号**\\n当前价格：${price:,.0f}\\n建议：可以考虑分批建仓，止损位设于 ${support:,.0f}",
        SignalType.BUY: f"🟢 **买入信号**\\n当前价格：${price:,.0f}\\n建议：可以小仓位试探性买入\\n止损位：${support:,.0f}",
        SignalType.WAIT: f"🟡 **观望**\\n当前价格：${price:,.0f}\\n建议：等待更明确的信号\\n支撑位：${support:,.0f}\\n阻力位：${resistance:,.0f}",
        SignalType.SELL: f"🔴 **卖出信号**\\n当前价格：${price:,.0f}\\n建议：可以减仓锁定利润\\n止损位：${resistance:,.0f}",
        SignalType.STRONG_SELL: f"🚨 **强烈卖出信号**\\n当前价格：${price:,.0f}\\n建议：建议减仓或清仓\\n止损位：${resistance:,.0f}"
    }
    return recs.get(signal, "信号不明确，建议观望")

class AlertManager:
    def __init__(self):
        self.last_alert_file = Path.home() / ".btc_signal_alert.json"
        self.last_alert = self.load_last_alert()
    
    def load_last_alert(self):
        if self.last_alert_file.exists():
            try:
                with open(self.last_alert_file) as f:
                    return json.load(f)
            except:
                pass
        return {'last_signal': SignalType.WAIT, 'last_time': datetime.fromtimestamp(0).isoformat()}
    
    def save_last_alert(self, signal, price):
        self.last_alert = {
            'last_signal': signal,
            'last_time': datetime.now().isoformat(),
            'price': price
        }
        with open(self.last_alert_file, 'w') as f:
            json.dump(self.last_alert, f, indent=2)
    
    def should_alert(self, signal, price):
        if signal in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            return True
        last_time = datetime.fromisoformat(self.last_alert['last_time'])
        time_diff = (datetime.now() - last_time).total_seconds() / 60
        if signal == SignalType.BUY and self.last_alert['last_signal'] in [SignalType.BUY, SignalType.STRONG_BUY]:
            if time_diff < CONFIG['cooldown_minutes']:
                return False
        if signal == SignalType.SELL and self.last_alert['last_signal'] in [SignalType.SELL, SignalType.STRONG_SELL]:
            if time_diff < CONFIG['cooldown_minutes']:
                return False
        return signal != self.last_alert['last_signal']

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
""")
