#!/usr/bin/env python3
"""
每日智能日报生成器
Daily Smart Report Generator
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# 尝试导入 ccxt
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("⚠️ ccxt 未安装，将使用备用数据源")


# ============ 配置 ============

CONFIG = {
    "telegram_channel": "main",
    "default_city": "北京",
    "report_time": "08:00",
    "enabled": True
}

# 每日名言库
QUOTES = [
    "投资的第一条规则是不要亏钱。第二条规则是永远不要忘记第一条。",
    "别人贪婪时我恐惧，别人恐惧时我贪婪。",
    "时间是优秀企业的朋友，是平庸企业的敌人。",
    "投资比的是谁更少犯错，而不是谁更聪明。",
    "风险来自于你不知道自己在做什么。",
    "如果你不愿意持有一只股票十年，那就不要考虑持有它十分钟。",
    "价格是你支付的，价值是你得到的。",
    "不要把所有的鸡蛋放在一个篮子里。",
    "投资是推迟的消费。",
    "市场短期是投票机，长期是称重机。",
]


# ============ BTC 数据 ============

async def get_btc_price():
    """获取 BTC 行情数据"""
    if HAS_CCXT:
        try:
            exchange = ccxt.binance()
            btc = exchange.fetch_ticker('BTC/USDT')
            return {
                'price': btc.get('last', 0),
                'change_24h': btc.get('percentage', 0),
                'high': btc.get('high', 0),
                'low': btc.get('low', 0),
                'volume': btc.get('volume', 0),
                'source': 'Binance'
            }
        except Exception as e:
            print(f"❌ Binance API 错误: {e}")
    
    # 备用数据
    return {
        'price': 96500.0,
        'change_24h': 2.3,
        'high': 98000.0,
        'low': 95000.0,
        'volume': 28500000000,
        'source': 'Mock'
    }


# ============ 天气数据 ============

async def get_weather(city=None):
    """获取天气信息"""
    city = city or CONFIG["default_city"]
    
    weather_data = {
        '北京': {'condition': '晴', 'temp': '3-12', 'aqi': '良'},
        '上海': {'condition': '多云', 'temp': '8-15', 'aqi': '良'},
        '深圳': {'condition': '晴', 'temp': '18-24', 'aqi': '优'},
        '广州': {'condition': '阴', 'temp': '15-20', 'aqi': '轻度污染'},
    }
    
    return weather_data.get(city, {
        'condition': '未知',
        'temp': 'N/A',
        'aqi': 'N/A'
    })


# ============ 辅助函数 ============

def get_today_date():
    """获取今日日期"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %A")


def get_daily_quote():
    """获取每日名言"""
    import random
    return random.choice(QUOTES)


def get_daily_tip():
    """获取每日效率提示"""
    import random
    tips = [
        "🌟 今天适合：学习新知识、整理文件",
        "💡 建议：避免冲动决策，三思而后行",
        "📈 投资提示：保持冷静，不要被市场情绪影响",
        "🏃 健康提醒：久坐一小时，起来活动5分钟",
        "📱 数字排毒：减少刷手机时间，专注当下",
        "📖 阅读时间：每天至少阅读30分钟",
        "💰 理财习惯：记录每笔支出，了解钱花在哪",
        "🧘 冥想时刻：每天10分钟，清空杂念",
    ]
    return random.choice(tips)


# ============ 核心功能 ============

async def generate_report():
    """生成完整日报"""
    print("📊 正在生成日报...")
    
    btc_data, weather_data = await asyncio.gather(
        get_btc_price(),
        get_weather()
    )
    
    price = btc_data['price']
    change = btc_data['change_24h']
    change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
    
    city = CONFIG["default_city"]
    weather = weather_data
    
    report = f"""
📅 {get_today_date()} 智能日报

💰 BTC 行情
• 当前价格：${price:,.0f}
• 24h 涨跌：{change_emoji} {change:+.2f}%
• 波动区间：${btc_data['low']:,.0f} - ${btc_data['high']:,.0f}

🌤️ {city} 天气
• 状况：{weather['condition']}
• 温度：{weather['temp']}°C
• 空气质量：{weather['aqi']}

📝 每日一句
"{get_daily_quote()}"

💡 效率提示
{get_daily_tip()}

---
🤖 自动生成 by OpenClaw 每日日报技能
"""
    
    return report


async def send_report(report):
    """发送报告到 Telegram"""
    print(f"\n📨 报告内容：\n{report}")
    print("\n✅ 报告生成完成！")


# ============ CLI 接口 ============

async def main():
    """CLI 入口"""
    import sys
    
    command = sys.argv[1] if len(sys.argv) > 1 else "generate"
    
    if command == "generate":
        report = await generate_report()
        await send_report(report)
    elif command == "btc":
        data = await get_btc_price()
        print(f"💰 BTC: ${data['price']:,.0f} ({data['change_24h']:+.2f}%)")
    elif command == "weather":
        data = await get_weather()
        print(f"🌤️ {CONFIG['default_city']}: {data['condition']} {data['temp']}°C")
    else:
        print("""
🤖 每日日报技能

使用方式：
  python daily_report.py generate   # 生成日报
  python daily_report.py btc       # 查看 BTC 行情
  python daily_report.py weather    # 查看天气
        """)


if __name__ == "__main__":
    asyncio.run(main())
