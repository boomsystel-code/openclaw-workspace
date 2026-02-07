#!/usr/bin/env python3
"""
BTC 信号监控 - Telegram 测试工具
"""

import asyncio
import json
import os
from datetime import datetime

# 加载配置
with open('/Users/wangshice/.openclaw/openclaw.json') as f:
    openclaw_config = json.load(f)

TELEGRAM_CONFIG = openclaw_config.get('channels', {}).get('telegram', {})
BOT_TOKEN = TELEGRAM_CONFIG.get('botToken', '')
CHAT_ID = ""  # 空表示发送到当前对话

async def send_telegram(message: str) -> bool:
    """发送 Telegram 消息"""
    
    if not BOT_TOKEN:
        print("❌ 未配置 Bot Token")
        return False
    
    try:
        import requests
        
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "text": message,
            "parse_mode": "Markdown"
        }
        
        if CHAT_ID:
            payload["chat_id"] = CHAT_ID
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ Telegram 发送成功！")
            return True
        else:
            print(f"❌ Telegram 错误: {response.status_code}")
            print(response.text[:200])
            return False
            
    except Exception as e:
        print(f"❌ 发送失败: {e}")
        return False


async def test_telegram():
    """测试 Telegram 发送"""
    
    print("""
╔══════════════════════════════════════╗
║   📱 Telegram 报警测试              ║
╚══════════════════════════════════════╝
""")
    
    print(f"📱 Bot Token: {BOT_TOKEN[:15]}...")
    print(f"💬 Chat ID: {CHAT_ID or '默认对话'}")
    print()
    
    # 发送测试消息
    test_message = f"""
🧪 *BTC Signal Monitor - 测试消息*

✅ Telegram 报警功能已配置成功！

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    print("📤 发送测试消息...")
    success = await send_telegram(test_message)
    
    if success:
        print("\n🎉 Telegram 配置正确！")
        print("✅ 现在可以接收 BTC 信号提醒")
    else:
        print("\n❌ Telegram 配置有问题")
        print("💡 请检查 Bot Token 是否正确")
    
    return success


async def send_signal_demo():
    """发送示例信号消息"""
    
    demo_message = f"""
🚨 *BTC 信号提醒*

📊 *当前状态*
• 价格：`$70,031`
• RSI：`42.5` (NEUTRAL)
• 趋势：`UPTREND` | `BULLISH`

📈 *均线系统*
• MA7：`$69,800`
• MA25：`$71,200`
• 交叉：`GOLDEN`

🎯 *交易信号*
• 类型：`BUY`
• 评分：`2`
• 置信度：`70%`

💡 *建议*
🟢 **买入信号**
当前价格：$70,031
建议：可以小仓位试探性买入
止损位：$68,500

⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

🤖 *BTC Signal Monitor*
"""
    
    print("\n📤 发送示例信号...")
    return await send_telegram(demo_message)


async def main():
    """主函数"""
    
    mode = "test"
    
    if len(os.sys.argv) > 1:
        mode = os.sys.argv[1]
    
    if mode == "demo":
        await send_signal_demo()
    else:
        await test_telegram()


if __name__ == "__main__":
    asyncio.run(main())
