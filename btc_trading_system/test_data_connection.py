#!/usr/bin/env python3
"""
BTC数据获取测试和诊断工具
"""

import asyncio
import ccxt
import json
import os
from datetime import datetime

# 配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CACHE_FILE = os.path.join(BASE_DIR, 'data/cache/market_cache.json')


def test_binance_connection():
    """测试Binance连接"""
    print("🔍 测试Binance API连接...")
    
    try:
        # 尝试多种方式获取数据
        methods = [
            ("ccxt.binance()", lambda: ccxt.binance()),
            ("ccxt.binance({'enableRateLimit': True})", 
             lambda: ccxt.binance({'enableRateLimit': True})),
        ]
        
        for name, method in methods:
            try:
                print(f"  尝试: {name}")
                exchange = method()
                
                # 获取Ticker数据
                ticker = exchange.fetch_ticker('BTC/USDT')
                print(f"  ✅ 成功获取Ticker数据")
                print(f"     价格: ${ticker['last']:,.2f}")
                print(f"     24h涨跌: ${ticker['change']:,.2f}")
                
                # 获取K线数据（1天）
                ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=10)
                print(f"  ✅ 成功获取K线数据")
                print(f"     获取到 {len(ohlcv)} 条K线")
                
                return exchange, ticker, ohlcv
                
            except Exception as e:
                print(f"  ❌ {name} 失败: {e}")
                continue
        
        print("❌ 所有方法都失败")
        return None, None, None
        
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return None, None, None


def test_coinbase_connection():
    """测试Coinbase连接"""
    print("\n🔍 测试Coinbase API连接...")
    
    try:
        exchange = ccxt.coinbase()
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"  ✅ Coinbase成功")
        print(f"     价格: ${ticker['last']:,.2f}")
        return exchange, ticker
    except Exception as e:
        print(f"  ❌ Coinbase失败: {e}")
        return None, None


def save_market_data(ticker, ohlcv=None):
    """保存市场数据到缓存"""
    print("\n💾 保存市场数据...")
    
    try:
        data = {
            "symbol": "BTCUSDT",
            "current_price": ticker['last'],
            "high_24h": ticker['high'],
            "low_24h": ticker['low'],
            "volume_24h": ticker['baseVolume'],
            "price_change_24h": ticker['change'],
            "price_change_percent_24h": ticker['percentage'],
            "source": "binance",
            "timestamp": datetime.now().isoformat()
        }
        
        if ohlcv:
            data['ohlcv'] = {
                'count': len(ohlcv),
                'last_close': ohlcv[-1][4] if ohlcv else None
            }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(DATA_CACHE_FILE), exist_ok=True)
        
        with open(DATA_CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✅ 数据已保存到: {DATA_CACHE_FILE}")
        return True
        
    except Exception as e:
        print(f"  ❌ 保存失败: {e}")
        return False


def main():
    """主测试流程"""
    print("=" * 60)
    print("BTC交易系统 - 数据获取诊断工具")
    print("=" * 60)
    print()
    
    # 1. 测试Binance
    exchange, ticker, ohlcv = test_binance_connection()
    
    if exchange is None:
        # 2. 尝试Coinbase
        exchange, ticker = test_coinbase_connection()
    
    if exchange:
        # 3. 保存数据
        save_market_data(ticker, ohlcv)
        
        print("\n" + "=" * 60)
        print("✅ 诊断完成 - 数据获取正常")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 诊断完成 - 需要修复数据获取")
        print("=" * 60)
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 检查API密钥配置")
        print("3. 尝试使用其他交易所数据")
        print("4. 添加离线数据支持")


if __name__ == "__main__":
    main()
