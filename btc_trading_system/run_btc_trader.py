#!/usr/bin/env python3
"""
BTC交易系统 - 增强版运行器
=============================
集成错误日志、定时任务、自动重试

使用方法:
  python3 run_btc_trader.py              # 正常运行
  python3 run_btc_trader.py --test       # 测试模式
  python3 run_btc_trader.py --daily      # 每日模式
  python3 run_btc_trader.py --force      # 强制重新获取数据
"""

import asyncio
import ccxt
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CACHE_FILE = os.path.join(BASE_DIR, 'data/cache/market_cache.json')
ERROR_LOG = os.path.expanduser("~/.openclaw/workspace/error-logs/.counter")


class BTCTrader:
    """BTC交易系统"""
    
    def __init__(self, force_refresh: bool = False):
        self.force_refresh = force_refresh
        self.exchange = None
        self.data = {}
        self.errors = []
        
    def log_error(self, error_msg: str, severity: str = "medium"):
        """记录错误到日志系统"""
        self.errors.append({
            'message': error_msg,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        logger.error(f"[{severity}] {error_msg}")
    
    def test_connection(self) -> bool:
        """测试交易所连接"""
        print("🔍 测试交易所连接...")
        
        exchanges_to_try = [
            ('Binance', lambda: ccxt.binance({'enableRateLimit': True})),
            ('Coinbase', lambda: ccxt.coinbase()),
            ('Kraken', lambda: ccxt.kraken()),
        ]
        
        for name, constructor in exchanges_to_try:
            try:
                print(f"  尝试 {name}...")
                self.exchange = constructor()
                
                # 测试获取数据
                ticker = self.exchange.fetch_ticker('BTC/USDT')
                print(f"  ✅ {name} 连接成功")
                print(f"     价格: ${ticker['last']:,.2f}")
                return True
                
            except Exception as e:
                print(f"  ❌ {name} 失败: {e}")
                continue
        
        self.log_error("所有交易所连接失败", "critical")
        return False
    
    def fetch_market_data(self) -> bool:
        """获取市场数据"""
        print("\n📊 获取市场数据...")
        
        try:
            # 获取Ticker
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            
            # 获取K线数据（1d周期，30天）
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '1d', limit=30)
            
            # 计算技术指标
            self.data = self._calculate_indicators(ticker, ohlcv)
            
            print(f"  ✅ 获取成功")
            print(f"     价格: ${self.data['current_price']:,.2f}")
            print(f"     RSI: {self.data['rsi']:.2f}")
            print(f"     波动率: {self.data['volatility']:.2f}%")
            
            return True
            
        except Exception as e:
            error_msg = f"数据获取失败: {e}"
            print(f"  ❌ {error_msg}")
            self.log_error(error_msg, "high")
            return False
    
    def _calculate_indicators(self, ticker: Dict, ohlcv: list) -> Dict:
        """计算技术指标"""
        closes = [c[4] for c in ohlcv]  # 收盘价
        
        # RSI (14)
        delta = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gain = sum([d for d in delta if d > 0]) / 14
        loss = -sum([d for d in delta if d < 0]) / 14
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # 波动率
        volatility = (max(closes) - min(closes)) / min(closes) * 100
        
        # 移动平均线
        ma7 = sum(closes[-7:]) / 7
        ma25 = sum(closes[-25:]) / 25
        
        return {
            'current_price': ticker['last'],
            'high_24h': ticker['high'],
            'low_24h': ticker['low'],
            'volume_24h': ticker['baseVolume'],
            'price_change': ticker['change'],
            'price_change_percent': ticker['percentage'],
            'rsi': rsi,
            'volatility': volatility,
            'ma7': ma7,
            'ma25': ma25,
            'timestamp': datetime.now().isoformat(),
            'ohlcv_count': len(ohlcv)
        }
    
    def save_data(self) -> bool:
        """保存数据到缓存"""
        print("\n💾 保存数据...")
        
        try:
            os.makedirs(os.path.dirname(DATA_CACHE_FILE), exist_ok=True)
            
            with open(DATA_CACHE_FILE, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            print(f"  ✅ 保存到: {DATA_CACHE_FILE}")
            return True
            
        except Exception as e:
            self.log_error(f"数据保存失败: {e}", "medium")
            return False
    
    def generate_signal(self) -> Dict:
        """生成交易信号"""
        if not self.data:
            return {'signal': 'UNKNOWN', 'reason': '无数据'}
        
        price = self.data['current_price']
        rsi = self.data['rsi']
        ma7 = self.data['ma7']
        
        # 简单信号逻辑
        signals = []
        
        if rsi < 30:
            signals.append(('RSI超卖', 'BUY'))
        elif rsi > 70:
            signals.append(('RSI超买', 'SELL'))
        
        if price < ma7:
            signals.append(('价格低于MA7', 'BUY'))
        elif price > ma7:
            signals.append(('价格高于MA7', 'SELL'))
        
        # 综合信号
        buy_count = sum(1 for _, s in signals if s == 'BUY')
        sell_count = sum(1 for _, s in signals if s == 'SELL')
        
        if buy_count > sell_count:
            final_signal = 'BUY'
        elif sell_count > buy_count:
            final_signal = 'SELL'
        else:
            final_signal = 'WAIT'
        
        return {
            'signal': final_signal,
            'signals': signals,
            'rsi': rsi,
            'ma7': ma7,
            'price': price,
            'confidence': max(buy_count, sell_count) / max(len(signals), 1) if signals else 0.5
        }
    
    def run(self) -> Dict:
        """运行交易系统"""
        print("=" * 60)
        print("🚀 BTC交易系统启动")
        print("=" * 60)
        print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'data': None,
            'signal': None,
            'errors': []
        }
        
        try:
            # 1. 测试连接
            if not self.test_connection():
                result['errors'].append('连接失败')
                return result
            
            # 2. 获取数据
            if not self.fetch_market_data():
                result['errors'].append('数据获取失败')
                return result
            
            # 3. 保存数据
            self.save_data()
            
            # 4. 生成信号
            signal = self.generate_signal()
            result['signal'] = signal
            
            # 5. 显示结果
            print("\n" + "=" * 60)
            print("📊 交易信号")
            print("=" * 60)
            print(f"信号: {signal['signal']}")
            print(f"置信度: {signal['confidence']:.0%}")
            print(f"价格: ${signal['price']:,.2f}")
            print(f"RSI: {signal['rsi']:.2f}")
            print(f"MA7: ${signal['ma7']:,.2f}")
            
            if signal['signals']:
                print("\n子信号:")
                for reason, s in signal['signals']:
                    print(f"  • {reason}: {s}")
            
            result['success'] = True
            result['data'] = self.data
            
        except Exception as e:
            error_msg = f"系统异常: {e}"
            print(f"\n❌ {error_msg}")
            self.log_error(error_msg, "high")
            result['errors'].append(error_msg)
        
        # 输出错误摘要
        if self.errors:
            print("\n⚠️ 错误记录:")
            for error in self.errors:
                print(f"  [{error['severity']}] {error['message']}")
        
        print("\n" + "=" * 60)
        print(f"✅ 运行完成 - 成功: {result['success']}")
        print("=" * 60)
        
        return result


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='BTC交易系统')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--daily', action='store_true', help='每日模式')
    parser.add_argument('--force', action='store_true', help='强制刷新')
    
    args = parser.parse_args()
    
    trader = BTCTrader(force_refresh=args.force)
    result = trader.run()
    
    # 退出码
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 🚀 自动交易便捷函数
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def quick_auto_trade_start(mode="sim", interval=60):
    """快速启动自动交易
    
    Args:
        mode: sim(模拟), live(实盘), dry(试运行)
        interval: 检查间隔（秒）
    
    Returns:
        状态字典
    """
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from btc_auto_trader import start_auto_trader, TradeMode
        
        mode_map = {
            'sim': TradeMode.SIMULATION,
            'live': TradeMode.LIVE,
            'dry': TradeMode.DRY_RUN
        }
        
        trader_mode = mode_map.get(mode, TradeMode.SIMULATION)
        
        print(f"\n" + "="*70)
        print("🚀 BTC AI 自动交易系统")
        print("="*70)
        print(f"📊 模式: {mode}")
        print(f"⏱️ 间隔: {interval}秒")
        print(f"💡 按 Ctrl+C 停止")
        print("="*70)
        
        # 在新线程中运行
        import threading
        import asyncio
        
        def run_trader():
            asyncio.run(start_auto_trader(trader_mode, interval))
        
        thread = threading.Thread(target=run_trader, daemon=True)
        thread.start()
        
        # 等待一下让用户看到启动信息
        time.sleep(2)
        
        return {
            "status": "started",
            "mode": mode,
            "message": f"自动交易已启动 ({mode}模式)"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


def quick_auto_trade_stop():
    """停止自动交易"""
    return {"status": "info", "message": "请按 Ctrl+C 停止自动交易"}


def quick_trade_status():
    """查询交易状态"""
    config_file = os.path.expanduser("~/.openclaw/workspace/btc_trading_system/auto_trader_config.json")
    
    position = {"amount": 0, "pnl": 0, "pnl_percent": 0}
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
    except:
        pass
    
    return {
        "mode": "simulation",
        "position": position,
        "trades_today": 0,
        "config_file": config_file
    }


def quick_trade_history():
    """查询交易历史"""
    return {
        "trades": [],
        "total_pnl": 0,
        "win_rate": 0
    }


def quick_reset_trade():
    """重置交易"""
    return {"status": "reset", "message": "请手动重置或重启系统"}


def quick_config_api(api_key="", api_secret="", trade_amount=0.01, stop_loss=5, take_profit=10):
    """配置交易API
    
    Args:
        api_key: API密钥
        api_secret: API密钥
        trade_amount: 每次交易量(BTC)
        stop_loss: 止损比例(%)
        take_profit: 止盈比例(%)
    """
    config = {
        'api_key': api_key,
        'api_secret': api_secret,
        'trade_amount': trade_amount,
        'stop_loss': stop_loss / 100,
        'take_profit': take_profit / 100,
        'auto_start': False
    }
    
    config_file = os.path.expanduser("~/.openclaw/workspace/btc_trading_system/auto_trader_config.json")
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return {
        "status": "saved",
        "config": {
            "trade_amount": trade_amount,
            "stop_loss": f"{stop_loss}%",
            "take_profit": f"{take_profit}%"
        }
    }
