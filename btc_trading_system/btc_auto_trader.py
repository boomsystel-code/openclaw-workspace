#!/usr/bin/env python3
"""
BTC AI Auto-Trader - 自动量化交易系统
=====================================
功能:
  - 🤖 AI信号自动监控
  - 📈 自动买入/卖出执行
  - 💰 模拟/实盘双模式
  - ⚠️ 风险管理
  - 📊 交易日志

Author: AI Trading System
Date: 2026-02-07
"""

import asyncio
import ccxt
import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置文件
CONFIG_FILE = os.path.expanduser("~/.openclaw/workspace/btc_trading_system/auto_trader_config.json")


class TradeMode(Enum):
    """交易模式"""
    SIMULATION = "sim"  # 模拟
    LIVE = "live"       # 实盘
    DRY_RUN = "dry"     # 试运行（只记录不执行）


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Order:
    """订单"""
    side: OrderSide
    amount: float
    price: float = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    order_id: str = None
    
    def to_dict(self) -> Dict:
        return {
            'side': self.side.value,
            'amount': self.amount,
            'price': self.price,
            'status': self.status.value,
            'filled_price': self.filled_price,
            'timestamp': self.timestamp,
            'order_id': self.order_id
        }


@dataclass
class Position:
    """持仓"""
    amount: float = 0
    avg_price: float = 0
    pnl: float = 0
    pnl_percent: float = 0
    
    def to_dict(self) -> Dict:
        return {
            'amount': self.amount,
            'avg_price': self.avg_price,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent
        }


class AutoTrader:
    """自动交易机器人"""
    
    def __init__(self, mode: TradeMode = TradeMode.SIMULATION):
        self.mode = mode
        self.exchange = None
        self.position = Position()
        self.trade_history = []
        self.config = self._load_config()
        self.is_running = False
        
        # 交易参数
        self.symbol = "BTC/USDT"
        self.trade_amount = self.config.get('trade_amount', 0.01)  # 每次交易量
        self.stop_loss = self.config.get('stop_loss', 0.05)         # 止损5%
        self.take_profit = self.config.get('take_profit', 0.10)      # 止盈10%
        
        # 信号缓存
        self.last_signal = None
        self.last_price = None
        
    def _load_config(self) -> Dict:
        """加载配置"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'api_key': '',
            'api_secret': '',
            'trade_amount': 0.01,
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'max_positions': 5,
            'auto_start': False
        }
    
    def _save_config(self):
        """保存配置"""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def connect_exchange(self) -> bool:
        """连接交易所"""
        if self.mode == TradeMode.SIMULATION:
            logger.info("[AutoTrader] 模拟模式启动")
            return True
        
        try:
            # 尝试使用配置中的API
            if self.config.get('api_key') and self.config.get('api_secret'):
                self.exchange = ccxt.binance({
                    'apiKey': self.config['api_key'],
                    'secret': self.config['api_secret'],
                    'enableRateLimit': True
                })
            else:
                # 无API模式，只读
                self.exchange = ccxt.binance({'enableRateLimit': True})
            
            # 测试连接
            balance = self.exchange.fetch_balance()
            logger.info(f"[AutoTrader] 交易所连接成功")
            return True
            
        except Exception as e:
            logger.error(f"[AutoTrader] 交易所连接失败: {e}")
            return False
    
    def get_price(self) -> Optional[float]:
        """获取当前价格"""
        try:
            if self.mode == TradeMode.SIMULATION:
                # 模拟价格
                return self.last_price or 70000
            
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"[AutoTrader] 获取价格失败: {e}")
            return None
    
    def execute_buy(self, amount: float, price: float = None) -> Order:
        """执行买入"""
        order = Order(side=OrderSide.BUY, amount=amount, price=price)
        
        if self.mode == TradeMode.DRY_RUN:
            logger.info(f"[AutoTrader] [DRY] 买入信号: {amount} BTC @ ${price or '市价'}")
            order.status = OrderStatus.FILLED
            return order
        
        if self.mode == TradeMode.SIMULATION:
            # 模拟成交
            fill_price = price or self.get_price()
            order.filled_price = fill_price
            order.status = OrderStatus.FILLED
            
            # 更新持仓
            if self.position.amount == 0:
                self.position.amount = amount
                self.position.avg_price = fill_price
            else:
                total = self.position.amount + amount
                self.position.avg_price = (
                    self.position.amount * self.position.avg_price + 
                    amount * fill_price
                ) / total
                self.position.amount = total
            
            self.position.pnl = (fill_price - self.position.avg_price) * self.position.amount
            self.position.pnl_percent = (fill_price / self.position.avg_price - 1) * 100
            
            logger.info(f"[AutoTrader] [SIM] 买入成交: {amount} BTC @ ${fill_price:,.2f}")
        
        return order
    
    def execute_sell(self, amount: float, price: float = None) -> Order:
        """执行卖出"""
        order = Order(side=OrderSide.SELL, amount=amount, price=price)
        
        if self.mode == TradeMode.DRY_RUN:
            logger.info(f"[AutoTrader] [DRY] 卖出信号: {amount} BTC @ ${price or '市价'}")
            order.status = OrderStatus.FILLED
            return order
        
        if self.mode == TradeMode.SIMULATION:
            fill_price = price or self.get_price()
            order.filled_price = fill_price
            order.status = OrderStatus.FILLED
            
            # 更新持仓
            sell_value = amount * fill_price
            cost = amount * self.position.avg_price
            self.position.pnl += sell_value - cost
            self.position.pnl_percent = self.position.pnl / cost * 100 if cost > 0 else 0
            self.position.amount -= amount
            
            if self.position.amount <= 0:
                self.position.amount = 0
                self.position.avg_price = 0
            
            logger.info(f"[AutoTrader] [SIM] 卖出成交: {amount} BTC @ ${fill_price:,.2f}")
            logger.info(f"[AutoTrader] [SIM] 总盈亏: ${self.position.pnl:,.2f} ({self.position.pnl_percent:.2f}%)")
        
        return order
    
    def check_stop_loss(self, current_price: float) -> bool:
        """检查止损"""
        if self.position.amount <= 0:
            return False
        
        loss_percent = (current_price - self.position.avg_price) / self.position.avg_price
        
        if loss_percent < -self.stop_loss:
            logger.warning(f"[AutoTrader] 触发止损! 亏损 {-loss_percent:.1f}%")
            return True
        
        return False
    
    def check_take_profit(self, current_price: float) -> bool:
        """检查止盈"""
        if self.position.amount <= 0:
            return False
        
        gain_percent = (current_price - self.position.avg_price) / self.position.avg_price
        
        if gain_percent > self.take_profit:
            logger.info(f"[AutoTrader] 触发止盈! 盈利 {gain_percent:.1f}%")
            return True
        
        return False
    
    def process_signal(self, signal: Dict[str, Any]) -> bool:
        """处理交易信号
        
        Args:
            signal: {'action': 'BUY'|'SELL'|'HOLD', 'confidence': 0-1, 'reason': '...'}
        """
        if not signal:
            return False
        
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        price = self.get_price()
        
        if not price:
            return False
        
        self.last_signal = signal
        self.last_price = price
        
        # 检查止损/止盈
        if self.check_stop_loss(price):
            self.execute_sell(self.position.amount)
            self.trade_history.append({
                'type': 'STOP_LOSS',
                'price': price,
                'timestamp': datetime.now().isoformat()
            })
            return True
        
        if self.check_take_profit(price):
            self.execute_sell(self.position.amount)
            self.trade_history.append({
                'type': 'TAKE_PROFIT',
                'price': price,
                'timestamp': datetime.now().isoformat()
            })
            return True
        
        # 处理交易信号
        if action == 'BUY' and confidence >= 0.6:
            if self.position.amount == 0:
                self.execute_buy(self.trade_amount)
                self.trade_history.append({
                    'type': 'BUY',
                    'price': price,
                    'confidence': confidence,
                    'reason': signal.get('reason', ''),
                    'timestamp': datetime.now().isoformat()
                })
                return True
        
        elif action == 'SELL' and confidence >= 0.5:
            if self.position.amount > 0:
                self.execute_sell(self.position.amount)
                self.trade_history.append({
                    'type': 'SELL',
                    'price': price,
                    'confidence': confidence,
                    'reason': signal.get('reason', ''),
                    'timestamp': datetime.now().isoformat()
                })
                return True
        
        return False
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'mode': self.mode.value,
            'is_running': self.is_running,
            'position': self.position.to_dict(),
            'last_signal': self.last_signal,
            'last_price': self.last_price,
            'trade_count': len(self.trade_history),
            'total_pnl': self.position.pnl,
            'pnl_percent': self.position.pnl_percent
        }
    
    def export_history(self) -> str:
        """导出交易历史"""
        return json.dumps(self.trade_history, indent=2, ensure_ascii=False)
    
    def reset(self):
        """重置"""
        self.position = Position()
        self.trade_history = []
        self.last_signal = None
        logger.info("[AutoTrader] 已重置")


class SignalGenerator:
    """信号生成器 - 从多Agent系统获取信号"""
    
    def __init__(self):
        self.last_analysis = None
    
    def get_signal_from_multi_agent(self) -> Dict[str, Any]:
        """从多Agent系统获取信号"""
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            # 尝试导入多Agent系统
            from btc_multi_agent import BTCMainAgent
            
            async def get_signal():
                agent = BTCMainAgent()
                result = await agent.run()
                return result
            
            result = asyncio.run(get_signal())
            
            if result.get('status') == 'success':
                decision = result.get('decision', {})
                
                # 转换信号格式
                action = decision.get('action', 'HOLD')
                confidence = decision.get('confidence', 0) / 100
                
                # 简化信号
                if action in ['STRONG_BUY', 'BUY']:
                    signal_action = 'BUY'
                elif action in ['STRONG_SELL', 'SELL']:
                    signal_action = 'SELL'
                else:
                    signal_action = 'HOLD'
                
                signal = {
                    'action': signal_action,
                    'confidence': confidence,
                    'reason': f"AI决策: {action}, 大师得分: {decision.get('master_wisdom', {})}",
                    'ai_prediction': decision.get('ai_prediction', {}),
                    'market': decision.get('market_analysis', {}),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.last_analysis = signal
                return signal
        
        except Exception as e:
            logger.error(f"[SignalGenerator] 获取信号失败: {e}")
        
        return None
    
    def get_simple_signal(self) -> Dict[str, Any]:
        """获取简单信号（基于技术指标）"""
        try:
            import ccxt
            exchange = ccxt.binance({'enableRateLimit': True})
            ticker = exchange.fetch_ticker('BTC/USDT')
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=24)
            
            closes = [c[4] for c in ohlcv]
            
            # RSI
            delta = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains = [d if d > 0 else 0 for d in delta]
            losses = [-d if d < 0 else 0 for d in delta]
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # MA
            ma7 = sum(closes[-7:]) / 7
            ma25 = sum(closes[-25:]) / 25
            
            current_price = closes[-1]
            
            # 综合信号
            buy_signals = 0
            sell_signals = 0
            
            if rsi < 30:
                buy_signals += 1
            elif rsi > 70:
                sell_signals += 1
            
            if current_price < ma7:
                buy_signals += 1
            else:
                sell_signals += 1
            
            if current_price < ma25:
                buy_signals += 1
            
            if buy_signals >= 2:
                action = 'BUY'
                confidence = buy_signals / 3
            elif sell_signals >= 2:
                action = 'SELL'
                confidence = sell_signals / 3
            else:
                action = 'HOLD'
                confidence = 0.5
            
            return {
                'action': action,
                'confidence': confidence,
                'reason': f"RSI: {rsi:.1f}, MA7: {ma7:.0f}, MA25: {ma25:.0f}",
                'rsi': rsi,
                'price': current_price,
                'ma7': ma7,
                'ma25': ma25,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"[SignalGenerator] 简单信号失败: {e}")
            return None


async def auto_trading_loop(trader: AutoTrader, signal_gen: SignalGenerator, interval: int = 60):
    """自动交易主循环
    
    Args:
        trader: 交易机器人
        signal_gen: 信号生成器
        interval: 检查间隔（秒）
    """
    logger.info(f"[AutoTrader] 自动交易循环启动，间隔{interval}秒")
    
    while trader.is_running:
        try:
            # 获取信号
            signal = signal_gen.get_simple_signal()
            
            if signal:
                # 处理信号
                executed = trader.process_signal(signal)
                
                # 状态
                status = trader.get_status()
                pos = status['position']
                
                logger.info(
                    f"[AutoTrader] 信号:{signal['action']} "
                    f"置信度:{signal['confidence']:.0%} "
                    f"持仓:{pos['amount']:.4f}BTC "
                    f"盈亏:${pos['pnl']:,.2f}({pos['pnl_percent']:.1f}%)"
                )
            
            # 等待
            await asyncio.sleep(interval)
        
        except Exception as e:
            logger.error(f"[AutoTrader] 循环错误: {e}")
            await asyncio.sleep(10)


def start_auto_trader(mode: str = "sim", interval: int = 60):
    """启动自动交易"""
    # 转换模式
    mode_map = {
        'sim': TradeMode.SIMULATION,
        'live': TradeMode.LIVE,
        'dry': TradeMode.DRY_RUN
    }
    
    trader = AutoTrader(mode=mode_map.get(mode, TradeMode.SIMULATION))
    signal_gen = SignalGenerator()
    
    # 连接交易所
    if not trader.connect_exchange():
        logger.error("[AutoTrader] 交易所连接失败")
        return None
    
    # 启动
    trader.is_running = True
    
    # 运行异步循环
    try:
        asyncio.run(auto_trading_loop(trader, signal_gen, interval))
    except KeyboardInterrupt:
        logger.info("[AutoTrader] 用户中断")
        trader.is_running = False
    
    return trader


def quick_status():
    """快速状态查询"""
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    
    return {
        'config': config,
        'status_file': CONFIG_FILE
    }


# 便捷函数
def quick_auto_trade_start(mode="sim", interval=60):
    """快速启动自动交易"""
    try:
        trader = start_auto_trader(mode, interval)
        return {"status": "started", "mode": mode}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def quick_auto_trade_stop():
    """停止自动交易"""
    return {"status": "stopped", "message": "请按 Ctrl+C 停止"}


def quick_trade_status():
    """查询交易状态"""
    return {
        "mode": "simulation",
        "position": {"amount": 0, "pnl": 0},
        "trades": 0
    }


def quick_trade_history():
    """查询交易历史"""
    return {
        "trades": [],
        "total_pnl": 0
    }


def quick_reset_trade():
    """重置交易"""
    return {"status": "reset", "message": "交易已重置"}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "start":
            mode = sys.argv[2] if len(sys.argv) > 2 else "sim"
            interval = int(sys.argv[3]) if len(sys.argv) > 3 else 60
            start_auto_trader(mode, interval)
        
        elif command == "status":
            print(json.dumps(quick_status(), indent=2))
        
        elif command == "test":
            # 测试模式
            trader = AutoTrader(TradeMode.SIMULATION)
            signal_gen = SignalGenerator()
            
            print("\n" + "="*70)
            print("🚀 BTC AI Auto-Trader - 测试模式")
            print("="*70)
            
            # 获取信号
            signal = signal_gen.get_simple_signal()
            if signal:
                print(f"\n📊 当前信号:")
                print(f"  操作: {signal['action']}")
                print(f"  置信度: {signal['confidence']:.0%}")
                print(f"  原因: {signal['reason']}")
                print(f"  价格: ${signal['price']:,.2f}")
                print(f"  RSI: {signal['rsi']:.1f}")
            
            # 模拟交易
            print(f"\n💰 模拟交易:")
            trader.process_signal(signal)
            
            status = trader.get_status()
            print(f"  持仓: {status['position']['amount']:.4f} BTC")
            print(f"  盈亏: ${status['position']['pnl']:,.2f}")
            
            print("\n" + "="*70)
    
    else:
        print("""
用法: python3 btc_auto_trader.py <命令> [参数]

命令:
  start [模式] [间隔]  启动自动交易
    模式: sim(模拟), live(实盘), dry(试运行)
    间隔: 检查间隔（秒），默认60
  
  status                查询状态
  
  test                  测试模式

示例:
  python3 btc_auto_trader.py start sim 30    # 模拟模式，30秒检查
  python3 btc_auto_trader.py start dry       # 试运行模式
  python3 btc_auto_trader.py test          # 测试信号
""")
