#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Data Agent - 数据获取与处理
==============================
职责：
- 从多数据源获取BTC价格数据
- 数据清洗和预处理
- 计算技术指标
- 生成市场信号

数据源：
- Binance API (主要)
- CoinGecko API (备用)
- 本地缓存

Author: AI Trading System
Date: 2024-02-06
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import aiohttp
import numpy as np

logger = logging.getLogger(__name__)

# 配置
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'market_cache.json')
os.makedirs(CACHE_DIR, exist_ok=True)


@dataclass
class MarketData:
    """市场数据结构"""
    timestamp: str
    current_price: float
    high_24h: float
    low_24h: float
    volume_24h: float
    price_change_24h: float
    price_change_percent_24h: float


class BTCDataAgent:
    """BTC数据Agent"""
    
    def __init__(self):
        self.name = "btc_data"
        self.status = "idle"
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """加载缓存数据"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self, data: Dict):
        """保存缓存"""
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def run(self, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行数据获取和预处理
        
        Args:
            initial_data: 初始数据（可选）
            
        Returns:
            处理后的市场数据和技术指标
        """
        start_time = time.time()
        self.status = "running"
        
        try:
            logger.info("[btc_data] 开始获取市场数据...")
            
            # 1. 获取市场数据
            market_data = await self._fetch_market_data(initial_data)
            
            # 2. 数据清洗
            cleaned_data = self._clean_data(market_data)
            
            # 3. 计算技术指标
            technical_indicators = self._calculate_indicators(cleaned_data)
            
            # 4. 生成市场信号
            market_signal = self._generate_market_signal(cleaned_data, technical_indicators)
            
            # 5. 更新缓存
            self._save_cache(cleaned_data)
            
            execution_time = time.time() - start_time
            self.status = "completed"
            
            result = {
                'status': 'success',
                'data': {
                    'market_data': cleaned_data,
                    'technical_indicators': technical_indicators,
                    'market_signal': market_signal
                },
                'execution_time': execution_time
            }
            
            logger.info(f"[btc_data] 完成，耗时: {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"[btc_data] 错误: {e}")
            self.status = "error"
            return {
                'status': 'error',
                'error': str(e),
                'data': self._get_mock_data()
            }
    
    async def _fetch_market_data(self, initial_data: Dict = None) -> Dict:
        """从多数据源获取市场数据"""
        
        # 如果有初始数据，直接使用
        if initial_data:
            return initial_data
        
        # 尝试从Binance获取
        try:
            return await self._fetch_binance_data()
        except Exception as e:
            logger.warning(f"Binance获取失败: {e}")
        
        # 尝试从CoinGecko获取
        try:
            return await self._fetch_coingecko_data()
        except Exception as e:
            logger.warning(f"CoinGecko获取失败: {e}")
        
        # 返回缓存数据或模拟数据
        return self.cache or self._generate_simulated_data()
    
    async def _fetch_binance_data(self) -> Dict:
        """从Binance获取数据"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                'https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT',
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'symbol': 'BTCUSDT',
                        'current_price': float(data['lastPrice']),
                        'high_24h': float(data['highPrice']),
                        'low_24h': float(data['lowPrice']),
                        'volume_24h': float(data['quoteVolume']),
                        'price_change_24h': float(data['priceChange']),
                        'price_change_percent_24h': float(data['priceChangePercent']),
                        'weighted_avg_price': float(data['weightedAvgPrice']),
                        'source': 'binance',
                        'timestamp': datetime.now().isoformat()
                    }
        raise Exception("Binance API响应异常")
    
    async def _fetch_coingecko_data(self) -> Dict:
        """从CoinGecko获取数据"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                'https://api.coingecko.com/api/v3/coins/bitcoin',
                params={'localization': 'false', 'tickers': 'false', 'community_data': 'false', 'developer_data': 'false'},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    market = data['market_data']
                    return {
                        'symbol': 'BTC',
                        'current_price': market['current_price']['usd'],
                        'high_24h': market['high_24h']['usd'],
                        'low_24h': market['low_24h']['usd'],
                        'volume_24h': market['total_volume']['usd'],
                        'price_change_24h': market['price_change_24h'],
                        'price_change_percent_24h': market['price_change_percentage_24h'],
                        'source': 'coingecko',
                        'timestamp': datetime.now().isoformat()
                    }
        raise Exception("CoinGecko API响应异常")
    
    def _clean_data(self, data: Dict) -> Dict:
        """数据清洗"""
        if not data:
            return self._get_mock_data()
        
        # 确保必要字段存在
        required_fields = ['current_price', 'high_24h', 'low_24h', 'volume_24h']
        for field in required_fields:
            if field not in data or data[field] is None:
                data[field] = 0
        
        # 计算价格区间
        data['price_range'] = data['high_24h'] - data['low_24h']
        data['price_range_percent'] = (data['price_range'] / data['current_price'] * 100) if data['current_price'] else 0
        
        # 计算支撑/阻力位
        data['support_level'] = data['current_price'] * 0.95
        data['resistance_level'] = data['current_price'] * 1.05
        
        return data
    
    def _calculate_indicators(self, data: Dict) -> Dict:
        """计算技术指标"""
        indicators = {}
        
        # 使用模拟历史数据计算指标（实际应使用真实历史数据）
        historical_data = self._get_historical_prices(data.get('current_price', 45000))
        
        # RSI (Relative Strength Index)
        indicators['rsi'] = self._calculate_rsi(historical_data, 14)
        
        # MACD
        macd, signal, hist = self._calculate_macd(historical_data, 12, 26, 9)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = hist
        
        # 移动平均线
        indicators['sma_7'] = self._calculate_sma(historical_data, 7)
        indicators['sma_25'] = self._calculate_sma(historical_data, 25)
        indicators['sma_99'] = self._calculate_sma(historical_data, 99)
        
        # 布林带
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(historical_data, 20, 2)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # ATR (Average True Range)
        indicators['atr'] = self._calculate_atr(historical_data, 14)
        
        # 波动率
        indicators['volatility'] = self._calculate_volatility(historical_data)
        
        # 成交量变化
        indicators['volume_change'] = self._calculate_volume_change(data.get('volume_24h', 0))
        
        return indicators
    
    def _get_historical_prices(self, current_price: float, days: int = 100) -> List[float]:
        """生成模拟历史价格数据"""
        np.random.seed(42)  # 可重复性
        base_price = current_price
        
        # 生成随机波动
        returns = np.random.normal(0.001, 0.02, days)
        prices = [base_price * (1 + sum(returns[i:])) for i in range(days)]
        
        return prices[::-1]  #  oldest to newest
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
        """计算MACD"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema([macd_line], signal)
        histogram = macd_line - signal_line
        
        return round(macd_line, 2), round(signal_line, 2), round(histogram, 2)
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """计算简单移动平均"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        return round(np.mean(prices[-period:]), 2)
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """计算指数移动平均"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        prices_array = np.array(prices[-period:])
        multiplier = 2 / (period + 1)
        
        ema = prices_array[0]
        for price in prices_array[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return round(ema, 2)
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2):
        """计算布林带"""
        if len(prices) < period:
            current = prices[-1] if prices else 0
            return current, current, current
        
        prices_array = np.array(prices[-period:])
        middle = np.mean(prices_array)
        std = np.std(prices_array)
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return round(upper, 2), round(middle, 2), round(lower, 2)
    
    def _calculate_atr(self, prices: List[float], period: int = 14) -> float:
        """计算ATR"""
        if len(prices) < period + 1:
            return 0
        
        true_ranges = []
        for i in range(1, len(prices)):
            high = max(prices[i], prices[i-1])
            low = min(prices[i], prices[i-1])
            tr = high - low
            true_ranges.append(tr)
        
        atr = np.mean(true_ranges[-period:]) if len(true_ranges) >= period else np.mean(true_ranges)
        return round(atr, 2)
    
    def _calculate_volatility(self, prices: List[float], period: int = 30) -> float:
        """计算波动率"""
        if len(prices) < 2:
            return 0
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-period:]) * np.sqrt(365) * 100
        
        return round(volatility, 2)
    
    def _calculate_volume_change(self, volume: float) -> float:
        """计算成交量变化（模拟）"""
        # 实际应与历史成交量比较
        return 0.0
    
    def _generate_market_signal(self, data: Dict, indicators: Dict) -> Dict:
        """生成市场信号"""
        signals = {}
        
        # 趋势判断
        current_price = data.get('current_price', 0)
        sma_7 = indicators.get('sma_7', 0)
        sma_25 = indicators.get('sma_25', 0)
        sma_99 = indicators.get('sma_99', 0)
        
        if sma_7 > sma_25 > sma_99:
            trend = "STRONG_UPTREND"
        elif sma_7 > sma_25:
            trend = "UPTREND"
        elif sma_7 < sma_25 < sma_99:
            trend = "STRONG_DOWNTREND"
        elif sma_7 < sma_25:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"
        
        signals['trend'] = trend
        
        # RSI信号
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            rsi_signal = "OVERBOUGHT"
        elif rsi < 30:
            rsi_signal = "OVERSOLD"
        else:
            rsi_signal = "NEUTRAL"
        signals['rsi'] = {'value': rsi, 'signal': rsi_signal}
        
        # MACD信号
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist > 0:
            macd_signal = "BULLISH"
        elif macd_hist < 0:
            macd_signal = "BEARISH"
        else:
            macd_signal = "NEUTRAL"
        signals['macd'] = {'value': macd_hist, 'signal': macd_signal}
        
        # 综合技术得分 (0-100)
        technical_score = 50  # 基础分
        
        # RSI贡献
        if rsi_signal == "OVERSOLD":
            technical_score += 15
        elif rsi_signal == "OVERBOUGHT":
            technical_score -= 15
        elif rsi_signal == "NEUTRAL":
            technical_score += 5
        
        # MACD贡献
        if macd_signal == "BULLISH":
            technical_score += 15
        elif macd_signal == "BEARISH":
            technical_score -= 15
        
        # 趋势贡献
        if "UPTREND" in trend:
            technical_score += 10
        elif "DOWNTREND" in trend:
            technical_score -= 10
        
        signals['technical_score'] = max(0, min(100, technical_score))
        
        # 波动率评估
        volatility = indicators.get('volatility', 0)
        if volatility > 80:
            volatility_level = "EXTREME"
        elif volatility > 50:
            volatility_level = "HIGH"
        elif volatility > 30:
            volatility_level = "MODERATE"
        else:
            volatility_level = "LOW"
        signals['volatility'] = volatility
        signals['volatility_level'] = volatility_level
        
        return signals
    
    def _generate_simulated_data(self) -> Dict:
        """生成模拟市场数据（用于测试）"""
        base_price = 45000  # 假设BTC价格
        
        return {
            'symbol': 'BTCUSDT',
            'current_price': base_price,
            'high_24h': base_price * 1.02,
            'low_24h': base_price * 0.98,
            'volume_24h': 25000000000,
            'price_change_24h': 500,
            'price_change_percent_24h': 1.1,
            'source': 'simulated',
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_mock_data(self) -> Dict:
        """获取模拟数据（错误回退）"""
        return {
            'status': 'mock',
            'data': self._generate_simulated_data()
        }


# 单元测试
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    agent = BTCDataAgent()
    result = asyncio.run(agent.run())
    print(json.dumps(result, indent=2, ensure_ascii=False))
