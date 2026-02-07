#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Wisdom Agent - 投资大师智慧分析
====================================
职责：
- 编码四位投资大师的投资理念
- 生成量化的大师智慧特征信号
- 分析当前市场是否符合大师的投资标准

大师投资理念：
- Warren Buffett (巴菲特): 价值投资，安全边际，长期持有
- Charlie Munger (芒格): 逆向思维，心理倾向，多学科思维
- Peter Lynch (林奇): 成长投资，PEG指标，关注身边企业
- Robert Kiyosaki (清崎): 现金流，资产配置，风险管理

Author: AI Trading System
Date: 2024-02-06
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# 大师智慧配置
MASTERS_CONFIG = {
    'buffett': {
        'name': 'Warren Buffett',
        'principles': ['安全边际', '内在价值', '长期持有', '简单业务'],
        'weights': {'value': 0.35, 'margin': 0.30, 'quality': 0.20, 'duration': 0.15}
    },
    'munger': {
        'name': 'Charlie Munger',
        'principles': ['逆向思维', '心理学', '耐心', '避免错误'],
        'weights': {'psychology': 0.40, 'patience': 0.25, 'inversion': 0.20, 'avoidance': 0.15}
    },
    'lynch': {
        'name': 'Peter Lynch',
        'principles': ['成长动量', 'PEG指标', '简单易懂', '关注细节'],
        'weights': {'growth': 0.35, 'peg': 0.30, 'momentum': 0.20, 'simplicity': 0.15}
    },
    'kiyosaki': {
        'name': 'Robert Kiyosaki',
        'principles': ['现金流', '资产配置', '风险管理', '杠杆控制'],
        'weights': {'cashflow': 0.30, 'allocation': 0.25, 'risk': 0.30, 'leverage': 0.15}
    }
}


class BTCWisdomAgent:
    """
    BTC Wisdom Agent - 大师智慧分析
    
    将大师的投资理念编码为可量化的指标
    """
    
    def __init__(self):
        self.name = "btc_wisdom"
        self.status = "idle"
        
    async def run(self, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行大师智慧分析"""
        start_time = time.time()
        self.status = "running"
        
        try:
            logger.info("[btc_wisdom] 开始大师智慧分析...")
            
            if not market_data:
                market_data = self._get_default_market_data()
            
            # 1. 巴菲特价值分析
            buffett_score = self._analyze_buffett_value(market_data)
            
            # 2. 芒格心理分析
            munger_score = self._analyze_munger_psychology(market_data)
            
            # 3. 林奇成长分析
            lynch_score = self._analyze_lynch_growth(market_data)
            
            # 4. 清崎风险分析
            kiyosaki_score = self._analyze_kiyosaki_risk(market_data)
            
            # 5. 计算综合大师智慧得分
            master_wisdom_score = self._calculate_master_wisdom_score(
                buffett_score, munger_score, lynch_score, kiyosaki_score
            )
            
            # 6. 生成交易建议
            wisdom_signals = self._generate_wisdom_signals(
                buffett_score, munger_score, lynch_score, kiyosaki_score
            )
            
            execution_time = time.time() - start_time
            self.status = "completed"
            
            result = {
                'status': 'success',
                'data': {
                    'buffett_value_score': buffett_score['total_score'],
                    'buffett_details': buffett_score,
                    'munger_psychology_score': munger_score['total_score'],
                    'munger_details': munger_score,
                    'lynch_growth_score': lynch_score['total_score'],
                    'lynch_details': lynch_score,
                    'kiyosaki_risk_score': kiyosaki_score['total_score'],
                    'kiyosaki_details': kiyosaki_score,
                    'master_wisdom_score': master_wisdom_score,
                    'wisdom_signals': wisdom_signals,
                    'master_quotes': self._get_master_quotes()
                },
                'execution_time': execution_time
            }
            
            logger.info(f"[btc_wisdom] 完成，耗时: {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"[btc_wisdom] 错误: {e}")
            self.status = "error"
            return {
                'status': 'error',
                'error': str(e),
                'data': self._get_mock_result()
            }
    
    def _analyze_buffett_value(self, data: Dict) -> Dict:
        """巴菲特价值投资分析"""
        indicators = data.get('technical_indicators', {})
        price = data.get('current_price', 45000)
        
        # 1. 安全边际计算 (0-100)
        sma_99 = indicators.get('sma_99', price)
        price_vs_ma = ((price - sma_99) / sma_99) * 100
        
        if price_vs_ma < -20:
            margin_score = 100
        elif price_vs_ma < -10:
            margin_score = 80
        elif price_vs_ma < -5:
            margin_score = 65
        elif price_vs_ma < 0:
            margin_score = 55
        elif price_vs_ma < 10:
            margin_score = 40
        elif price_vs_ma < 20:
            margin_score = 30
        else:
            margin_score = 15
        
        # 2. 内在价值评分
        rsi = indicators.get('rsi', 50)
        volatility = indicators.get('volatility', 30)
        value_score = 50
        value_score += (50 - volatility) * 0.3
        if 40 < rsi < 70:
            value_score += 15
        if volatility < 40:
            value_score += 10
        value_score = min(100, max(0, value_score))
        
        # 3. 质量评分
        trend = data.get('trend', 'SIDEWAYS')
        trend_score = {
            'STRONG_UPTREND': 90,
            'UPTREND': 75,
            'SIDEWAYS': 50,
            'DOWNTREND': 30,
            'STRONG_DOWNTREND': 15
        }.get(trend, 50)
        
        # 4. 长期持有准备度
        bb_position = self._get_bollinger_position(price, indicators)
        holding_score = 50
        if bb_position < 0.2:
            holding_score = 85
        elif bb_position < 0.4:
            holding_score = 70
        elif bb_position > 0.8:
            holding_score = 25
        elif bb_position > 0.6:
            holding_score = 40
        else:
            holding_score = 55
        
        weights = MASTERS_CONFIG['buffett']['weights']
        total_score = (
            value_score * weights['value'] +
            margin_score * weights['margin'] +
            trend_score * weights['quality'] +
            holding_score * weights['duration']
        )
        
        return {
            'total_score': round(total_score, 2),
            'components': {
                'value_score': round(value_score, 2),
                'margin_score': round(margin_score, 2),
                'quality_score': round(trend_score, 2),
                'holding_score': round(holding_score, 2)
            },
            'verdict': self._get_buffett_verdict(total_score),
            'signal': 'BUY' if total_score > 65 else ('HOLD' if total_score > 45 else 'WAIT')
        }
    
    def _analyze_munger_psychology(self, data: Dict) -> Dict:
        """芒格心理学分析"""
        indicators = data.get('technical_indicators', {})
        price = data.get('current_price', 45000)
        rsi = indicators.get('rsi', 50)
        volatility = indicators.get('volatility', 30)
        
        # 1. 市场情绪评分
        fear_greed_score = 50
        if rsi < 20:
            fear_greed_score = 95
        elif rsi < 30:
            fear_greed_score = 80
        elif rsi < 40:
            fear_greed_score = 60
        elif rsi > 80:
            fear_greed_score = 10
        elif rsi > 70:
            fear_greed_score = 25
        elif rsi > 60:
            fear_greed_score = 40
        else:
            fear_greed_score = 50
        
        # 2. 逆向思维评分
        bb_position = self._get_bollinger_position(price, indicators)
        contrarian_score = 50
        if bb_position < 0.15:
            contrarian_score = 90
        elif bb_position < 0.25:
            contrarian_score = 75
        elif bb_position > 0.85:
            contrarian_score = 15
        elif bb_position > 0.75:
            contrarian_score = 30
        
        contrarian_score = min(100, max(0, contrarian_score))
        
        # 3. 耐心度评分
        patience_score = 50
        trend = data.get('trend', 'SIDEWAYS')
        if trend in ['STRONG_UPTREND', 'UPTREND']:
            patience_score = 35
        elif trend in ['STRONG_DOWNTREND', 'DOWNTREND']:
            patience_score = 70
        else:
            patience_score = 55
        
        if 40 < rsi < 60:
            patience_score = 65
        
        # 4. 错误避免评分
        avoidance_score = 50
        if rsi > 80:
            avoidance_score -= 30
        if rsi < 20:
            avoidance_score += 20
        if volatility > 70:
            avoidance_score -= 15
        avoidance_score = min(100, max(0, avoidance_score))
        
        weights = MASTERS_CONFIG['munger']['weights']
        total_score = (
            fear_greed_score * weights['psychology'] +
            contrarian_score * weights['inversion'] +
            patience_score * weights['patience'] +
            avoidance_score * weights['avoidance']
        )
        
        return {
            'total_score': round(total_score, 2),
            'components': {
                'psychology_score': round(fear_greed_score, 2),
                'contrarian_score': round(contrarian_score, 2),
                'patience_score': round(patience_score, 2),
                'avoidance_score': round(avoidance_score, 2)
            },
            'verdict': self._get_munger_verdict(total_score),
            'signal': 'BUY' if total_score > 60 else ('WAIT' if total_score > 40 else 'CAUTION')
        }
    
    def _analyze_lynch_growth(self, data: Dict) -> Dict:
        """林奇成长投资分析"""
        indicators = data.get('technical_indicators', {})
        price = data.get('current_price', 45000)
        sma_7 = indicators.get('sma_7', price)
        sma_25 = indicators.get('sma_25', price)
        rsi = indicators.get('rsi', 50)
        volatility = indicators.get('volatility', 30)
        
        # 1. 成长动量评分
        momentum_score = 50
        if sma_7 > sma_25 * 1.05:
            momentum_score = 90
        elif sma_7 > sma_25 * 1.02:
            momentum_score = 75
        elif sma_7 > sma_25:
            momentum_score = 60
        elif sma_7 < sma_25 * 0.95:
            momentum_score = 20
        elif sma_7 < sma_25 * 0.98:
            momentum_score = 35
        else:
            momentum_score = 50
        
        if 50 < rsi < 70:
            momentum_score += 10
        elif rsi > 70:
            momentum_score += 5
        elif rsi < 30:
            momentum_score -= 10
        momentum_score = min(100, max(0, momentum_score))
        
        # 2. PEG评估
        peg_score = 50
        if volatility < 30:
            peg_score = 40
        elif volatility < 50:
            peg_score = 55
        elif volatility < 70:
            peg_score = 65
        else:
            peg_score = 50
        
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist > 0:
            peg_score += 10
        elif macd_hist < -100:
            peg_score -= 15
        peg_score = min(100, max(0, peg_score))
        
        # 3. 动量加速度
        acceleration_score = 50
        bb_position = self._get_bollinger_position(price, indicators)
        if bb_position > 0.7 and bb_position < 0.9:
            acceleration_score = 75
        elif bb_position >= 0.9:
            acceleration_score = 50
        elif bb_position < 0.3:
            acceleration_score = 35
        else:
            acceleration_score = 55
        acceleration_score = min(100, max(0, acceleration_score))
        
        # 4. 简单性评分
        simplicity_score = 85
        
        weights = MASTERS_CONFIG['lynch']['weights']
        total_score = (
            momentum_score * weights['growth'] +
            peg_score * weights['peg'] +
            acceleration_score * weights['momentum'] +
            simplicity_score * weights['simplicity']
        )
        
        return {
            'total_score': round(total_score, 2),
            'components': {
                'momentum_score': round(momentum_score, 2),
                'peg_score': round(peg_score, 2),
                'acceleration_score': round(acceleration_score, 2),
                'simplicity_score': round(simplicity_score, 2)
            },
            'verdict': self._get_lynch_verdict(total_score),
            'signal': 'BUY' if total_score > 60 else ('HOLD' if total_score > 45 else 'AVOID')
        }
    
    def _analyze_kiyosaki_risk(self, data: Dict) -> Dict:
        """清崎风险管理分析"""
        indicators = data.get('technical_indicators', {})
        price = data.get('current_price', 45000)
        rsi = indicators.get('rsi', 50)
        volatility = indicators.get('volatility', 30)
        trend = data.get('trend', 'SIDEWAYS')
        bb_position = self._get_bollinger_position(price, indicators)
        
        # 1. 现金流评分
        cashflow_score = 50
        if rsi < 30:
            cashflow_score = 80
        elif rsi < 40:
            cashflow_score = 65
        elif 40 <= rsi <= 60:
            cashflow_score = 55
        elif rsi > 70:
            cashflow_score = 30
        elif rsi > 60:
            cashflow_score = 40
        
        if volatility < 30:
            cashflow_score -= 10
        elif volatility > 60:
            cashflow_score += 15
        elif volatility > 40:
            cashflow_score += 5
        cashflow_score = min(100, max(0, cashflow_score))
        
        # 2. 资产配置评分
        allocation_score = 60
        if trend in ['STRONG_UPTREND']:
            allocation_score = 40
        elif trend in ['UPTREND']:
            allocation_score = 55
        elif trend in ['DOWNTREND']:
            allocation_score = 70
        elif trend in ['STRONG_DOWNTREND']:
            allocation_score = 85
        else:
            allocation_score = 60
        
        if rsi < 25:
            allocation_score += 10
        elif rsi > 75:
            allocation_score -= 20
        allocation_score = min(100, max(0, allocation_score))
        
        # 3. 风险控制评分
        risk_score = 50
        if bb_position < 0.2:
            risk_score += 20
        elif bb_position > 0.8:
            risk_score -= 25
        risk_score = min(100, max(0, risk_score))
        
        # 4. 杠杆控制评分
        leverage_score = 70
        if volatility > 60:
            leverage_score = 40
        elif volatility > 40:
            leverage_score = 55
        elif volatility < 30:
            leverage_score = 80
        
        if trend in ['STRONG_UPTREND', 'UPTREND']:
            leverage_score += 10
        elif trend in ['STRONG_DOWNTREND', 'DOWNTREND']:
            leverage_score -= 15
        leverage_score = min(100, max(0, leverage_score))
        
        weights = MASTERS_CONFIG['kiyosaki']['weights']
        total_score = (
            cashflow_score * weights['cashflow'] +
            allocation_score * weights['allocation'] +
            risk_score * weights['risk'] +
            leverage_score * weights['leverage']
        )
        
        return {
            'total_score': round(total_score, 2),
            'components': {
                'cashflow_score': round(cashflow_score, 2),
                'allocation_score': round(allocation_score, 2),
                'risk_score': round(risk_score, 2),
                'leverage_score': round(leverage_score, 2)
            },
            'verdict': self._get_kiyosaki_verdict(total_score),
            'signal': 'SAFE' if total_score > 60 else ('MODERATE' if total_score > 45 else 'RISKY')
        }
    
    def _calculate_master_wisdom_score(self, buffett: Dict, munger: Dict, 
                                       lynch: Dict, kiyosaki: Dict) -> float:
        """计算综合大师智慧得分"""
        weights = {
            'buffett': 0.30,
            'munger': 0.25,
            'lynch': 0.20,
            'kiyosaki': 0.25
        }
        
        total = (
            buffett['total_score'] * weights['buffett'] +
            munger['total_score'] * weights['munger'] +
            lynch['total_score'] * weights['lynch'] +
            kiyosaki['total_score'] * weights['kiyosaki']
        )
        
        return round(total, 2)
    
    def _generate_wisdom_signals(self, buffett: Dict, munger: Dict,
                                  lynch: Dict, kiyosaki: Dict) -> Dict:
        """生成大师智慧交易信号"""
        signals = {
            'buffett': buffett.get('signal', 'HOLD'),
            'munger': munger.get('signal', 'WAIT'),
            'lynch': lynch.get('signal', 'AVOID'),
            'kiyosaki': kiyosaki.get('signal', 'MODERATE')
        }
        
        buy_signals = sum(1 for s in signals.values() if s in ['BUY', 'SAFE'])
        sell_signals = sum(1 for s in signals.values() if s in ['SELL', 'RISKY'])
        wait_signals = sum(1 for s in signals.values() if s in ['WAIT', 'HOLD', 'AVOID', 'MODERATE', 'CAUTION'])
        
        if buy_signals >= 3:
            final_signal = 'STRONG_BUY'
        elif buy_signals > sell_signals:
            final_signal = 'BUY'
        elif sell_signals >= 3:
            final_signal = 'STRONG_SELL'
        elif sell_signals > buy_signals:
            final_signal = 'SELL'
        else:
            final_signal = 'WAIT'
        
        return {
            'individual_signals': signals,
            'final_signal': final_signal,
            'consensus': f"{buy_signals}BUY/{sell_signals}SELL/{wait_signals}WAIT"
        }
    
    def _get_bollinger_position(self, price: float, indicators: Dict) -> float:
        """计算价格在布林带中的位置 (0-1)"""
        bb_upper = indicators.get('bb_upper', price * 1.05)
        bb_lower = indicators.get('bb_lower', price * 0.95)
        
        if bb_upper == bb_lower:
            return 0.5
        
        position = (price - bb_lower) / (bb_upper - bb_lower)
        return max(0, min(1, position))
    
    def _get_buffett_verdict(self, score: float) -> str:
        """巴菲特判断"""
        if score > 80:
            return "绝佳买入机会！安全边际极高"
        elif score > 65:
            return "良好买入时机"
        elif score > 50:
            return "合理估值，观望为主"
        elif score > 35:
            return "估值偏高，谨慎"
        else:
            return "远离或减持"
    
    def _get_munger_verdict(self, score: float) -> str:
        """芒格判断"""
        if score > 75:
            return "逆向买入良机！市场极度恐惧"
        elif score > 60:
            return "可考虑逆向操作"
        elif score > 45:
            return "保持理性，观望"
        elif score > 30:
            return "市场过热，谨慎"
        else:
            return "极度贪婪，远离"
    
    def _get_lynch_verdict(self, score: float) -> str:
        """林奇判断"""
        if score > 75:
            return "高成长动量，可积极配置"
        elif score > 60:
            return "稳健增长，适度配置"
        elif score > 45:
            return "成长放缓，观望"
        else:
            return "缺乏增长动能，避免"
    
    def _get_kiyosaki_verdict(self, score: float) -> str:
        """清崎判断"""
        if score > 75:
            return "风险可控，可适度加仓"
        elif score > 60:
            return "风险管理良好"
        elif score > 45:
            return "风险中等，谨慎操作"
        else:
            return "高风险，建议减仓"
    
    def _get_master_quotes(self) -> Dict:
        """获取大师经典语录"""
        return {
            'buffett': '"价格是你支付的，价值是你得到的"',
            'munger': '"反过来想，总是反过来想"',
            'lynch': '"投资你了解的东西"',
            'kiyosaki': '"让钱为你工作"'
        }
    
    def _get_default_market_data(self) -> Dict:
        """获取默认市场数据"""
        return {
            'current_price': 45000,
            'trend': 'SIDEWAYS',
            'technical_indicators': {
                'rsi': 50,
                'sma_7': 44800,
                'sma_25': 44500,
                'sma_99': 44000,
                'bb_upper': 47000,
                'bb_middle': 45000,
                'bb_lower': 43000,
                'macd': 100,
                'macd_histogram': 50,
                'atr': 1500,
                'volatility': 40
            }
        }
    
    def _get_mock_result(self) -> Dict:
        """获取模拟结果"""
        return {
            'buffett_value_score': 55,
            'munger_psychology_score': 50,
            'lynch_growth_score': 55,
            'kiyosaki_risk_score': 50,
            'master_wisdom_score': 52.5,
            'wisdom_signals': {
                'final_signal': 'WAIT',
                'consensus': '1BUY/1SELL/2WAIT'
            }
        }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    agent = BTCWisdomAgent()
    result = asyncio.run(agent.run())
    print(json.dumps(result, indent=2, ensure_ascii=False))
