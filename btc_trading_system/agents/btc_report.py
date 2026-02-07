#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Report Agent - 报告生成
============================
职责：
- 汇总各Agent结果
- 生成综合交易报告
- 提供可读性建议

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

logger = logging.getLogger(__name__)

# 报告保存目录
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


class BTCReportAgent:
    """BTC报告Agent"""
    
    def __init__(self):
        self.name = "btc_report"
        self.status = "idle"
    
    async def run(self, market_data: Dict[str, Any] = None, wisdom_data: Dict[str, Any] = None, ai_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成报告
        
        Args:
            market_data: 市场数据（包含technical_indicators, market_signal等）
            wisdom_data: 大师智慧分析结果
            ai_data: AI预测结果
        """
        start_time = time.time()
        self.status = "running"
        
        try:
            logger.info("[btc_report] 开始生成报告...")
            
            if not market_data:
                market_data = self._get_default_data()
            
            # 如果wisdom_data和ai_data没有传入，尝试从market_data中提取
            if wisdom_data is None:
                wisdom_data = market_data.get('wisdom_data', market_data.get('master_wisdom', {}))
            if ai_data is None:
                ai_data = market_data.get('ai_data', market_data.get('ai_prediction', {}))
            
            # 获取技术数据 - 从market_data和market_signal合并
            market_inner = market_data.get('market_data', {})
            tech_signal = market_data.get('market_signal', {})
            tech_indicators = market_data.get('technical_indicators', {})
            
            # 合并所有技术数据（market_data优先，然后是indicators，然后是signal）
            technical_data = {**tech_signal, **tech_indicators, **market_inner}
            
            # 生成报告
            summary = self._generate_summary(wisdom_data, technical_data, ai_data)
            full_report = self._generate_full_report(wisdom_data, technical_data, ai_data)
            
            # 保存报告
            self._save_report(full_report)
            
            execution_time = time.time() - start_time
            self.status = "completed"
            
            result = {
                'status': 'success',
                'data': {
                    'summary': summary,
                    'full_report': full_report,
                    'recommendation': self._get_recommendation(wisdom_data, ai_data),
                    'risk_warning': self._get_risk_warning(technical_data),
                    'next_review': self._get_next_review_time()
                },
                'execution_time': execution_time
            }
            
            logger.info(f"[btc_report] 完成，耗时: {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"[btc_report] 错误: {e}")
            self.status = "error"
            return {
                'status': 'error',
                'error': str(e),
                'data': self._get_mock_result()
            }
    
    def _generate_summary(self, wisdom: Dict, technical: Dict, ai: Dict) -> str:
        """生成摘要"""
        # 大师智慧得分
        buffett = wisdom.get('buffett_value_score', 50)
        munger = wisdom.get('munger_psychology_score', 50)
        lynch = wisdom.get('lynch_growth_score', 50)
        kiyosaki = wisdom.get('kiyosaki_risk_score', 50)
        master_avg = (buffett + munger + lynch + kiyosaki) / 4
        
        # AI预测
        direction = ai.get('direction', 'SIDEWAYS')
        probability = ai.get('probability', 0.5)
        
        # 趋势 - 处理字典格式
        trend = technical.get('trend', 'SIDEWAYS')
        if isinstance(trend, dict):
            trend = trend.get('signal', trend.get('SIDEWAYS'))
        
        # 价格处理
        price = technical.get('current_price', technical.get('price', 'N/A'))
        if isinstance(price, (int, float)):
            price_str = f"{price:,.2f} USDT"
        else:
            price_str = str(price)
        
        # 生成摘要
        if master_avg > 65:
            wisdom_status = "✅ 大师信号偏多"
        elif master_avg > 50:
            wisdom_status = "➖ 大师信号中性"
        else:
            wisdom_status = "⚠️ 大师信号偏空"
        
        if direction == 'UP':
            ai_status = f"🤖 AI预测: 上涨 ({probability:.1%})"
        elif direction == 'DOWN':
            ai_status = f"🤖 AI预测: 下跌 ({probability:.1%})"
        else:
            ai_status = f"🤖 AI预测: 震荡 ({probability:.1%})"
        
        return f"""
📊 BTC AI 交易信号摘要
━━━━━━━━━━━━━━━━━━━━━━━━
💰 当前价格: {price_str}
📈 趋势: {trend}
━━━━━━━━━━━━━━━━━━━━━━━━
🎓 大师智慧评估: {wisdom_status}
   • 巴菲特价值: {buffett:.1f}
   • 芒格心理: {munger:.1f}
   • 林奇成长: {lynch:.1f}
   • 清崎风险: {kiyosaki:.1f}
   • 综合得分: {master_avg:.1f}
━━━━━━━━━━━━━━━━━━━━━━━━
{ai_status}
   • 置信度: {ai.get('confidence', 0.6):.1%}
   • 综合评分: {ai.get('composite_score', 50):.1f}
━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()
    
    def _generate_full_report(self, wisdom: Dict, technical: Dict, ai: Dict) -> str:
        """生成完整报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 各维度详情
        buffett = wisdom.get('buffett_details', {})
        munger = wisdom.get('munger_details', {})
        lynch = wisdom.get('lynch_details', {})
        kiyosaki = wisdom.get('kiyosaki_details', {})
        
        # 处理技术数据中的字典格式字段
        rsi = technical.get('rsi', 50)
        rsi_value = rsi.get('value', rsi) if isinstance(rsi, dict) else rsi
        
        macd = technical.get('macd', 0)
        macd_value = macd.get('value', macd) if isinstance(macd, dict) else macd
        
        price = technical.get('current_price', technical.get('price', 'N/A'))
        
        # 处理price_change_24h
        price_change_24h = technical.get('price_change_24h', 0)
        if isinstance(price_change_24h, (int, float)):
            price_change_str = f"{price_change_24h:+,.2f}"
        else:
            price_change_str = str(price_change_24h)
        
        # 处理volatility
        volatility = technical.get('volatility', 0)
        if isinstance(volatility, dict):
            volatility = volatility.get('value', volatility)
        
        trend = technical.get('trend', 'N/A')

        # 处理price为字符串
        price = technical.get('current_price', technical.get('price', 'N/A'))
        if isinstance(price, (int, float)):
            price_str = f"{price:,.2f} USDT"
        else:
            price_str = str(price)
        
        report = f"""
{'='*60}
             BTC AI Trader - 综合分析报告
{'='*60}
📅 生成时间: {timestamp}
{'='*60}

📈 市场概览
────────────────────────────────────────
当前价格: {price_str}
24h涨跌: {price_change_str}
24h波动率: {volatility}%
趋势: {trend}
RSI: {rsi_value:.2f}
MACD: {macd_value}

🎓 投资大师智慧分析
────────────────────────────────────────

🧙 巴菲特价值投资 (得分: {wisdom.get('buffett_value_score', 50):.1f})
   • 内在价值: {buffett.get('components', {}).get('value_score', 'N/A')}
   • 安全边际: {buffett.get('components', {}).get('margin_score', 'N/A')}
   • 质量评分: {buffett.get('components', {}).get('quality_score', 'N/A')}
   • 持有准备: {buffett.get('components', {}).get('holding_score', 'N/A')}
   • 判定: {buffett.get('verdict', 'N/A')}
   • 信号: {buffett.get('signal', 'N/A')}

🎭 芒格投资心理 (得分: {wisdom.get('munger_psychology_score', 50):.1f})
   • 市场情绪: {munger.get('components', {}).get('psychology_score', 'N/A')}
   • 逆向思维: {munger.get('components', {}).get('contrarian_score', 'N/A')}
   • 耐心度: {munger.get('components', {}).get('patience_score', 'N/A')}
   • 错误避免: {munger.get('components', {}).get('avoidance_score', 'N/A')}
   • 判定: {munger.get('verdict', 'N/A')}
   • 信号: {munger.get('signal', 'N/A')}

📈 林奇成长投资 (得分: {wisdom.get('lynch_growth_score', 50):.1f})
   • 成长动量: {lynch.get('components', {}).get('momentum_score', 'N/A')}
   • PEG评估: {lynch.get('components', {}).get('peg_score', 'N/A')}
   • 动量加速: {lynch.get('components', {}).get('acceleration_score', 'N/A')}
   • 简单评分: {lynch.get('components', {}).get('simplicity_score', 'N/A')}
   • 判定: {lynch.get('verdict', 'N/A')}
   • 信号: {lynch.get('signal', 'N/A')}

💰 清崎风险管理 (得分: {wisdom.get('kiyosaki_risk_score', 50):.1f})
   • 现金流: {kiyosaki.get('components', {}).get('cashflow_score', 'N/A')}
   • 配置建议: {kiyosaki.get('components', {}).get('allocation_score', 'N/A')}
   • 风险控制: {kiyosaki.get('components', {}).get('risk_score', 'N/A')}
   • 杠杆控制: {kiyosaki.get('components', {}).get('leverage_score', 'N/A')}
   • 判定: {kiyosaki.get('verdict', 'N/A')}
   • 信号: {kiyosaki.get('signal', 'N/A')}

🤖 AI预测分析
────────────────────────────────────────
预测方向: {ai.get('direction', 'N/A')}
上涨概率: {ai.get('probability', 0):.2%}
置信度: {ai.get('confidence', 0):.2%}
预测价格变动: {ai.get('price_change', 0):.2f}%
综合评分: {ai.get('composite_score', 50):.1f}

📋 大师经典语录
────────────────────────────────────────
🧙 巴菲特: "{wisdom.get('quotes', {}).get('buffett', wisdom.get('master_quotes', {}).get('buffett', 'N/A'))}"
🎭 芒格: "{wisdom.get('quotes', {}).get('munger', wisdom.get('master_quotes', {}).get('munger', 'N/A'))}"
📈 林奇: "{wisdom.get('quotes', {}).get('lynch', wisdom.get('master_quotes', {}).get('lynch', 'N/A'))}"
💰 清崎: "{wisdom.get('quotes', {}).get('kiyosaki', wisdom.get('master_quotes', {}).get('kiyosaki', 'N/A'))}"

{'='*60}
                    报告结束
{'='*60}
"""
        return report
    
    def _get_recommendation(self, wisdom: Dict, ai: Dict) -> Dict:
        """获取建议"""
        master_score = wisdom.get('master_wisdom_score', 50)
        ai_prob = ai.get('probability', 0.5)
        
        if master_score > 65 and ai_prob > 0.6:
            action = "STRONG_BUY"
            description = "多维度信号显示上涨概率较高"
        elif master_score > 55 and ai_prob > 0.55:
            action = "BUY"
            description = "多个信号偏多，可考虑买入"
        elif master_score < 35 or ai_prob < 0.4:
            action = "STRONG_SELL"
            description = "风险较高，建议减仓"
        elif master_score < 45 or ai_prob < 0.45:
            action = "SELL"
            description = "部分信号偏空，适当减仓"
        else:
            action = "HOLD"
            description = "信号不明确，建议观望"
        
        return {
            'action': action,
            'description': description,
            'position_size': self._calculate_position_size(master_score, ai_prob)
        }
    
    def _calculate_position_size(self, master_score: float, ai_prob: float) -> str:
        """计算建议仓位"""
        combined = (master_score / 100 * 0.5 + ai_prob * 0.5)
        
        if combined > 0.7:
            return "30-50% (积极配置)"
        elif combined > 0.6:
            return "20-30% (适度配置)"
        elif combined > 0.5:
            return "10-20% (轻仓尝试)"
        elif combined > 0.4:
            return "5-10% (极轻仓)"
        else:
            return "0% (空仓观望)"
    
    def _get_risk_warning(self, technical: Dict) -> List[str]:
        """获取风险警告"""
        warnings = []
        
        # 处理字典格式的volatility
        volatility = technical.get('volatility', 0)
        if isinstance(volatility, dict):
            volatility = volatility.get('value', volatility)
        
        if volatility > 60:
            warnings.append("⚠️ 波动率处于高位，价格可能剧烈波动")
        
        # 处理字典格式的RSI
        rsi = technical.get('rsi', 50)
        if isinstance(rsi, dict):
            rsi = rsi.get('value', rsi)
        
        if rsi > 75:
            warnings.append("⚠️ RSI处于超买区域，可能面临回调")
        elif rsi < 25:
            warnings.append("⚠️ RSI处于超卖区域，可能存在反弹机会")
        
        trend = technical.get('trend', '')
        if 'DOWN' in trend:
            warnings.append("⚠️ 下降趋势中，注意风险控制")
        
        return warnings if warnings else ["✅ 当前无特殊风险警告"]
    
    def _get_next_review_time(self) -> str:
        """获取下次review时间"""
        next_time = datetime.now()
        next_time = next_time.replace(hour=8, minute=0, second=0, microsecond=0)
        if next_time <= datetime.now():
            next_time = next_time.replace(day=next_time.day + 1)
        
        return next_time.strftime("%Y-%m-%d 08:00:00")
    
    def _save_report(self, report: str):
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(REPORTS_DIR, f"report_{timestamp}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已保存至: {filepath}")
    
    def _get_default_data(self) -> Dict:
        """获取默认数据"""
        return {
            'current_price': 45000,
            'price_change_24h': 500,
            'price_change': '+1.2%',
            'volatility': 40,
            'volatility_level': 'MODERATE',
            'trend': 'SIDEWAYS',
            'rsi': {'value': 50, 'signal': 'NEUTRAL'},
            'technical_indicators': {
                'rsi': {'value': 50, 'signal': 'NEUTRAL'},
                'volatility': 40
            },
            'wisdom_data': {
                'buffett_value_score': 50,
                'munger_psychology_score': 50,
                'lynch_growth_score': 50,
                'kiyosaki_risk_score': 50,
                'master_wisdom_score': 50,
                'master_quotes': {
                    'buffett': '价格是你支付的，价值是你得到的',
                    'munger': '反过来想，总是反过来想',
                    'lynch': '投资你了解的东西',
                    'kiyosaki': '让钱为你工作，而不是你为钱工作'
                }
            },
            'ai_data': {
                'direction': 'SIDEWAYS',
                'probability': 0.5,
                'confidence': 0.6,
                'composite_score': 50
            }
        }
    
    def _get_mock_result(self) -> Dict:
        """获取模拟结果"""
        return {
            'summary': '报告生成失败，使用默认摘要',
            'full_report': '详细报告',
            'recommendation': {
                'action': 'HOLD',
                'description': '数据不足，观望为主',
                'position_size': '10-20%'
            },
            'risk_warning': ['✅ 波动率正常'],
            'next_review': datetime.now().strftime("%Y-%m-%d 08:00:00")
        }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    agent = BTCReportAgent()
    result = asyncio.run(agent.run())
    print(json.dumps(result, indent=2, ensure_ascii=False))
