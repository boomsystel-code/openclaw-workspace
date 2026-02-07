#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC AI Trader - Multi-Agent Architecture
=========================================
主调度器：协调各专业Agent完成BTC交易分析

Agents:
- btc_main: 总调度，协调各代理
- btc_data: 数据获取、清洗、计算技术指标
- btc_wisdom: 投资大师智慧分析（巴菲特、芒格、林奇、清崎）
- btc_ai: AI预测模型（融合大师智慧特征）
- btc_report: 汇总报告生成

Author: AI Trading System
Date: 2024-02-06
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


class AgentStatus(Enum):
    """Agent状态"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING = "waiting"


@dataclass
class AgentResult:
    """Agent执行结果"""
    agent_name: str
    status: AgentStatus
    data: Dict[str, Any] = None
    error: str = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'agent_name': self.agent_name,
            'status': self.status.value,
            'data': self.data,
            'error': self.error,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp
        }


class AgentWorkspace:
    """Agent独立工作空间"""
    
    def __init__(self, name: str, base_dir: str):
        self.name = name
        self.base_dir = base_dir
        self.workspace_dir = os.path.join(base_dir, f"workspace_{name}")
        self.data_dir = os.path.join(self.workspace_dir, 'data')
        self.config_file = os.path.join(self.workspace_dir, 'config.json')
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_config(self) -> Dict:
        """获取配置"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_config(self, config: Dict):
        """保存配置"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def get_data_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)


class BTCMainAgent:
    """
    BTC主调度Agent
    
    职责：
    - 协调各专业Agent的并行运行
    - 管理整体工作流程
    - 汇总各Agent结果生成最终决策
    """
    
    def __init__(self):
        self.name = "btc_main"
        self.status = AgentStatus.IDLE
        self.workspace = AgentWorkspace(self.name, BASE_DIR)
        self.agents: Dict[str, Any] = {}
        self.results: Dict[str, AgentResult] = {}
        self.lock = threading.Lock()
        
        # 初始化子Agent
        self._init_agents()
        
    def _init_agents(self):
        """初始化所有子Agent"""
        try:
            # 使用绝对导入
            from agents.btc_data import BTCDataAgent
            from agents.btc_wisdom import BTCWisdomAgent
            from agents.btc_ai import BTCAIAgent
            from agents.btc_report import BTCReportAgent
            
            self.agents['btc_data'] = BTCDataAgent()
            self.agents['btc_wisdom'] = BTCWisdomAgent()
            self.agents['btc_ai'] = BTCAIAgent()
            self.agents['btc_report'] = BTCReportAgent()
            
            logger.info("所有子Agent初始化完成")
        except ImportError as e:
            logger.warning(f"子Agent导入失败: {e}，将使用模拟模式")
            self.agents = {}
    
    async def run(self, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行主调度流程
        
        Args:
            market_data: 初始市场数据（可选）
            
        Returns:
            最终交易决策
        """
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        
        try:
            logger.info("=" * 60)
            logger.info("BTC AI Trader - Multi-Agent 系统启动")
            logger.info("=" * 60)
            
            # 并行运行各专业Agent
            results = await self._run_parallel_agents(market_data)
            
            # 汇总结果
            final_decision = self._aggregate_results(results)
            
            # 保存执行结果
            self._save_execution_results(results, final_decision)
            
            execution_time = time.time() - start_time
            logger.info(f"主调度执行完成，耗时: {execution_time:.2f}秒")
            
            self.status = AgentStatus.COMPLETED
            
            return {
                'status': 'success',
                'decision': final_decision,
                'agent_results': {k: v.to_dict() for k, v in results.items()},
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"主调度执行错误: {e}")
            self.status = AgentStatus.ERROR
            return {
                'status': 'error',
                'error': str(e),
                'decision': None
            }
    
    def _dict_to_agent_result(self, agent_name: str, result: Any) -> AgentResult:
        """将Agent返回的dict转换为AgentResult"""
        if isinstance(result, AgentResult):
            return result
        elif isinstance(result, dict):
            if 'status' in result:
                # 已经是类似结构
                return AgentResult(
                    agent_name=agent_name,
                    status=AgentStatus.COMPLETED,
                    data=result.get('data'),
                    execution_time=result.get('execution_time', 0)
                )
            else:
                # 直接是数据
                return AgentResult(
                    agent_name=agent_name,
                    status=AgentStatus.COMPLETED,
                    data=result
                )
        else:
            return AgentResult(
                agent_name=agent_name,
                status=AgentStatus.ERROR,
                error=f"未知返回类型: {type(result)}"
            )
    
    async def _run_parallel_agents(self, initial_data: Dict[str, Any]) -> Dict[str, AgentResult]:
        """并行运行所有子Agent"""
        results = {}
        
        # btc_data: 数据获取和预处理（最先执行，供给其他Agent）
        logger.info("[btc_main] 启动 btc_data 获取市场数据...")
        data_result_raw = await self.agents['btc_data'].run(initial_data)
        data_result = self._dict_to_agent_result('btc_data', data_result_raw)
        results['btc_data'] = data_result
        
        if data_result.status != AgentStatus.COMPLETED:
            logger.warning("btc_data 执行失败，将使用模拟数据")
        
        # 提取数据供其他Agent使用
        market_data = data_result.data if data_result.data else initial_data
        
        # 并行运行wisdom、ai、report
        logger.info("[btc_main] 并行启动 wisdom、ai、report...")
        
        # 先运行wisdom和ai，再传递结果给report
        wisdom_raw, ai_raw = await asyncio.gather(
            self.agents['btc_wisdom'].run(market_data),
            self.agents['btc_ai'].run(market_data)
        )
        
        # 提取wisdom和ai数据传递给report
        wisdom_result = self._dict_to_agent_result('btc_wisdom', wisdom_raw)
        ai_result = self._dict_to_agent_result('btc_ai', ai_raw)
        
        results['btc_wisdom'] = wisdom_result
        results['btc_ai'] = ai_result
        
        wisdom_data = wisdom_result.data if wisdom_result.status == AgentStatus.COMPLETED else {}
        ai_data = ai_result.data if ai_result.status == AgentStatus.COMPLETED else {}
        
        # 运行report，传递所有数据
        report_raw = await self.agents['btc_report'].run(market_data, wisdom_data, ai_data)
        results['btc_report'] = self._dict_to_agent_result('btc_report', report_raw)
        
        return results
    
    def _aggregate_results(self, results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """汇总各Agent结果生成最终决策"""
        
        # 提取各Agent的关键信号
        market_signal = results.get('btc_data', AgentResult('btc_data', AgentStatus.ERROR)).data or {}
        wisdom_signal = results.get('btc_wisdom', AgentResult('btc_wisdom', AgentStatus.ERROR)).data or {}
        ai_prediction = results.get('btc_ai', AgentResult('btc_ai', AgentStatus.ERROR)).data or {}
        report_data = results.get('btc_report', AgentResult('btc_report', AgentStatus.ERROR)).data or {}
        
        # 计算综合得分
        final_score = self._calculate_final_score(
            market_signal, wisdom_signal, ai_prediction
        )
        
        # 生成决策
        decision = {
            'timestamp': datetime.now().isoformat(),
            'action': self._determine_action(final_score),
            'confidence': self._calculate_confidence(wisdom_signal, ai_prediction),
            'score': final_score,
            'market_analysis': {
                'price': market_signal.get('current_price'),
                'trend': market_signal.get('trend'),
                'volatility': market_signal.get('volatility')
            },
            'master_wisdom': {
                'buffett_score': wisdom_signal.get('buffett_value_score'),
                'munger_score': wisdom_signal.get('munger_psychology_score'),
                'lynch_score': wisdom_signal.get('lynch_growth_score'),
                'kiyosaki_score': wisdom_signal.get('kiyosaki_risk_score')
            },
            'ai_prediction': {
                'direction': ai_prediction.get('direction'),
                'probability': ai_prediction.get('probability'),
                'features': ai_prediction.get('master_features')
            },
            'report_summary': report_data.get('summary'),
            'detailed_report': report_data.get('full_report')
        }
        
        return decision
    
    def _calculate_final_score(self, market: Dict, wisdom: Dict, ai: Dict) -> float:
        """计算综合得分（0-100）"""
        scores = []
        weights = []
        
        # 市场技术面 (权重30%)
        if market.get('technical_score'):
            scores.append(market['technical_score'])
            weights.append(0.30)
        
        # 大师智慧 (权重35%)
        master_scores = [
            wisdom.get('buffett_value_score', 50),
            wisdom.get('munger_psychology_score', 50),
            wisdom.get('lynch_growth_score', 50),
            wisdom.get('kiyosaki_risk_score', 50)
        ]
        if any(s is not None for s in master_scores):
            avg_master = sum(s for s in master_scores if s is not None) / len([s for s in master_scores if s is not None])
            scores.append(avg_master)
            weights.append(0.35)
        
        # AI预测 (权重35%)
        if ai.get('composite_score'):
            scores.append(ai['composite_score'])
            weights.append(0.35)
        
        if not scores:
            return 50.0
        
        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 加权平均
        final_score = sum(s * w for s, w in zip(scores, normalized_weights))
        return max(0, min(100, final_score))
    
    def _determine_action(self, score: float) -> str:
        """根据得分确定操作"""
        if score >= 75:
            return "STRONG_BUY"
        elif score >= 60:
            return "BUY"
        elif score >= 45:
            return "HOLD"
        elif score >= 30:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _calculate_confidence(self, wisdom: Dict, ai: Dict) -> float:
        """计算决策置信度"""
        confidence_factors = []
        
        # 大师信号一致性
        master_scores = [
            wisdom.get('buffett_value_score'),
            wisdom.get('munger_psychology_score'),
            wisdom.get('lynch_growth_score'),
            wisdom.get('kiyosaki_risk_score')
        ]
        if all(s is not None for s in master_scores):
            variance = sum((s - 50)**2 for s in master_scores) / 4
            consistency = max(0, 100 - variance)
            confidence_factors.append(consistency)
        
        # AI预测置信度
        if ai.get('probability'):
            confidence_factors.append(ai['probability'] * 100)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 50.0
    
    def _save_execution_results(self, results: Dict[str, AgentResult], decision: Dict):
        """保存执行结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存汇总结果
        summary = {
            'timestamp': timestamp,
            'decision': decision,
            'agent_results': {k: v.to_dict() for k, v in results.items()}
        }
        
        output_file = os.path.join(REPORTS_DIR, f"trading_decision_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"决策结果已保存至: {output_file}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'main_agent': {
                'name': self.name,
                'status': self.status.value
            },
            'agents': {
                name: {
                    'status': agent.status.value if hasattr(agent, 'status') else 'unknown'
                }
                for name, agent in self.agents.items()
            }
        }


async def main():
    """主入口函数"""
    # 检查是否已有市场数据
    market_data = None
    
    # 如果有命令行参数，使用提供的市场数据
    if len(sys.argv) > 1:
        try:
            market_data = json.loads(sys.argv[1])
            logger.info("使用命令行提供的市场数据")
        except json.JSONDecodeError:
            logger.warning("命令行参数解析失败，将自动获取数据")
    
    # 初始化主Agent
    main_agent = BTCMainAgent()
    
    # 执行交易分析
    result = await main_agent.run(market_data)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("BTC AI Trader - 最终交易决策")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
