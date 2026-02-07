# 📊 BTC交易系统专用索引

> 最后更新: 2026-02-07
> 包含BTC交易相关的所有模板、代码、文档索引

---

## 📁 文件结构

```
btc_trading_system/
├── 📄 run_btc_trader.py          # 🚀 增强版运行器
├── 📄 btc_ai_system.py           # 🤖 AI综合交易系统
├── 📄 btc_ai_trader_pro.py       # 📈 Pro版交易程序
├── 📄 btc_multi_agent.py         # 🤝 多Agent系统
├── 📄 btc_report.py              # 📋 报告生成器
├── 📄 test_data_connection.py    # 🔧 诊断工具
├── 📄 OPTIMIZATION_PLAN.md       # 📝 优化计划
├── 📄 README.md                  # 📖 使用指南
└── 📁 .venv/                    # 🐍 虚拟环境

~/Desktop/btc_models/
├── btc_ridge_model.pkl          # Ridge模型 (80.4%)
├── btc_mlp_model.pkl            # MLP模型
├── btc_adaboost_model.pkl       # AdaBoost模型
├── btc_gb_model.pkl             # GradientBoosting模型
└── btc_rf_model.pkl             # RandomForest模型
```

---

## 📊 模板索引

### 交易策略模板
| 模板 | 文件 | 说明 |
|------|------|------|
| 趋势跟踪策略 | `BTC_STRATEGY_TEMPLATES.md` | MA200+MACD策略 |
| 均值回归策略 | `BTC_STRATEGY_TEMPLATES.md` | RSI+波动率策略 |
| AI信号策略 | `BTC_STRATEGY_TEMPLATES.md` | 多模型集成 |
| 策略日志 | `BTC_STRATEGY_TEMPLATES.md` | 每日记录 |

### 报告模板
| 模板 | 文件 | 说明 |
|------|------|------|
| BTC每日分析 | `REPORT_TEMPLATES.md` | 完整日报 |
| 项目报告 | `REPORT_TEMPLATES.md` | 周报/月报 |
| 数据分析 | `REPORT_TEMPLATES.md` | 趋势分析 |

---

## 🚀 快速命令

### 运行交易系统
```bash
cd ~/.openclaw/workspace/btc_trading_system
source .venv/bin/activate
python3 run_btc_trader.py           # 正常模式
python3 run_btc_trader.py --force   # 强制刷新

# 测试模式
python3 run_btc_trader.py --test
```

### 模型训练
```bash
cd ~/.openclaw/workspace/btc_trading_system
source .venv/bin/activate
python3 btc_train_model.py           # 训练所有模型
python3 btc_train_model.py --model ridge  # 只训练Ridge
```

### 诊断工具
```bash
cd ~/.openclaw/workspace/btc_trading_system
source .venv/bin/activate
python3 test_data_connection.py      # 测试数据连接
```

---

## 📈 模型性能

| 模型 | 验证准确率 | 权重 | 状态 |
|------|-----------|------|------|
| **Ridge** | **80.4%** | 40% | ⭐ 最佳 |
| AdaBoost | 66.8% | 30% | ✅ |
| MLP | 63.8% | 30% | ✅ |
| GradientBoosting | 50.6% | - | ✅ |
| RandomForest | 46.9% | - | ✅ |

### 特征工程
- **总特征数**: 114个增强特征
- **数据源**: Binance + Coinbase + CryptoCompare
- **训练样本**: 1531条 (2020-08 ~ 2026-02)

---

## 🎯 交易信号

### 当前信号配置
| 指标 | 阈值 | 信号 |
|------|------|------|
| RSI | <30 超卖 | BUY |
| RSI | >70 超买 | SELL |
| AI概率 | >55% | BUY |
| AI概率 | <45% | SELL |
| 贪婪指数 | <30 | BUY |

### 信号集成公式
```
Signal = 0.4 × Ridge + 0.3 × MLP + 0.3 × AdaBoost
```

---

## 📊 交易规则

### 买入条件
| 条件 | 说明 | 优先级 |
|------|------|--------|
| RSI < 30 | 超卖区域 | P1 |
| AI预测 > 55% | 机器学习信号 | P1 |
| 恐惧指数 < 30 | 市场情绪 | P2 |

### 卖出条件
| 条件 | 说明 | 优先级 |
|------|------|--------|
| RSI > 70 | 超买区域 | P1 |
| AI预测 < 45% | 机器学习信号 | P1 |
| 贪婪指数 > 70 | 市场情绪 | P2 |

### 仓位管理
| 场景 | 仓位 |
|------|------|
| 高置信度 (>70%) | 50-100% |
| 中置信度 (50-70%) | 25-50% |
| 低置信度 (<50%) | 0-25% |

---

## 💡 使用指南

### 1. 查看当前信号
```bash
python3 run_btc_trader.py
```

### 2. 生成分析报告
```bash
python3 btc_report.py
```

### 3. 训练新模型
```bash
python3 btc_train_model.py --data fresh
```

### 4. 排查问题
```bash
python3 test_data_connection.py
```

---

## 📝 常见问题

| 问题 | 解决方案 |
|------|----------|
| API连接失败 | 检查网络，查看`test_data_connection.py`输出 |
| 模型加载错误 | 确认模型文件存在`~/Desktop/btc_models/` |
| 信号不一致 | 参考"多维度信号验证" |
| 数据过期 | 运行`run_btc_trader.py --force` |

---

## 📁 相关文件

| 文件 | 位置 | 说明 |
|------|------|------|
| MEMORY.md | `~/.openclaw/workspace/` | 长期记忆（含BTC配置） |
| ERROR_LOGS | `~/.openclaw/workspace/error-logs/` | 错误日志 |
| CRON配置 | `CRON_SYSTEM.md` | 定时任务 |

---

*创建时间: 2026-02-07*
