# BTC交易系统使用指南

*增强版 - 集成自动化和错误追踪*

---

## 🚀 快速开始

### 1. 安装依赖
```bash
cd ~/.openclaw/workspace/btc_trading_system
uv venv
uv pip install ccxt pandas scikit-learn joblib
```

### 2. 运行交易系统
```bash
# 正常模式
cd ~/.openclaw/workspace/btc_trading_system
source .venv/bin/activate
python3 run_btc_trader.py

# 测试模式（不保存）
python3 run_btc_trader.py --test

# 强制刷新数据
python3 run_btc_trader.py --force
```

### 3. 查看结果
```bash
# 查看缓存数据
cat ~/.openclaw/workspace/btc_trading_system/data/cache/market_cache.json

# 查看最新报告
cat ~/.openclaw/workspace/btc_trading_system/reports/*.txt
```

---

## 📁 文件结构

```
btc_trading_system/
├── run_btc_trader.py       # 🚀 增强版运行器（推荐）
├── test_data_connection.py # 🔧 诊断工具
├── btc_multi_agent.py      # 🤖 原版多Agent系统
├── agents/                 # 各专业Agent
│   ├── btc_data.py        # 数据获取
│   ├── btc_wisdom.py      # 投资大师智慧
│   ├── btc_ai.py          # AI预测
│   └── btc_report.py      # 报告生成
├── models/                 # 训练好的模型
│   ├── ridge_model.joblib
│   ├── mlp_model.joblib
│   ├── rf_model.joblib
│   └── gb_model.joblib
├── data/                   # 数据缓存
│   └── cache/
│       └── market_cache.json
├── reports/                # 分析报告
├── OPTIMIZATION_PLAN.md    # 优化计划
└── .venv/                  # Python虚拟环境
```

---

## 📊 当前功能

### ✅ 已实现
- Binance实时数据获取
- 技术指标计算（RSI、MA、波动率）
- 交易信号生成
- 数据缓存
- 错误日志记录
- 定时自动化

### 🔜 开发中
- 多模型集成预测
- 投资大师智慧分析
- 可视化报告
- 邮件/推送通知

---

## 🎯 交易信号

### 信号含义
| 信号 | 说明 | 操作建议 |
|------|------|---------|
| **BUY** | 多个指标看涨 | 考虑买入 |
| **SELL** | 多个指标看跌 | 考虑卖出 |
| **WAIT** | 信号不确定 | 观望 |

### 指标说明
- **RSI**: 相对强弱指数（<30超卖，>70超买）
- **MA7**: 7日移动平均线
- **波动率**: 价格波动范围

---

## ⏰ 定时任务

### 已配置任务
| 任务 | 时间 | 功能 |
|------|------|------|
| BTC每日交易信号 | 08:30 | 获取信号 |

### 手动运行
```bash
# 立即运行
openclaw cron run <任务ID>

# 查看任务列表
openclaw cron list
```

---

## 🐛 故障排除

### 问题1: 连接失败
```bash
# 测试连接
python3 test_data_connection.py

# 检查网络
ping binance.com
```

### 问题2: 缺少依赖
```bash
# 重新安装
uv pip install ccxt pandas scikit-learn joblib
```

### 问题3: 数据异常
```bash
# 强制刷新
python3 run_btc_trader.py --force

# 清理缓存
rm -rf data/cache/*
```

---

## 📈 优化路线图

### v1.0 (已完成)
- ✅ Binance数据获取
- ✅ 技术指标计算
- ✅ 基础交易信号
- ✅ 错误日志记录

### v1.1 (进行中)
- 🔄 多模型集成
- 🔄 投资大师智慧
- 🔄 高级技术指标

### v2.0 (计划中)
- 📅 定时自动运行
- 📊 可视化图表
- 📧 推送通知
- 🌐 Web界面

---

## 🔗 相关文档

- 主文档: `btc_knowledge_base.md`
- 优化计划: `OPTIMIZATION_PLAN.md`
- 错误日志: `~/.openclaw/workspace/error-logs/`
- 定时任务: `CRON_SYSTEM.md`

---

## 💡 使用建议

1. **每日查看**: 每天08:30自动获取信号
2. **结合分析**: RSI <30 + MA7支撑 = 更强BUY信号
3. **风险管理**: 不要全仓梭哈，设置止损
4. **持续优化**: 关注模型准确率变化

---

*创建时间: 2026-02-07*
*版本: v1.0*
