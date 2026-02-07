#!/bin/bash
# 🚀 OpenClaw AI 助手 - 快速启动脚本

echo "🤖 OpenClaw AI Assistant 启动器"
echo "================================"

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "Python版本: $python_version"

# 创建必要目录
workspace="/Users/wangshice/.openclaw/workspace"
mkdir -p "$workspace/knowledge"
mkdir -p "$workspace/task_results"

# 菜单
echo ""
echo "请选择操作:"
echo "1. 🤖 启动AI助手 (交互模式)"
echo "2. 📊 执行数据分析"
echo "3. 🤖 生成机器学习代码"
echo "4. 🧠 生成深度学习代码"
echo "5. 📝 生成NLP代码"
echo "6. 💻 编写自定义代码"
echo "7. 📖 学习新知识"
echo "8. 🔄 更新知识库"
echo "9. 📊 生成状态报告"
echo "10. 🔧 调试代码"
echo "11. 🚀 运行所有测试"
echo "0. 退出"
echo ""
read -p "请输入选项 (0-11): " choice

case $choice in
    1)
        echo "🚀 启动AI助手交互模式..."
        python3 "$workspace/ai_assistant.py"
        ;;
    2)
        read -p "请输入数据文件路径: " data_file
        python3 -c "
from task_executor import TaskExecutor
executor = TaskExecutor()
result = executor.execute('数据分析', {'file': '$data_file'})
print('✅ 分析完成!')
print('输出:', result)
"
        ;;
    3)
        read -p "请输入模型类型 (random_forest/xgboost/svm): " model
        python3 -c "
from task_executor import TaskExecutor
executor = TaskExecutor()
result = executor.execute('机器学习', {'model': '$model'})
print('✅ 代码已生成!')
"
        ;;
    4)
        python3 -c "
from task_executor import TaskExecutor
executor = TaskExecutor()
result = executor.execute('深度学习', {'framework': 'pytorch'})
print('✅ PyTorch代码已生成!')
"
        ;;
    5)
        read -p "请输入NLP任务类型 (text_classification/ner/summarization): " nlp_type
        python3 -c "
from task_executor import TaskExecutor
executor = TaskExecutor()
result = executor.execute('NLP任务', {'type': '$nlp_type'})
print('✅ NLP代码已生成!')
"
        ;;
    6)
        read -p "请输入代码语言 (python/javascript/java/cpp): " lang
        read -p "请输入代码描述: " desc
        python3 -c "
from task_executor import TaskExecutor
executor = TaskExecutor()
result = executor.execute('编写代码', {'language': '$lang', 'description': '$desc'})
print('✅ 代码已生成!')
print('文件位置:', result['result']['outputs'][0]['path'])
"
        ;;
    7)
        read -p "请输入要学习的主题: " topic
        python3 -c "
from task_executor import TaskExecutor
executor = TaskExecutor()
result = executor.execute('学习新知识', {'topic': '$topic', 'source': 'manual'})
print('✅ 知识已保存!')
"
        ;;
    8)
        python3 -c "
from task_executor import TaskExecutor
executor = TaskExecutor()
result = executor.execute('更新知识库', {})
print('✅ 知识库已更新!')
"
        ;;
    9)
        python3 -c "
from task_executor import TaskExecutor
executor = TaskExecutor()
result = executor.execute('生成报告', {'type': 'status'})
print('✅ 报告已生成!')
"
        ;;
    10)
        python3 -c "
from task_executor import TaskExecutor
executor = TaskExecutor()
result = executor.execute('代码调试', {})
print('✅ 调试脚本已生成!')
"
        ;;
    11)
        echo "🚀 运行所有测试..."
        python3 "$workspace/task_executor.py"
        ;;
    0)
        echo "👋 再见!"
        exit 0
        ;;
    *)
        echo "⚠️ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "================================"
echo "✨ 操作完成!"
