#!/usr/bin/env python3
"""
多智能体协作系统 - API 配置助手
"""

import os
import subprocess

def check_api_key():
    """检查 API Key 是否已配置"""
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    
    if api_key:
        print(f"✅ MINIMAX_API_KEY 已配置")
        print(f"   Key 前缀: {api_key[:10]}...")
        return True
    else:
        print(f"❌ MINIMAX_API_KEY 未配置")
        return False

def setup_api_key():
    """设置 API Key"""
    print("""
🔧 Minimax API Key 配置

步骤：
1. 打开 https://platform.minimaxi.com/
2. 注册/登录账号
3. 在 API Keys 页面创建新 Key
4. 复制 Key 并粘贴到下方

或者设置环境变量：
  export MINIMAX_API_KEY="你的密钥"
  
或者添加到 ~/.bashrc 或 ~/.zshrc：
  echo 'export MINIMAX_API_KEY="你的密钥"' >> ~/.bashrc
""")
    
    key = input("请粘贴你的 Minimax API Key: ").strip()
    
    if key:
        # 临时设置
        os.environ["MINIMAX_API_KEY"] = key
        print(f"\n✅ 已临时设置 API Key")
        print(f"   Key 前缀: {key[:10]}...")
        print(f"\n💡 提示：要永久保存，请添加到 ~/.bashrc 或 ~/.zshrc")
        return key
    else:
        print("❌ 未输入 Key")
        return None

def test_api():
    """测试 API 连接"""
    import requests
    
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        print("❌ 请先配置 API Key")
        return False
    
    try:
        response = requests.post(
            "https://api.minimaxi.com/v1/messages",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "MiniMax-M2.1",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ API 连接成功！")
            return True
        else:
            print(f"❌ API 错误: {response.status_code}")
            print(f"   {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "check":
            check_api_key()
        elif command == "setup":
            setup_api_key()
        elif command == "test":
            test_api()
        else:
            print("用法: python3 setup_api.py [check|setup|test]")
    else:
        print("""
🤖 多智能体系统 - API 配置工具

用法：
  python3 setup_api.py check   # 检查 API Key
  python3 setup_api.py setup   # 设置 API Key
  python3 setup_api.py test    # 测试 API 连接

当前状态：
""")
        check_api_key()
