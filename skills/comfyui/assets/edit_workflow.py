import json

# 读取工作流
with open('~/.openclaw/workspace/skills/comfyui/assets/tmp-workflow.json', 'r') as f:
    workflow = json.load(f)

# 修改节点58的提示词
workflow['58']['inputs']['value'] = '一只边牧狗在月球上弹吉他，月球表面，星空背景，太空环境，高清摄影风格'

# 修改种子
if '57:3' in workflow:
    workflow['57:3']['inputs']['seed'] = 1234567890

# 保存
with open('~/.openclaw/workspace/skills/comfyui/assets/tmp-workflow.json', 'w') as f:
    json.dump(workflow, f, indent=2)

print("✅ 工作流已修改")
print(f"📝 提示词: {workflow['58']['inputs']['value']}")
print(f"🎲 种子: {workflow['57:3']['inputs']['seed']}")
