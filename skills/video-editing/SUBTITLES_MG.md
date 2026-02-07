# 字幕设计与动态图形 (Subtitle Design & Motion Graphics)

## 一、字幕基础

### 1. 字幕类型
```
📝 对白字幕
- 翻译字幕（外语→中文）
- 还原字幕（方言→普通话）
- 注释字幕（专业术语解释）

📢 旁白字幕
- 解说文字
- 画外音
- 内心独白

🎯 效果字幕
- 强调字幕
- 转场字幕
- 节奏字幕

🖼️ 图形字幕
- Logo动画
- 标题设计
- 品牌元素
```

### 2. 字幕规范
```
平台字幕规范
| 平台 | 位置 | 字号 | 字体 |
|------|------|------|------|
| 抖音 | 下1/3 | 40-50 | 思源黑体 |
| B站  | 下1/3 | 36-42 | 思源黑体 |
| YouTube| 下1/3 | 36-48 | 任意清晰字体 |
| 小红书 | 下1/3 | 32-40 | 思源黑体 |

字幕时长公式
- 英文: 每词约0.4秒
- 中文: 每字约0.5秒
- 最短停留: 1.5秒
- 最长单行: 10字以内
```

### 3. FFmpeg字幕处理
```bash
# 硬字幕（烧录进视频）
ffmpeg -i input.mp4 -vf "subtitles=subtitle.srt" output.mp4

# 软字幕（独立文件）
ffmpeg -i input.mp4 -c copy -c:s mov_text subtitle.mp4

# 添加ASS高级字幕
ffmpeg -i input.mp4 -vf "ass=subtitle.ass" output.mp4

# 字幕位置调整
ffmpeg -i input.mp4 \
  -vf "subtitles=subs.srt:force_style='Alignment=2,MarginV=50'" \
  output.mp4

# Alignment: 2=底部居中, 1=左下, 3=右下
# MarginV: 距离底部像素
```

## 二、字幕样式设计

### 1. 基础样式
```python
subtitle_styles = {
    "清晰标准": {
        "字体": "思源黑体 Medium",
        "大小": "42px",
        "颜色": "白色",
        "描边": "黑色 2px",
        "阴影": "黑色 3px 45度",
    },
    
    "电影风格": {
        "字体": "思源宋体",
        "大小": "48px",
        "颜色": "白色",
        "描边": "黑色 3px",
        "阴影": "无",
    },
    
    "可爱风格": {
        "字体": "圆体",
        "大小": "38px",
        "颜色": "粉色 #FF69B4",
        "描边": "白色 1px",
        "阴影": "粉色 50%透明度",
    },
    
    "科技风格": {
        "字体": "DIN Alternate",
        "大小": "44px",
        "颜色": "青色 #00FFFF",
        "描边": "蓝色 2px",
        "发光": "青色 20px",
    },
}
```

### 2. 高级ASS样式
```ass
# subtitle.ass 格式

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text

# 对话样式
Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,20,,{\an5\pos(960,900)\bord2\3c&H000000&\shad2}这是对话内容

# 特效样式
Dialogue: 0,0:00:05.00,0:00:08.00,Style,,0,0,0,,{\an5\pos(960,900)\t(0,500,\fscx120\fscy120)\t(3000,500,\fscx100\fscy100)}强调文字

# 位置标签
\an7  左上 \an8 中上 \an9 右上
\an4  左中 \an5 中中 \an6 右中
\an1  左下 \an2 中下 \an3 右下

# 字体标签
{\fn思源黑体} 字体
{\fs42} 大小
{\c&H00FFFF&} 颜色(BGR)
{\bord3} 描边
{\shad2} 阴影

# 动画标签
{\t(开始,持续,\fscx120\fscy120)} 缩放
{\t(开始,持续,\alpha&HFF&)} 透明度
{\move(960,900,960,850,0,3000)} 移动
{\pos(960,900)} 位置
\fad(500,500)} 淡入淡出
```

### 3. Python批量生成字幕
```python
import ass
from ass import document

def create_subtitle_file(script, output_path):
    """生成ASS字幕文件"""
    doc = document.Document()
    
    # 添加样式
    doc.styles.add(
        name="Default",
        fontname="思源黑体",
        fontsize=42,
        primarycolor=ass.Color(255, 255, 255, 0),
        outlinecolor=ass.Color(0, 0, 0, 0),
        shadowcolor=ass.Color(0, 0, 0, 0),
        outline=2,
        shadow=3,
        alignment=2,
    )
    
    doc.styles.add(
        name="Emphasis",
        fontname="思源黑体",
        fontsize=52,
        primarycolor=ass.Color(255, 105, 180, 0),
        outlinecolor=ass.Color(0, 0, 0, 0),
        outline=2,
        shadow=2,
        alignment=2,
    )
    
    # 添加对话
    for i, line in enumerate(script):
        start, end, text, style = line
        doc.events.add(
            start=start,
            end=end,
            style=style,
            text=text,
        )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(str(doc))

# 使用示例
script = [
    ("0:00:01.00", "0:00:04.00", "这是第一句台词", "Default"),
    ("0:00:04.50", "0:00:07.00", "这是第二句台词", "Default"),
    ("0:00:07.50", "0:00:10.00", "重点强调内容！", "Emphasis"),
]

create_subtitle_file(script, "video.ass")
```

## 三、动态图形设计

### 1. 标题动画设计
```bash
# 使用FFmpeg创建动态标题
ffmpeg -f lavfi -i "color=s=1920x1080:c=black:d=5" -f lavfi -i "drawtext=text='HELLO':fontsize=80:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0.5,4.5)':alpha='if(between(t,0.5,1.0),t-0.5,if(between(t,1.0,4.0),0.5,0.5-(t-4.0))'" \
  -c:v libx264 -t 5 title_anim.mp4

# 进阶：缩放+旋转
ffmpeg -f lavfi -i "color=s=1920x1080:c=black:d=5" \
  -f lavfi -i "drawtext=text='HELLO':fontsize=120:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,5)':expansion=none" \
  -vf "zoompan=z='min(zoom+0.0015*on,1.5)':d=5000:s=1920x1080" \
  -c:v libx264 -t 5 title_zoom.mp4
```

### 2. Logo动画
```bash
# Logo入场动画
ffmpeg -i logo.png -loop 1 \
  -vf "fade=t=in:st=0:d=1,fade=t=out:st=4:d=1,format=rgba" \
  -t 5 logo_anim.mp4

# 打字机效果
ffmpeg -f lavfi -i "color=s=800x200:c=black:d=5" \
  -vf "drawtext=text='HELLO WORLD':fontsize=60:fontcolor=white:x=50:y=80:enable='between(t,0,3)':expansion=none" \
  -c:v libx264 -t 5 typewriter.mp4

# 闪光过渡
ffmpeg -i clip1.mp4 -i clip2.mp4 -i light.png \
  -filter_complex "[2:v]scale=1920:1080[light];[0:v][light][1:v]xfade=transition=rectcrop:duration=1:offset=3[out]" \
  -map "[out]" -map 1:a \
  light_transition.mp4
```

### 3. 图形转场
```bash
# 溶解转场
ffmpeg -i a.mp4 -i b.mp4 \
  -filter_complex "[0:v][1:v]xfade=transition=dissolve:duration=1:offset=0[out]" \
  -map "[out]" -map 0:a \
  dissolve.mp4

# 滑动转场
ffmpeg -i a.mp4 -i b.mp4 \
  -filter_complex "[0:v][1:v]xfade=transition=slideleft:duration=1:offset=0[out]" \
  -map "[out]" -map 0:a \
  slide.mp4

# 缩放转场
ffmpeg -i a.mp4 -i b.mp4 \
  -filter_complex "[0:v][1:v]xfade=transition=zoomin:duration=1:offset=0[out]" \
  -map "[out]" -map 0:a \
  zoom.mp4

# 百叶窗
ffmpeg -i a.mp4 -i b.mp4 \
  -filter_complex "[0:v][1:v]xfade=transition=circleopen:duration=1:offset=0[out]" \
  -map "[out]" -map 0:a \
  circle.mp4
```

## 四、品牌图形元素

### 1. 角标设计
```bash
# 右下角角标
ffmpeg -i main.mp4 -i watermark.png \
  -filter_complex "[1:v]scale=80:-1,format=rgba,loop=1:size=1:rate=1[wm];[0:v][wm]overlay=W-w-20:H-h-20:enable='between(t,0,60)'[out]" \
  -map "[out]" -map 0:a \
  with_watermark.mp4
```

### 2. 片头片尾
```bash
# 5秒片头
ffmpeg -f lavfi -i "color=s=1920x1080:c=#1a1a1a:d=5" \
  -f lavfi -i "drawtext=text='CHANNEL NAME':fontsize=64:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-100:enable='between(t,0,5)'" \
  -f lavfi -i "drawtext=text='2024':fontsize=128:fontcolor=#FFD700:x=(w-text_w)/2:y=(h-text_h)/2+50:enable='between(t,1,5)'" \
  -c:v libx264 intro.mp4

# 片尾致谢
ffmpeg -f lavfi -i "color=s=1920x1080:c=black:d=5" \
  -f lavfi -i "drawtext=text='THANKS FOR WATCHING':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0.5,4.5)':alpha='if(btw(t,0.5,1.0),t-0.5,if(btw(t,3.5,4.5),0.5-(t-3.5),0.5))'" \
  -f lavfi -i "drawtext=text='SUBSCRIBE':fontsize=48:fontcolor=#FF0000:x=(w-text_w)/2:y=(h-text_h)/2+150:enable='between(t,2,5)'" \
  -c:v libx264 outro.mp4
```

### 3. 进度条
```bash
# 底部进度条
ffmpeg -i video.mp4 \
  -vf "drawbox=x=0:y=H-h-10:w=iw*t/duration:h=10:color=#FF0000@0.8:t=fill" \
  progress_bar.mp4
```

## 五、动态信息图

### 1. 数据动画
```python
# 简单柱状图动画
import subprocess

def create_bar_chart(data, output, duration=5):
    """创建柱状图动画"""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi', '-i', f'color=s=800x400:c=black:d={duration}',
    ]
    
    for i, (label, value, color) in enumerate(data):
        # 计算柱状图
        h = int(400 * value / 100)
        y = 400 - h
        x = 100 + i * 150
        
        cmd.extend([
            '-vf', f"drawbox=x={x}:y={y}:w=100:h={h}:color={color}:t=fill:enable='between(t,{i*0.5},{duration})'"
        ])
    
    cmd.extend(['-c:v', 'libx264', output])
    subprocess.run(cmd)
```

### 2. 计数器动画
```bash
# 数字滚动动画
ffmpeg -f lavfi -i "color=s=1920x1080:c=black:d=10" \
  -vf "drawtext=text='%{eif\\:1+trunc(t*10)\\::d}:fontsize=200:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,10)'" \
  counter.mp4
```

### 3. 地图动画
```bash
# 简单路径动画
ffmpeg -i map.png \
  -vf "drawbox=x=100:y=200:w=50:h=50:color=red:t=fill:enable='between(t,1,3)':xy='if(between(t,1,3),100+1000*t/2,100)',format=rgba" \
  -c:v libx264 \
  map_anim.mp4
```

## 六、特效字幕

### 1. 打字机效果
```python
def typewriter_text(text, duration=5):
    """打字机效果的字幕文件"""
    ass_content = f"""[Script Info]
ScriptType: v4.00+
Collisions: Normal

[V4+ Styles]
Style: Default,思源黑体,42,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,2,1,1,20,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    char_time = duration / len(text)
    for i, char in enumerate(text):
        start = i * char_time
        end = start + char_time * 1.5
        ass_content += f"Dialogue: 0,0:00:{start:.2f},0:00:{end:.2f},Default,,0,0,0,,{{\\an7\\pos(100,900)}}{char}\n"
    
    return ass_content
```

### 2. 弹跳文字
```ass
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\an5\pos(960,900)\t(0,300,\fscx80\fscy80)\t(300,500,\fscx120\fscy120)\t(500,800,\fscx100\fscy100)}弹跳文字！
```

### 3. 霓虹灯效果
```ass
Dialogue: 0,0:00:01.00,0:00:04.00,Neon,,0,0,0,,{\an5\pos(960,500)\blur3\fscx105\fscy105\1c&H00FFFF&}霓虹灯效果
```

## 七、色彩与字体

### 1. 常用配色
```python
color_palettes = {
    "科技蓝": {
        "主色": "#007AFF",
        "辅色": "#5AC8FA",
        "背景": "#1C1C1E",
        "文字": "#FFFFFF",
        "强调": "#FFD60A",
    },
    
    "温暖橙": {
        "主色": "#FF9500",
        "辅色": "#FF3B30",
        "背景": "#1C1C1E",
        "文字": "#FFFFFF",
        "强调": "#34C759",
    },
    
    "极简白": {
        "主色": "#000000",
        "辅色": "#333333",
        "背景": "#FFFFFF",
        "文字": "#000000",
        "强调": "#007AFF",
    },
    
    "赛博朋克": {
        "主色": "#FF00FF",
        "辅色": "#00FFFF",
        "背景": "#0D0221",
        "文字": "#FFFFFF",
        "强调": "#FFFF00",
    },
}
```

### 2. 推荐字体
```
中文字体
- 思源黑体 (Source Han Sans) - 通用
- 思源宋体 (Source Han Serif) - 正式
- 阿里巴巴普惠体 - 商用免费
- 站酷系列 - 免费商用
- 优设标题黑 - 标题

英文字体
- Roboto - 通用
- Montserrat - 现代
- Playfair Display - 优雅
- Bebas Neue - 标题
- Open Sans - 正文

数字字体
- DIN Alternate - 科技感
- Bebas Neue Pro - 数据展示
- SF Mono - 代码/技术
```

## 八、实战案例

### 案例1：产品评测字幕
```python
product_review_style = {
    "开头": {
        "特效": "缩放+发光",
        "停留": "3秒",
        "文字": "产品名称",
    },
    
    "价格": {
        "特效": "打字机",
        "颜色": "#FFD700",
        "停留": "2秒",
    },
    
    "优点": {
        "特效": "绿色勾选",
        "颜色": "#34C759",
    },
    
    "缺点": {
        "特效": "红色叉号",
        "颜色": "#FF3B30",
    },
    
    "总结": {
        "特效": "放大",
        "停留": "3秒",
        "颜色": "#007AFF",
    },
}
```

### 案例2：知识科普字幕
```python
education_style = {
    "关键概念": {
        "字体": "思源黑体 Heavy",
        "大小": "48px",
        "颜色": "#007AFF",
        "特效": "下划线+放大",
    },
    
    "数字数据": {
        "字体": "DIN Alternate",
        "大小": "72px",
        "颜色": "#FFD700",
        "特效": "缩放弹跳",
    },
    
    "引用内容": {
        "字体": "思源宋体",
        "样式": "斜体",
        "背景": "半透明黑色",
        "停留": "4秒",
    },
    
    "步骤": {
        "编号": "① ② ③",
        "颜色": "渐变绿→蓝",
    },
}
```

### 案例3：Vlog字幕风格
```python
vlog_style = {
    "日期地点": {
        "位置": "左上角",
        "字体": "圆体",
        "大小": "32px",
        "颜色": "白色+阴影",
    },
    
    "心情标注": {
        "位置": "右上角",
        "字体": "手写体",
        "大小": "40px",
        "颜色": "粉色",
        "特效": "轻微抖动",
    },
    
    "对话气泡": {
        "样式": "圆角矩形",
        "背景": "半透明白",
        "描边": "卡通风格",
    },
    
    "时间码": {
        "格式": "10:23",
        "位置": "右下角",
        "字体": "DIN",
        "颜色": "白色50%",
    },
}
```

---

**学习时间**: 2026-02-08 07:25
**主题**: 字幕设计与动态图形
**技能等级**: 动态设计专家 📝🎨
