# FFmpeg视频剪辑高级技巧 (2026-02-08学习)

## 一、添加背景音乐

### 方法1：混音（视频原声 + 背景音乐）
```bash
ffmpeg -i video.mp4 -i background_music.mp3 \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=first:weights=1 0.5[aout]" \
  -map 0:v -map "[aout]" \
  -c:v copy -c:a aac \
  output.mp4
```

### 方法2：背景音乐淡入淡出
```bash
# 生成淡入淡出的背景音乐
ffmpeg -i background_music.mp3 \
  -af "afade=t=in:st=0:d=3,afade=t=out:st=25:d=3" \
  music_faded.mp3

# 混合
ffmpeg -i video.mp4 -i music_faded.mp3 \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=first[aout]" \
  -map 0:v -map "[aout]" \
  output.mp4
```

## 二、添加字幕

### 添加SRT字幕
```bash
ffmpeg -i video.mp4 -vf subtitles=subtitle.srt \
  -c:a copy output_with_subs.mp4
```

### 添加文字水印
```bash
ffmpeg -i video.mp4 \
  -vf "drawtext=text='@你的名字':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=h-40" \
  -c:a copy output_with_watermark.mp4
```

## 三、画中画（画中画效果）
```bash
ffmpeg -i main_video.mp4 -i logo.png \
  -filter_complex "[1:v]scale=100:-1[logo];[0:v][logo]overlay=10:10[out]" \
  -map "[out]" \
  -c:a copy output_pip.mp4
```

## 四、倍速播放
```bash
# 2倍速
ffmpeg -i video.mp4 -filter_complex "[0:v]setpts=0.5*PTS[v];[0:a]atempo=2.0[a]" \
  -map "[v]" -map "[a]" output_2x.mp4

# 0.5倍速（慢动作）
ffmpeg -i video.mp4 -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" \
  -map "[v]" -map "[a]" output_slow.mp4
```

## 五、画面特效

### 轻微模糊背景（人像突出）
```bash
ffmpeg -i video.mp4 \
  -vf "eq=brightness=0.05:contrast=1.1,saturation=1.1" \
  -c:v libx264 -crf 22 \
  output_bright.mp4
```

### 添加晕影效果（暗角）
```bash
ffmpeg -i video.mp4 \
  -vf "vignette=PI/4" \
  -c:a copy output_vignette.mp4
```

## 六、常用参数速查

| 参数 | 说明 | 示例 |
|------|------|------|
| `-i` | 输入文件 | `-i video.mp4` |
| `-c:v` | 视频编码器 | `-c:v libx264` |
| `-c:a` | 音频编码器 | `-c:a aac` |
| `-crf` | 质量（0-51，越低越好） | `-crf 22` |
| `-preset` | 编码速度 | `-preset fast` |
| `-vf` | 视频滤镜 | `-vf "eq=brightness=0.05"` |
| `-af` | 音频滤镜 | `-af "afade=t=in:d=2"` |
| `-map` | 选择流 | `-map 0:v -map 0:a` |
| `-y` | 覆盖输出 | `-y output.mp4` |

## 七、给奥利奥视频加欢快背景音乐

```bash
# 1. 下载免费背景音乐
yt-dlp -x --audio-format mp3 "https://music.youtube.com/watch?v=XXXXX"

# 2. 调整音量
ffmpeg -i background.mp3 -af "volume=0.3" bg_music.mp3

# 3. 混合视频+背景音乐
ffmpeg -i oreo_snow.mp4 -i bg_music.mp3 \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=first:weights=1 0.3[aout]" \
  -map 0:v -map "[aout]" \
  -c:v copy -c:a aac \
  oreo_snow_with_music.mp4
```

## 八、批量处理

```bash
# 批量优化视频
for f in *.mp4; do
  ffmpeg -i "$f" \
    -vf "eq=brightness=0.02:contrast=1.05" \
    -c:v libx264 -crf 24 -preset fast \
    -c:a copy \
    "optimized_$f"
done
```

---

## 九、颜色校正与调色

### 基础调色（亮度、对比度、饱和度）
```bash
# 轻微提亮+增强对比
ffmpeg -i video.mp4 \
  -vf "eq=brightness=0.05:contrast=1.1:saturation=1.2" \
  -c:v libx264 -crf 22 \
  output.mp4

# 冷色调（蓝调）
ffmpeg -i video.mp4 \
  -vf "colorbalance=rs=0.1:gs=-0.05:bs=0.2" \
  output.mp4

# 暖色调
ffmpeg -i video.mp4 \
  -vf "colorbalance=rs0.1:gs=0.=-05:bs=-0.15" \
  output.mp4
```

### LUT调色（电影感）
```bash
# 应用LUT文件
ffmpeg -i video.mp4 \
  -vf "lut3d=file=cinematic.cube" \
  -c:a copy \
  output.mp4
```

### 曲线调色
```bash
# 使用曲线调整颜色
ffmpeg -i video.mp4 \
  -vf "curves=vintage" \
  output.mp4
# 可用预设: default, negative, solarize, vintage, crossprocess, dramatic
```

## 十、绿幕抠像（换背景）

```bash
# 绿幕抠像
ffmpeg -i video_with_green_screen.mp4 -i background.jpg \
  -filter_complex "[0:v]chromakey=0x00ff00:0.1:0.2[fg];[bg][fg]overlay[out]" \
  -map "[out]" -map 0:a \
  -c:a copy \
  output.mp4
# 参数: 绿幕颜色 / 相似度容差 / 平滑度

# 蓝幕抠像
ffmpeg -i video.mp4 -i bg.jpg \
  -filter_complex "[0:v]chromakey=0x0000ff:0.1:0.2[fg];[bg][fg]overlay[out]" \
  -map "[out]" -c:a copy \
  output.mp4
```

## 十一、分屏/多画面效果

### 两分屏
```bash
ffmpeg -i left.mp4 -i right.mp4 \
  -filter_complex "[0:v][1:v]hstack[out]" \
  -map "[out]" -map 0:a \
  -c:a copy \
  output.mp4
```

### 四分屏
```bash
ffmpeg -i tl.mp4 -i tr.mp4 -i bl.mp4 -i br.mp4 \
  -filter_complex "[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]vstack[out]" \
  -map "[out]" -c:a copy \
  output.mp4
```

### 画中画（主视频+小窗口）
```bash
ffmpeg -i main.mp4 -i pip.mp4 \
  -filter_complex "[1:v]scale=200:-1[pip];[0:v][pip]overlay=W-w-10:10[out]" \
  -map "[out]" -map 0:a \
  output.mp4
```

## 十二、转场效果

### 淡入淡出
```bash
# 视频淡入淡出
ffmpeg -i video.mp4 \
  -vf "fade=t=in:st=0:d=2,fade=t=out:st=28:d=2" \
  -af "afade=t=in:st=0:d=2,afade=t=out:st=28:d=2" \
  output.mp4

# 黑场过渡
ffmpeg -i part1.mp4 -i part2.mp4 \
  -filter_complex "[0:v]fade=t=out:st=5:d=1[ v0];[1:v]fade=t=in:st=0:d=1[ v1];[ v0][ v1]xfade=transition=fade:duration=1:offset=5[out]" \
  -map "[out]" -map 0:a -map 1:a \
  output.mp4
```

## 十三、视频修复

### 去抖动（手持拍摄稳定）
```bash
ffmpeg -i shaky_video.mp4 \
  -vf "deshake" \
  output.mp4
```

### 去噪
```bash
# 轻度降噪
ffmpeg -i video.mp4 \
  -vf "hqdn3d=1.5:1.5:3:3" \
  output.mp4

# 强力降噪
ffmpeg -i video.mp4 \
  -vf "nlmeans=s=10:p=7:r=7" \
  output.mp4
```

### 去块（修复压缩伪影）
```bash
ffmpeg -i compressed.mp4 \
  -vf "deblock=filter=strong:block=8:thresh=20" \
  output.mp4
```

## 十四、音频处理

### 降噪
```bash
ffmpeg -i video.mp4 \
  -af "afftdn=nf=-25" \
  -c:v copy \
  output.mp4
```

### 音量标准化
```bash
# 标准化到目标音量
ffmpeg -i video.mp4 \
  -af "loudnorm=I=-16:TP=-1.5:LRA=11" \
  -c:v copy \
  output.mp4

# 静音检测
ffmpeg -i video.mp4 \
  -af "silencedetect=noise=-50dB:d=2" \
  -f null - \
  2>&1 | grep silence
```

### 音频特效
```bash
# 回声
ffmpeg -i video.mp4 \
  -af "aecho=0.8:0.9:500:0.3" \
  output.mp4

# 变速不变调
ffmpeg -i video.mp4 \
  -filter_complex "[0:v]setpts=0.5*PTS[v];[0:a]atempo=2.0[a]" \
  -map "[v]" -map "[a]" \
  output.mp4
```

## 十五、帧处理

### 抽帧/跳帧
```bash
# 每2秒抽1帧
ffmpeg -i video.mp4 \
  -vf "fps=0.5" \
  output_%04d.jpg

# 提取关键帧
ffmpeg -i video.mp4 -vf "select=eq(pict_type\,I)" \
  -fps_mode passthrough \
  keyframe_%04d.jpg
```

### 慢动作/快动作
```bash
# 0.5倍速（慢动作）
ffmpeg -i video.mp4 \
  -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" \
  -map "[v]" -map "[a]" \
  slow.mp4

# 2倍速（快进）
ffmpeg -i video.mp4 \
  -filter_complex "[0:v]setpts=0.5*PTS[v];[0:a]atempo=2.0[a]" \
  -map "[v]" -map "[a]" \
  fast.mp4
```

## 十六、GIF制作

```bash
# 视频转GIF
ffmpeg -i video.mp4 \
  -vf "fps=10,scale=320:-1:flags=lanczos" \
  -c:v gif \
  output.gif

# 高质量GIF
ffmpeg -i video.mp4 \
  -vf "fps=15,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  output.gif
```

## 十七、实用脚本

### 批量优化视频
```bash
#!/bin/bash
for f in *.MP4 *.mp4; do
  echo "处理: $f"
  ffmpeg -i "$f" \
    -vf "eq=brightness=0.03:contrast=1.05:saturation=1.1" \
    -c:v libx264 -crf 23 -preset fast \
    -c:a aac -b:a 128k \
    "optimized_$f"
done
```

### 批量转码
```bash
#!/bin/bash
for f in *.mov; do
  fname=$(basename "$f" .mov)
  echo "转换: $fname"
  ffmpeg -i "$f" \
    -c:v libx264 -crf 24 \
    -c:a aac \
    "${fname}.mp4"
done
```

### 提取音频
```bash
for f in *.mp4; do
  fname=$(basename "$f" .mp4)
  ffmpeg -i "$f" -vn -acodec libmp3lame -q:a 2 "${fname}.mp3"
done
```

---

## 常用滤镜速查表

| 滤镜 | 功能 | 示例 |
|------|------|------|
| `eq` | 亮度/对比度/饱和度 | `eq=brightness=0.05` |
| `fade` | 淡入淡出 | `fade=t=in:st=0:d=2` |
| `scale` | 缩放 | `scale=1280:-1` |
| `crop` | 裁剪 | `crop=800:600` |
| `rotate` | 旋转 | `rotate=PI/6` |
| `overlay` | 叠加 | `overlay=10:10` |
| `chromakey` | 绿幕抠像 | `chromakey=0x00ff00` |
| `hqdn3d` | 降噪 | `hqdn3d=1.5:1.5` |
| `vignette` | 暗角 | `vignette=PI/4` |
| `drawtext` | 添加文字 | `drawtext=text='Hello'` |
| `subtitles` | 添加字幕 | `subs.srt` |
| `setpts` | 时间戳修改 | `setpts=0.5*PTS` |
| `atempo` | 音频变速 | `atempo=2.0` |
| `amix` | 混音 | `amix=inputs=2` |
| `afade` | 音频淡入淡出 | `afade=t=in:d=2` |

---

**学习时间**: 2026-02-08 07:07
**来源**: FFmpeg官方文档 + 实践
**技能等级**: 高级 🎓
