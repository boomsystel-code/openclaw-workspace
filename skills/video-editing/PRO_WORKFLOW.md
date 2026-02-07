# 专业视频制作工作流 (Professional Video Production)

## 一、前期准备

### 1. 素材整理
```
📁 项目文件夹结构:
├── 01_原始素材/
│   ├── 视频/
│   ├── 音频/
│   └── 图片/
├── 02_代理文件/
│   ├── 低分辨率用于剪辑
│   └── 高码率用于最终输出
├── 03_工程文件/
│   ├── Premiere/
│   ├── Final Cut Pro/
│   └── DaVinci Resolve/
├── 04_输出文件/
│   ├── 草稿版本/
│   └── 最终版本/
└── 05_素材库/
    ├── 音乐/
    ├── 音效/
    └── 图形/
```

### 2. 代理工作流（4K/8K必备）
```bash
# 为4K视频创建1080P代理
ffmpeg -i 4K_video.mov \
  -vf "scale=-1:1080" \
  -c:v prores -profile 3 \
  -an \
  proxy_1080p.mov

# 代理文件命名规范
# 格式: [原始文件名]_proxy_[分辨率].mov
# 例如: vacation_4k_proxy_1080p.mov
```

## 二、调色流程（DaVinci Resolve风格）

### 1. 一级调色（基础校正）
```
曝光（Exposure）
对比度（Contrast）
高光（Highlights）
阴影（Shadows）
白平衡（White Balance）
黑平衡（Black Balance）
```

### 2. 二级调色（局部调整）
```
遮罩跟踪（Mask Tracking）
限定器（Qualifier）- 选择特定颜色
曲线（Curves）
HSL辅助
```

### 3. 风格化
```
胶片模拟（Film Simulation）
LUT应用
颗粒（Noise/Grain）
暗角（Vignette）
```

```bash
# 使用FFmpeg应用调色LUT
ffmpeg -i input.mp4 \
  -vf "lut3d=cine主导.cube" \
  -c:v libx264 -crf 18 \
  -c:a copy \
  graded.mp4
```

## 三、音频工作流

### 1. 音频层级
```
对白（Dialogue）    -50 LUFS
音乐（Music）       -55 LUFS
音效（Effects）     -60 LUFS
整体峰值（Peak）    -3 dBTP
```

### 2. 音频处理
```bash
# 降噪+标准化
ffmpeg -i input.mp4 \
  -af "afftdn=nf=-30,loudnorm=I=-16:TP=-1.5:LRA=11" \
  -c:v copy \
  audio_fixed.mp4

# 人声增强
ffmpeg -i input.mp4 \
  -af "equalizer=f=300:g=3:type=shelf, \
       equalizer=f=4000:g=-2:type=shelf, \
       compand=attacks=0.05:decays=0.2:points=-70/-70|-60/-20|0/0" \
  -c:v copy \
  voice_enhanced.mp4
```

### 3. 响度标准化
```bash
# Spotify标准
ffmpeg -i input.mp4 \
  -af "loudnorm=I=-14:TP=-1.0:LRA=11:print_format=summary" \
  -c:v copy \
  normalized.mp4
```

## 四、编码与导出

### 1. 平台优化编码设置

#### YouTube/抖音（H.264）
```bash
ffmpeg -i input.mp4 \
  -c:v libx264 -crf 20 -preset slow \
  -c:a aac -b:a 192k \
  -movflags +faststart \
  -vf "scale=-2:1080" \
  youtube_1080p.mp4
```

#### Instagram（竖屏）
```bash
ffmpeg -i input.mp4 \
  -c:v libx264 -crf 22 -preset fast \
  -c:a aac -b:a 128k \
  -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2" \
  instagram_9:16.mp4
```

#### 归档备份（无损）
```bash
ffmpeg -i input.mp4 \
  -c:v libx264 -crf 12 -preset veryslow \
  -c:a copy \
  archive_master.mkv
```

### 2. 多码率输出（DASH/HLS）
```bash
# 1080p
ffmpeg -i input.mp4 \
  -c:v libx264 -crf 22 -preset fast -vf "scale=-1:1080" \
  -c:a aac -b:a 128k \
  -hls_time 10 -hls_list_size 0 \
  -f hls \
  1080p.m3u8

# 720p
ffmpeg -i input.mp4 \
  -c:v libx264 -crf 23 -preset fast -vf "scale=-1:720" \
  -c:a aac -b:a 96k \
  -hls_time 10 -hls_list_size 0 \
  -f hls \
  720p.m3u8
```

## 五、专业特效

### 1. 速度曲线（Ease In/Out）
```bash
# 自定义速度曲线
ffmpeg -i input.mp4 \
  -filter_complex "[0:v]setpts=eq(t)*2[v]" \
  -map "[v]" -map 0:a \
  eased.mp4

# 关键帧插值（非线性编辑）
# IN: 快速进入 -> 中速 -> 缓慢停止
```

### 2. 运动模糊
```bash
# 模拟运动模糊
ffmpeg -i input.mp4 \
  -vf "mbd=2:zoom=1.2:zoomcch=0.1" \
  -c:v libx264 -crf 20 \
  motion_blur.mp4
```

### 3. 粒子效果
```bash
# 雪花粒子
ffmpeg -f lavfi -i "color=255:255:255:0.3:size=50x50:rate=30[fg]" \
  -i input.mp4 \
  -filter_complex "[1:v][fg]overlay[out]" \
  -map "[out]" \
  snow_effect.mp4
```

### 4. 光效叠加
```bash
# 镜头光晕
ffmpeg -i input.mp4 -i lensflare.png \
  -filter_complex "[0:v][1:v]overlay=W-w-100:100,lumakey=threshold=0.1:softness=0.3[out]" \
  -map "[out]" \
  flare.mp4
```

## 六、稳定与去抖动

### 1. 智能稳定
```bash
# 基础稳定
ffmpeg -i shaky.mp4 \
  -vf "deshake=rx=16:ry=16:edge=1" \
  stabilized.mp4

# 高级稳定（需要vid.stab）
ffmpeg -i shaky.mp4 \
  -vf "vidstabdetect=stepsize=32:shakiness=10:accuracy=15" \
  -f null -

ffmpeg -i shaky.mp4 \
  -vf "vidstabtransform=smoothing=30:optzoom=1:interpol=2,unsharp=5:5:1.0:5:5:0.0" \
  -c:v libx264 -crf 20 \
  stable.mp4
```

### 2. 慢门效果
```bash
# 1/4快门速度效果
ffmpeg -i input.mp4 \
  -vf "avgblur=opencl=true:sizeX=15" \
  -c:v libx264 -crf 18 \
  motion_blur_slow.mp4
```

## 七、修复老视频

### 1. 去划痕
```bash
ffmpeg -i old_video.mp4 \
  -vf "removegrain=mode=20:ss=2" \
  de_scratch.mp4
```

### 2. 去色带（隔行扫描）
```bash
ffmpeg -i old_interlaced.mp4 \
  -vf "bwdif=deint=1" \
  -c:v libx264 -crf 20 \
  deinterlaced.mp4
```

### 3. 上色（AI辅助）
```bash
# 注意：自动上色需要AI模型
# 推荐使用DaVinci Resolve的AI上色功能
```

## 八、批量处理脚本

### 批量转码
```bash
#!/bin/bash
# batch_encode.sh

INPUT_DIR="./to_encode"
OUTPUT_DIR="./encoded"

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.mp4; do
  filename=$(basename "$f" .mp4)
  echo "处理: $filename"
  
  ffmpeg -i "$f" \
    -c:v libx264 -crf 20 -preset slow \
    -c:a aac -b:a 192k \
    "$OUTPUT_DIR/${filename}_encoded.mp4"
  
  echo "完成: $filename"
done

echo "全部完成！"
```

### 批量加字幕
```bash
#!/bin/bash
# batch_subs.sh

for f in *.mp4; do
  srt="${f%.*}.srt"
  if [ -f "$srt" ]; then
    echo "添加字幕: $f"
    ffmpeg -i "$f" -vf "subtitles=$srt" \
      -c:v libx264 -crf 22 \
      "subs_$f"
  fi
done
```

## 九、质量检查清单

### 出品前检查
- [ ] 曝光正常（无过曝/欠曝）
- [ ] 白平衡正确
- [ ] 对比度适中
- [ ] 画面稳定
- [ ] 音频无削波
- [ ] 响度标准化（-14 LUFS）
- [ ] 无明显噪点/伪影
- [ ] 字幕同步
- [ ] 转场流畅
- [ ] 色彩空间正确

### 技术参数
| 平台 | 分辨率 | 码率 | 帧率 | 格式 |
|------|--------|------|------|------|
| YouTube | 4K | 35-68Mbps | 30/60 | H.264 |
| 抖音 | 1080p | 8-15Mbps | 30/60 | H.264 |
| Instagram Feed | 1080p | 10-20Mbps | 30 | H.264 |
| 微信/QQ | 720p | 4-8Mbps | 30 | H.264 |

---

**学习时间**: 2026-02-08 07:10
**主题**: 专业视频制作工作流
**来源**: 行业最佳实践 + FFmpeg文档
