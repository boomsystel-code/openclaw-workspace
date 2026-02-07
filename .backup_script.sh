#!/bin/bash
# Daily workspace backup script for OpenClaw

cd ~/.openclaw/workspace

# Add all changes
git add -A

# Commit with timestamp
DATE=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "Auto backup: $DATE" 2>/dev/null

# Push to GitHub
git push origin main 2>/dev/null

# Log result
echo "[$(date)] Backup completed" >> ~/.openclaw/workspace/.backup.log
