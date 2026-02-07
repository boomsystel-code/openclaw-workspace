#!/usr/bin/env python3
"""Multi-threaded downloader for large files."""

import os
import sys
import math
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import argparse

class MultiThreadDownloader:
    def __init__(self, url, output_path, num_threads=16):
        self.url = url
        self.output_path = output_path
        self.num_threads = num_threads
        self.chunk_size = 1024 * 1024  # 1MB chunks
        self.headers = {}
        
    def get_file_size(self):
        """Get file size from server."""
        response = requests.head(self.url, headers=self.headers)
        return int(response.headers.get('content-length', 0))
    
    def download_chunk(self, start, end, chunk_id):
        """Download a single chunk."""
        headers = self.headers.copy()
        headers['Range'] = f'bytes={start}-{end}'
        
        response = requests.get(self.url, headers=headers, stream=True)
        chunk_data = response.content
        
        return chunk_id, chunk_data
    
    def download(self):
        """Download file using multiple threads."""
        # Get file size
        file_size = self.get_file_size()
        print(f"文件大小: {file_size / (1024*1024*1024):.2f} GB")
        
        if file_size == 0:
            print("错误: 无法获取文件大小")
            return False
        
        # Calculate chunks
        num_chunks = min(self.num_threads, math.ceil(file_size / self.chunk_size))
        chunk_size = math.ceil(file_size / num_chunks)
        
        print(f"使用 {num_chunks} 个线程下载...")
        print(f"每个线程块大小: {chunk_size / (1024*1024):.2f} MB")
        
        # Prepare ranges
        ranges = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size - 1, file_size - 1)
            ranges.append((start, end, i))
        
        # Download in parallel
        downloaded = {}
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = {
                executor.submit(self.download_chunk, start, end, chunk_id): (start, end, chunk_id)
                for start, end, chunk_id in ranges
            }
            
            completed = 0
            for future in as_completed(futures):
                chunk_id, chunk_data = future.result()
                downloaded[chunk_id] = chunk_data
                completed += 1
                print(f"进度: {completed}/{num_chunks} ({completed/num_chunks*100:.1f}%)")
        
        # Write to file
        print("写入文件...")
        with open(self.output_path, 'wb') as f:
            for i in range(num_chunks):
                f.write(downloaded[i])
        
        print(f"下载完成: {self.output_path}")
        print(f"文件大小: {os.path.getsize(self.output_path) / (1024*1024*1024):.2f} GB")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-threaded file downloader')
    parser.add_argument('url', help='URL to download')
    parser.add_argument('output', help='Output file path')
    parser.add_argument('--threads', type=int, default=16, help='Number of threads (default: 16)')
    
    args = parser.parse_args()
    
    downloader = MultiThreadDownloader(args.url, args.output, args.threads)
    downloader.download()
