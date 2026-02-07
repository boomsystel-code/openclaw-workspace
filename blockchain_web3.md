# 区块链开发与Web3实战

## 第一章：Solidity智能合约

### 1.1 合约基础结构

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// 导入
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

// 错误处理
error InsufficientBalance(uint256 requested, uint256 available);
error UnauthorizedCaller();

// 合约定义
contract MyToken is ERC20, Ownable {
    // 事件定义
    event TokensMinted(address indexed to, uint256 amount);
    event TokensBurned(address indexed from, uint256 amount);
    
    // 常量
    uint256 public constant MAX_SUPPLY = 1000000 ether;
    
    // 构造函数
    constructor() ERC20("MyToken", "MTK") Ownable(msg.sender) {}
    
    // mint函数
    function mint(address to, uint256 amount) external onlyOwner {
        if (totalSupply() + amount > MAX_SUPPLY) {
            revert ExceedsMaxSupply();
        }
        _mint(to, amount);
        emit TokensMinted(to, amount);
    }
    
    // burn函数
    function burn(uint256 amount) external {
        if (balanceOf(msg.sender) < amount) {
            revert InsufficientBalance(amount, balanceOf(msg.sender));
        }
        _burn(msg.sender, amount);
        emit TokensBurned(msg.sender, amount);
    }
    
    // 获取余额
    function getBalance(address account) external view returns (uint256) {
        return balanceOf(account);
    }
}
```

### 1.2 高级合约模式

#### ERC721 NFT合约
```solidity
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract NFTContract is ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    
    Counters.Counter private _tokenIds;
    
    mapping(address => uint256[]) private _ownedTokens;
    mapping(uint256 => address) private _tokenApprovals;
    
    constructor() ERC721("MyNFT", "MNFT") Ownable(msg.sender) {}
    
    function mint(address recipient, string memory tokenURI) 
        external 
        onlyOwner 
        returns (uint256) 
    {
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _mint(recipient, newTokenId);
        _setTokenURI(newTokenId, tokenURI);
        
        _ownedTokens[recipient].push(newTokenId);
        
        return newTokenId;
    }
    
    function burn(uint256 tokenId) external {
        address owner = ownerOf(tokenId);
        require(
            msg.sender == owner,
            "Not the token owner"
        );
        
        _burn(tokenId);
        _removeTokenFromOwnerEnumeration(owner, tokenId);
    }
    
    function getOwnedTokens(address owner) 
        external 
        view 
        returns (uint256[] memory) 
    {
        return _ownedTokens[owner];
    }
}
```

#### DeFi借贷合约
```solidity
import "@openzeppelin/contracts/token/IERC20.sol";

contract LendingPool {
    // 存款利率结构
    struct Deposit {
        uint256 amount;
        uint256 timestamp;
    }
    
    // 借款利率结构
    struct Borrow {
        uint256 amount;
        uint256 collateralAmount;
        uint256 timestamp;
    }
    
    mapping(address => uint256) public deposits;
    mapping(address => Borrow[]) public borrows;
    
    uint256 public constant COLLATERAL_FACTOR = 150; // 150%
    uint256 public interestRate = 5; // 年利率5%
    
    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event Borrowed(address indexed user, uint256 amount, uint256 collateral);
    event Repaid(address indexed user, uint256 amount);
    
    function deposit(uint256 amount) external {
        require(amount > 0, "Deposit amount must be greater than 0");
        
        IERC20(msg.sender).transferFrom(
            msg.sender,
            address(this),
            amount
        );
        
        deposits[msg.sender] += amount;
        
        emit Deposited(msg.sender, amount);
    }
    
    function withdraw(uint256 amount) external {
        require(
            deposits[msg.sender] >= amount,
            "Insufficient deposit balance"
        );
        
        deposits[msg.sender] -= amount;
        
        IERC20(msg.sender).transfer(msg.sender, amount);
        
        emit Withdrawn(msg.sender, amount);
    }
    
    function borrow(uint256 collateralAmount, uint256 borrowAmount) 
        external 
    {
        require(
            borrowAmount * 100 <= collateralAmount * COLLATERAL_FACTOR,
            "Insufficient collateral"
        );
        
        borrows[msg.sender].push(Borrow(
            borrowAmount,
            collateralAmount,
            block.timestamp
        ));
        
        IERC20(msg.sender).transfer(msg.sender, borrowAmount);
        
        emit Borrowed(msg.sender, borrowAmount, collateralAmount);
    }
    
    function repay(uint256 borrowIndex, uint256 amount) external {
        Borrow storage borrow = borrows[msg.sender][borrowIndex];
        require(borrow.amount >= amount, "Repay amount exceeds debt");
        
        borrow.amount -= amount;
        
        IERC20(msg.sender).transferFrom(
            msg.sender,
            address(this),
            amount
        );
        
        emit Repaid(msg.sender, amount);
    }
}
```

---

## 第二章：Web3.js与Ethers.js

### 2.1 Ethers.js基础

```typescript
import { ethers } from 'ethers';

// 连接提供者
const provider = new ethers.providers.JsonRpcProvider(
    'https://mainnet.infura.io/v3/YOUR_PROJECT_ID'
);

// 读取合约
async function getTokenInfo() {
    const address = '0x123...'; // ERC20合约地址
    const abi = [
        'function name() view returns (string)',
        'function symbol() view returns (string)',
        'function totalSupply() view returns (uint256)',
        'function balanceOf(address owner) view returns (uint256)'
    ];
    
    const contract = new ethers.Contract(address, abi, provider);
    
    const name = await contract.name();
    const symbol = await contract.symbol();
    const totalSupply = await contract.totalSupply();
    const balance = await contract.balanceOf('0xUserAddress...');
    
    console.log({ name, symbol, totalSupply, balance });
}

// 签名者与钱包
function createWallet() {
    // 创建随机钱包
    const wallet = ethers.Wallet.createRandom();
    console.log('Address:', wallet.address);
    console.log('Private Key:', wallet.privateKey);
    
    // 从私钥导入
    const walletFromPK = new ethers.Wallet(
        '0xYourPrivateKey',
        provider
    );
    
    // 连接提供者
    const connectedWallet = wallet.connect(provider);
    
    return connectedWallet;
}

// 发送交易
async function sendETH() {
    const wallet = createWallet();
    
    const tx = {
        to: '0xRecipientAddress',
        value: ethers.utils.parseEther('0.1') // 0.1 ETH
    };
    
    const txResponse = await wallet.sendTransaction(tx);
    console.log('Transaction hash:', txResponse.hash);
    
    await txResponse.wait(); // 等待确认
    console.log('Transaction confirmed!');
}
```

### 2.2 交互智能合约

```typescript
import { ethers, Contract } from 'ethers';

// 部署合约
async function deployContract() {
    const factory = new ethers.ContractFactory(
        abi,
        bytecode,
        wallet
    );
    
    const contract = await factory.deploy();
    
    console.log('Deploying contract...');
    await contract.deployed();
    
    console.log('Contract deployed at:', contract.address);
}

// 调用合约方法
async function interactWithContract() {
    const contractAddress = '0xContractAddress';
    const contract = new Contract(
        contractAddress,
        abi,
        wallet
    );
    
    // 读取方法（view函数）
    const value = await contract.getValue();
    console.log('Value:', value);
    
    // 写入方法（交易）
    const tx = await contract.setValue(42);
    console.log('Transaction hash:', tx.hash);
    await tx.wait();
    
    // 监听事件
    contract.on('ValueChanged', (newValue, sender) => {
        console.log(`Value changed to ${newValue} by ${sender}`);
    });
    
    // 监听单个事件
    contract.once('ValueChanged', (newValue) => {
        console.log('First ValueChanged event:', newValue);
    });
}

// ERC20代币交互
async function transferTokens() {
    const tokenAddress = '0xTokenAddress';
    const token = new ethers.Contract(tokenAddress, erc20Abi, wallet);
    
    // 检查余额
    const balance = await token.balanceOf(wallet.address);
    console.log('Balance:', ethers.utils.formatEther(balance));
    
    // 授权转账
    const spender = '0xSpenderAddress';
    const amount = ethers.utils.parseEther('100');
    
    const approveTx = await token.approve(spender, amount);
    await approveTx.wait();
    
    // 转账（由spender触发）
    const transferFromTx = await token.transferFrom(
        wallet.address,
        '0xRecipient',
        amount
    );
    await transferFromTx.wait();
}
```

---

## 第三章：去中心化应用架构

### 3.1 DApp架构设计

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (React/Vue)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Web3.js    │  │  Wallet     │  │   IPFS      │    │
│  │  Provider   │  │  Connection │  │  Client     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
                            │
                            │ JSON-RPC / Wallet Link
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Smart Contracts                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Token     │  │   Governance│  │   NFT       │    │
│  │  Contract   │  │   Contract  │  │  Contract   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
                            │
                            │ Events / Logs
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Backend Service                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Indexer     │  │  GraphQL    │  │   Subgraph  │    │
│  │ (The Graph) │  │   API       │  │   Publish   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Storage Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  IPFS       │  │   Arweave   │  │  Ceramic    │    │
│  │  (内容存储)  │  │  (永久存储) │  │  (身份数据) │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 The Graph索引

```yaml
# subgraph.yaml
specVersion: 0.0.5
description: My DeFi Protocol
repository: https://github.com/myorg/subgraph
schema:
  file: ./schema.graphql
dataSources:
  - kind: ethereum/contract
    name: LendingPool
    network: mainnet
    source:
      address: "0xLendingPoolAddress"
      abi: LendingPool
      startBlock: 15000000
    mapping:
      kind: ethereum/events
      apiVersion: 0.0.7
      language: wasm/assemblyscript
      entities:
        - Deposit
        - Withdraw
        - Borrow
        - Repay
      abis:
        - name: LendingPool
          file: ./abis/LendingPool.json
      eventHandlers:
        - event: Deposit(indexed address,uint256)
          handler: handleDeposit
        - event: Withdraw(indexed address,uint256)
          handler: handleWithdraw
        - event: Borrow(indexed address,uint256,uint256)
          handler: handleBorrow
        - event: Repay(indexed address,uint256,uint256)
          handler: handleRepay
      file: ./src/mapping.ts
```

---

## 参考资源

### 官方文档
- Solidity: docs.soliditylang.org
- Ethers.js: docs.ethers.org
- OpenZeppelin: docs.openzeppelin.com
- The Graph: thegraph.com/docs

### 进阶资源
- CryptoZombies教程
- Buildspace项目
- SpeedRunEthereum

### 安全审计
- Slither静态分析
- Mythril智能合约审计
- Certik审计服务

---

*本知识文件最后更新：2026-02-07*
