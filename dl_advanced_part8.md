# 🚀 深度学习高级技术 Part 8

*前沿技术与系统设计*

---

## 111. 实时机器学习系统

### 111.1 在线学习

```python
class OnlineLearningSystem:
    """在线学习系统"""
    
    def __init__(self, model, window_size=1000):
        self.model = model
        self.window_size = window_size
        self.data_window = collections.deque(maxlen=window_size)
        self.model_update_freq = 100
    
    def process_request(self, features, feedback=None):
        """处理请求"""
        # 预测
        prediction = self.model.predict(features)
        
        # 存储反馈
        if feedback is not None:
            self.data_window.append((features, feedback))
        
        # 定期更新模型
        if len(self.data_window) % self.model_update_freq == 0:
            self.update_model()
        
        return prediction
    
    def update_model(self):
        """增量更新"""
        # 准备数据
        X, y = zip(*self.data_window)
        X = torch.stack(X)
        y = torch.stack(y)
        
        # 增量训练
        self.model.partial_fit(X, y)

class DriftDetector:
    """漂移检测"""
    
    def __init__(self, window_size=1000, threshold=0.5):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = None
        self.current_window = collections.deque(maxlen=window_size)
    
    def add_sample(self, prediction, actual=None):
        """添加样本"""
        sample = {'prediction': prediction, 'actual': actual}
        self.current_window.append(sample)
        
        if len(self.current_window) >= self.window_size:
            self._detect_drift()
    
    def _detect_drift(self):
        """检测漂移"""
        if self.reference_window is None:
            self.reference_window = list(self.current_window)
            return False
        
        # 计算预测精度变化
        ref_accuracy = self._calculate_accuracy(self.reference_window)
        curr_accuracy = self._calculate_accuracy(self.current_window)
        
        # 计算漂移分数
        drift_score = abs(ref_accuracy - curr_accuracy)
        
        # 触发漂移
        if drift_score > self.threshold:
            self.reference_window = list(self.current_window)
            return True
        
        return False
    
    def _calculate_accuracy(self, window):
        """计算准确率"""
        correct = sum(1 for s in window 
                     if s['prediction'] == s['actual'])
        return correct / len(window)
```

### 111.2 实时推理优化

```python
class InferenceOptimizer:
    """推理优化器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def optimize(self, input_shape):
        """优化推理"""
        import torch_tensorrt
        
        # TensorRT优化
        compiled_model = torch_tensorrt.compile(
            self.model,
            inputs=[],
            enabled_precisions={torch.float, torch.half}
        )
        
        return compiled_model
    
    def benchmark(self, model, input_shape, warmup=100, iterations=1000):
        """性能基准"""
        import time
        
        # 预热
        for _ in range(warmup):
            model(self._dummy_input(input_shape))
        
        # 测量
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(self._dummy_input(input_shape))
            latencies.append(time.perf_counter() - start)
        
        return {
            'mean_latency': np.mean(latencies) * 1000,  # ms
            'p95_latency': np.percentile(latencies, 95) * 1000,
            'throughput': 1 / np.mean(latencies)
        }
    
    def _dummy_input(self, shape):
        """生成虚拟输入"""
        return torch.randn(*shape).to(self.device)
```

### 111.3 服务架构

```python
class ModelServingArchitecture:
    """模型服务架构"""
    
    def __init__(self, models, load_balancer):
        self.models = models
        self.load_balancer = load_balancer
    
    def route_request(self, request):
        """路由请求"""
        # 负载均衡选择模型实例
        model = self.load_balancer.select()
        
        # 预处理
        features = self.preprocess(request)
        
        # 推理
        prediction = model.predict(features)
        
        # 后处理
        result = self.postprocess(prediction)
        
        return result

class BatchProcessor:
    """批处理"""
    
    def __init__(self, batch_size=32, timeout_ms=100):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.request_queue = queue.Queue()
        self.processing = False
    
    def add_request(self, request, callback):
        """添加请求"""
        self.request_queue.put((request, callback))
        
        if self.request_queue.qsize() >= self.batch_size:
            self._process_batch()
    
    def _process_batch(self):
        """处理批次"""
        requests = []
        callbacks = []
        
        while len(requests) < self.batch_size:
            try:
                request, callback = self.request_queue.get(timeout=self.timeout_ms/1000)
                requests.append(request)
                callbacks.append(callback)
            except queue.Empty:
                break
        
        # 批量推理
        if requests:
            batch_features = self._batch_features(requests)
            predictions = self.model.predict(batch_features)
            
            # 回调
            for callback, prediction in zip(callbacks, predictions):
                callback(prediction)
```

---

## 112. AI产品工程

### 112.1 功能设计

```python
class AIFeatureDesign:
    """AI功能设计"""
    
    @staticmethod
    def define_requirements():
        """定义需求"""
        return {
            'use_cases': [],
            'user_persona': None,
            'success_metrics': [],
            'constraints': []
        }
    
    @staticmethod
    def design_api():
        """设计API"""
        return {
            'endpoint': '/v1/predict',
            'method': 'POST',
            'input_schema': {},
            'output_schema': {}
        }
    
    @staticmethod
    def design_prompt():
        """设计提示词"""
        return {
            'system_prompt': '',
            'user_template': '',
            'few_shot_examples': []
        }

class PromptEngineering:
    """提示工程"""
    
    def __init__(self, base_prompt):
        self.base_prompt = base_prompt
    
    def add_context(self, context):
        """添加上下文"""
        return f"{self.base_prompt}\n\n上下文信息：\n{context}"
    
    def add_examples(self, examples):
        """添加示例"""
        example_str = "\n".join([
            f"输入：{ex['input']}\n输出：{ex['output']}"
            for ex in examples
        ])
        return f"{self.base_prompt}\n\n示例：\n{example_str}"
    
    def add_constraints(self, constraints):
        """添加约束"""
        constraints_str = "\n".join([f"- {c}" for c in constraints])
        return f"{self.base_prompt}\n\n约束条件：\n{constraints_str}"
    
    def format_output(self, format_type):
        """格式化输出"""
        formats = {
            'json': '请以JSON格式输出',
            'markdown': '请使用Markdown格式',
            'csv': '请使用CSV格式'
        }
        return f"{self.base_prompt}\n\n{formats.get(format_type, '')}"
```

### 112.2 A/B测试

```python
class ABTestManager:
    """A/B测试管理"""
    
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(self, name, variants, traffic_split=0.5):
        """创建实验"""
        self.experiments[name] = {
            'variants': variants,
            'traffic_split': traffic_split,
            'results': {v: [] for v in variants}
        }
    
    def assign_variant(self, user_id, experiment_name):
        """分配变体"""
        import hashlib
        hash_value = int(hashlib.md5(f"{user_id}_{experiment_name}".encode()).hexdigest(), 16)
        experiment = self.experiments[experiment_name]
        
        if hash_value % 100 < experiment['traffic_split'] * 100:
            return 'treatment'
        return 'control'
    
    def track_metric(self, experiment_name, variant, metric):
        """跟踪指标"""
        self.experiments[experiment_name]['results'][variant].append(metric)
    
    def analyze_results(self, experiment_name):
        """分析结果"""
        experiment = self.experiments[experiment_name]
        results = experiment['results']
        
        stats = {}
        for variant, metrics in results.items():
            if metrics:
                stats[variant] = {
                    'mean': np.mean(metrics),
                    'std': np.std(metrics),
                    'count': len(metrics)
                }
        
        # 统计显著性
        control = stats.get('control', {}).get('mean')
        treatment = stats.get('treatment', {}).get('mean')
        
        if control and treatment:
            lift = (treatment - control) / control
            
            # t检验
            from scipy import stats
            _, p_value = stats.ttest_ind(
                results['control'],
                results['treatment']
            )
            
            return {
                'control_mean': control,
                'treatment_mean': treatment,
                'lift': lift,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return stats
```

### 112.3 用户反馈循环

```python
class FeedbackLoop:
    """反馈循环"""
    
    def __init__(self, model, feedback_db):
        self.model = model
        self.feedback_db = feedback_db
    
    def collect_feedback(self, request_id, user_feedback):
        """收集反馈"""
        self.feedback_db.store(request_id, user_feedback)
    
    def analyze_feedback(self, time_window='7d'):
        """分析反馈"""
        feedback = self.feedback_db.query(time_window)
        
        # 分类反馈
        positive = [f for f in feedback if f['rating'] >= 4]
        negative = [f for f in feedback if f['rating'] <= 2]
        
        # 提取模式
        positive_patterns = self._extract_patterns(positive)
        negative_patterns = self._extract_patterns(negative)
        
        return {
            'positive_patterns': positive_patterns,
            'negative_patterns': negative_patterns,
            'sentiment_score': len(positive) / len(feedback) if feedback else 0.5
        }
    
    def improve_model(self, feedback_analysis):
        """改进模型"""
        # 基于反馈微调
        if feedback_analysis['sentiment_score'] < 0.7:
            # 使用负面反馈进行强化学习
            self._rlhf_finetune(feedback_analysis['negative_patterns'])
    
    def _extract_patterns(self, feedback_list):
        """提取模式"""
        patterns = {}
        for feedback in feedback_list:
            for key in feedback.get('tags', []):
                patterns[key] = patterns.get(key, 0) + 1
        return patterns
```

---

## 113. 多智能体系统

### 113.1 多智能体协作

```python
class MultiAgentSystem:
    """多智能体系统"""
    
    def __init__(self, agents):
        self.agents = agents
        self.communication = CommunicationChannel()
    
    def coordinate(self, task):
        """协调任务"""
        # 分解任务
        subtasks = self.decompose_task(task)
        
        # 分配给智能体
        assignments = self.assign_tasks(subtasks)
        
        # 并行执行
        results = self.execute_parallel(assignments)
        
        # 整合结果
        return self.integrate_results(results)
    
    def decompose_task(self, task):
        """分解任务"""
        return [subtask for subtask in task.split(';')]
    
    def assign_tasks(self, subtasks):
        """分配任务"""
        assignments = {}
        for i, subtask in enumerate(subtasks):
            agent = self.agents[i % len(self.agents)]
            assignments[agent] = subtask
        return assignments
    
    def execute_parallel(self, assignments):
        """并行执行"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                agent: executor.submit(agent.execute, task)
                for agent, task in assignments.items()
            }
            
            results = {}
            for agent, future in futures.items():
                results[agent] = future.result()
            
            return results

class AgentCommunication:
    """智能体通信"""
    
    def __init__(self):
        self.message_queue = queue.Queue()
        self.broadcast_channel = BroadcastChannel()
    
    def send_message(self, from_agent, to_agent, message):
        """发送消息"""
        self.message_queue.put({
            'from': from_agent,
            'to': to_agent,
            'content': message,
            'timestamp': time.time()
        })
    
    def broadcast(self, from_agent, message):
        """广播消息"""
        self.broadcast_channel.publish(from_agent, message)
    
    def receive_messages(self, agent_id):
        """接收消息"""
        messages = []
        while not self.message_queue.empty():
            msg = self.message_queue.get()
            if msg['to'] == agent_id:
                messages.append(msg)
        return messages
```

### 113.2 智能体协作模式

```python
class HierarchicalAgents:
    """层级智能体"""
    
    def __init__(self, manager_agent, worker_agents):
        self.manager = manager_agent
        self.workers = worker_agents
    
    def execute_task(self, task):
        """执行任务"""
        # 管理者分解任务
        plan = self.manager.plan(task)
        
        # 分配给工作者
        results = []
        for subtask in plan:
            worker = self._select_worker(subtask)
            result = worker.execute(subtask)
            results.append(result)
        
        # 管理者整合
        return self.manager.synthesize(results)
    
    def _select_worker(self, subtask):
        """选择工作者"""
        # 基于能力选择
        for worker in self.workers:
            if worker.can_handle(subtask):
                return worker
        return self.workers[0]

class DebateSystem:
    """辩论系统"""
    
    def __init__(self, agents, num_rounds=3):
        self.agents = agents
        self.num_rounds = num_rounds
    
    def discuss(self, question):
        """讨论"""
        # 初始立场
        stances = {agent.id: agent.initial_stance(question) 
                  for agent in self.agents}
        
        # 多轮辩论
        for round in range(self.num_rounds):
            for agent in self.agents:
                # 基于其他智能体的论点更新立场
                other_stances = {k: v for k, v in stances.items() 
                                if k != agent.id}
                agent.update_stance(other_stances)
                stances[agent.id] = agent.current_stance
        
        # 投票或聚合
        final_answer = self._aggregate_responses(stances)
        return final_answer
    
    def _aggregate_responses(self, stances):
        """聚合响应"""
        # 多数投票
        responses = [stance['answer'] for stance in stances.values()]
        from collections import Counter
        return Counter(responses).most_common(1)[0][0]
```

---

## 114. AI安全与鲁棒性

### 114.1 对抗攻击

```python
class AdversarialAttack:
    """对抗攻击"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def fgsm_attack(self, image, label, epsilon=0.03):
        """FGSM攻击"""
        image.requires_grad = True
        
        # 前向
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        
        # 反向
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗样本
        perturbed_image = image + epsilon * image.grad.sign()
        
        return perturbed_image
    
    def pgd_attack(self, image, label, epsilon=0.03, alpha=0.003, 
                   iterations=10):
        """PGD攻击"""
        original_image = image.clone()
        perturbed_image = image.clone()
        
        for _ in range(iterations):
            perturbed_image.requires_grad = True
            
            output = self.model(perturbed_image)
            loss = F.cross_entropy(output, label)
            
            self.model.zero_grad()
            loss.backward()
            
            # 更新
            perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
            
            # 投影
            perturbed_image = self._project(
                perturbed_image, original_image, epsilon
            )
        
        return perturbed_image
    
    def _project(self, perturbed, original, epsilon):
        """投影到epsilon球内"""
        return torch.clamp(
            perturbed - original, 
            -epsilon, epsilon
        ) + original

class AdversarialDefense:
    """对抗防御"""
    
    def __init__(self, model):
        self.model = model
    
    def adversarial_training(self, train_loader, epsilon=0.03):
        """对抗训练"""
        for batch in train_loader:
            images, labels = batch
            
            # 生成对抗样本
            adv_images = self.generate_adversarial(images, labels, epsilon)
            
            # 混合训练
            mixed_images = self.mixup(images, adv_images, alpha=1.0)
            
            # 训练
            self.train_step(mixed_images, labels)
    
    def mixup(self, images1, images2, alpha=1.0):
        """Mixup"""
        lam = np.random.beta(alpha, alpha)
        mixed = lam * images1 + (1 - lam) * images2
        return mixed
```

### 114.2 数据投毒防御

```python
class PoisonDetection:
    """投毒检测"""
    
    def __init__(self, model):
        self.model = model
    
    def detect_poison(self, data_loader):
        """检测投毒数据"""
        suspicious_samples = []
        
        for batch in data_loader:
            images, labels = batch
            
            # 异常检测
            anomaly_score = self._compute_anomaly(images)
            
            if anomaly_score > self.threshold:
                suspicious_samples.extend(
                    self._identify_poisoned(images, anomaly_score)
                )
        
        return suspicious_samples
    
    def _compute_anomaly(self, images):
        """计算异常分数"""
        with torch.no_grad():
            features = self.model.backbone(images)
            features = F.normalize(features, dim=1)
        
        # 基于KNN的异常检测
        return self._knn_distance(features)
    
    def _knn_distance(self, features):
        """KNN距离"""
        dist_matrix = torch.cdist(features, features)
        dist_matrix = dist_matrix + torch.eye(len(features)) * 1e6
        min_distances = dist_matrix.min(dim=1)[0]
        return min_distances.mean()

class CertifiedRobustness:
    """认证鲁棒性"""
    
    def __init__(self, model):
        self.model = model
    
    def certify(self, x, radius):
        """认证"""
        # 计算预测
        prediction = self.model(x).argmax(dim=1)
        
        # 认证半径
        certified_radius = self._compute_certified_radius(x, prediction)
        
        return {
            'prediction': prediction.item(),
            'certified_radius': certified_radius,
            'is_certified': certified_radius >= radius
        }
    
    def _compute_certified_radius(self, x, prediction):
        """计算认证半径"""
        # 基于平滑的认证
        return 0.1  # 示例
```

### 114.3 模型水印

```python
class ModelWatermark:
    """模型水印"""
    
    def __init__(self, model, watermark_key):
        self.model = model
        self.watermark_key = watermark_key
    
    def embed_watermark(self):
        """嵌入水印"""
        # 修改特定权重
        for name, param in self.model.named_parameters():
            if 'watermark' in name:
                param.data = self._encode_watermark(param.data)
    
    def verify_watermark(self, suspect_model):
        """验证水印"""
        suspect_params = dict(suspect_model.named_parameters())
        
        for name, param in self.model.named_parameters():
            if name in suspect_params:
                if not self._detect_watermark(param, suspect_params[name]):
                    return False
        
        return True
    
    def _encode_watermark(self, param):
        """编码水印"""
        # 基于密钥的编码
        return param + torch.randn_like(param) * 0.01
    
    def _detect_watermark(self, original, suspect):
        """检测水印"""
        return torch.allclose(original, suspect, atol=0.1)
```

---

## 115. 边缘AI与移动部署

### 115.1 移动优化

```python
class MobileOptimizer:
    """移动优化"""
    
    def __init__(self, model):
        self.model = model
    
    def optimize_for_mobile(self, input_shape):
        """移动优化"""
        import torch
        
        # 1. 量化
        quantized = self.quantize()
        
        # 2. 剪枝
        pruned = self.prune(ratio=0.5)
        
        # 3. 知识蒸馏
        distilled = self.distill_to_mobile()
        
        # 4. 导出
        self.export(distilled, input_shape)
        
        return distilled
    
    def quantize(self):
        """量化"""
        import torch.quantization
        return torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
    
    def export(self, model, input_shape):
        """导出"""
        import torch
        from torch.utils.mobile_optimizer import optimize_for_mobile
        
        example_input = torch.randn(*input_shape)
        traced = torch.jit.trace(model, example_input)
        optimized = optimize_for_mobile(traced)
        
        optimized.save("mobile_model.pt")
```

### 115.2 TFLite部署

```python
class TFLiteConverter:
    """TFLite转换器"""
    
    def __init__(self, model):
        self.model = model
    
    def convert(self, input_shape):
        """转换"""
        import tensorflow as tf
        
        # 转换
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 量化
        def representative_dataset():
            for _ in range(100):
                yield [np.random.randn(*input_shape).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATION_INT16_WEIGHTS_INT8
        ]
        
        tflite_model = converter.convert()
        
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        return 'model.tflite'
```

### 115.3 CoreML部署

```python
class CoreMLConverter:
    """CoreML转换器"""
    
    def __init__(self, model):
        self.model = model
    
    def convert(self, input_shape, output_path='model.mlpackage'):
        """转换"""
        import coremltools as ct
        
        # 转换
        traced = torch.jit.trace(self.model, torch.randn(*input_shape))
        
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=input_shape)],
            compute_units=ct.ComputeUnit.ALL,
            compute_precision=ct.precision.FLOAT16
        )
        
        # 保存
        mlmodel.save(output_path)
        
        return output_path
```

---

## 116. AI创业与商业化

### 116.1 商业模式

```python
class AIBusinessModel:
    """AI商业模式"""
    
    @staticmethod
    def saas_model():
        """SaaS模式"""
        return {
            'pricing': '按月/年订阅',
            'advantages': ['经常性收入', '客户锁定', '规模效应'],
            'challenges': ['客户获取成本', '流失率', '竞争']
        }
    
    @staticmethod
    def api_model():
        """API模式"""
        return {
            'pricing': '按调用次数计费',
            'advantages': ['低门槛', '即时收入', '易于扩展'],
            'challenges': ['定价压力', 'API稳定性', '安全']
        }
    
    @staticmethod
    def on_premise():
        """本地部署"""
        return {
            'pricing': '一次性许可费',
            'advantages': ['高客单价', '数据安全', '定制化'],
            'challenges': ['部署复杂', '维护成本', '扩展性']
        }

class PricingStrategy:
    """定价策略"""
    
    @staticmethod
    def usage_based_pricing(base_price, unit_price, usage_units):
        """用量定价"""
        return base_price + unit_price * usage_units
    
    @staticmethod
    def tiered_pricing(tiers):
        """分层定价"""
        def calculate(usage):
            for min_usage, max_usage, price in tiers:
                if min_usage <= usage < max_usage:
                    return price
            return tiers[-1][2]
        return calculate
```

### 116.2 产品市场匹配

```python
class PMFValidator:
    """PMF验证"""
    
    def __init__(self):
        self.metrics = {}
    
    def measure_pmf(self, survey_responses):
        """测量PMF"""
        # NPS
        promoters = sum(1 for r in survey_responses if r['nps'] >= 9)
        detractors = sum(1 for r in survey_responses if r['nps'] <= 6)
        nps = (promoters - detractors) / len(survey_responses) * 100
        
        # 使用频率
        avg_usage = np.mean([r['usage_frequency'] for r in survey_responses])
        
        # 推荐意愿
        avg_recommendation = np.mean([r['recommendation_likelihood'] 
                                     for r in survey_responses])
        
        return {
            'nps_score': nps,
            'avg_usage_frequency': avg_usage,
            'recommendation_score': avg_recommendation,
            'pmf_status': self._assess_pmf(nps, avg_usage)
        }
    
    def _assess_pmf(self, nps, usage):
        """评估PMF状态"""
        if nps > 50 and usage > 4:
            return 'Strong PMF'
        elif nps > 30 and usage > 3:
            return 'Weak PMF'
        else:
            return 'No PMF'
```

### 116.3 竞争分析

```python
class CompetitiveAnalysis:
    """竞争分析"""
    
    def __init__(self):
        self.competitors = {}
    
    def add_competitor(self, name, features, strengths, weaknesses):
        """添加竞争者"""
        self.competitors[name] = {
            'features': features,
            'strengths': strengths,
            'weaknesses': weaknesses
        }
    
    def compare(self, our_features, competitor_name):
        """比较"""
        competitor = self.competitors[competitor_name]
        
        comparison = {}
        for feature in our_features:
            our_score = our_features[feature]
            comp_score = competitor['features'].get(feature, 0)
            
            comparison[feature] = {
                'us': our_score,
                'them': comp_score,
                'advantage': our_score > comp_score
            }
        
        return comparison
    
    def find_blue_ocean(self, market_needs):
        """寻找蓝海"""
        opportunities = []
        
        for need in market_needs:
            addressed = False
            for competitor in self.competitors.values():
                if need in competitor['strengths']:
                    addressed = True
                    break
            
            if not addressed:
                opportunities.append(need)
        
        return opportunities
```

---

## 117. AI研究方法论

### 117.1 论文阅读

```python
class PaperReader:
    """论文阅读"""
    
    def __init__(self):
        self.read_papers = []
    
    def read_paper(self, paper_path):
        """阅读论文"""
        paper = self._parse_pdf(paper_path)
        
        # 提取关键信息
        summary = {
            'title': paper['title'],
            'authors': paper['authors'],
            'year': paper['year'],
            'problem': self._extract_problem(paper),
            'method': self._extract_method(paper),
            'results': self._extract_results(paper),
            'limitations': self._extract_limitations(paper),
            'future_work': self._extract_future_work(paper)
        }
        
        self.read_papers.append(summary)
        return summary
    
    def compare_papers(self, paper_ids):
        """比较论文"""
        papers = [self.read_papers[i] for i in paper_ids]
        
        comparison = {
            'problems': [p['problem'] for p in papers],
            'methods': [p['method'] for p in papers],
            'results': [p['results'] for p in papers],
            'strengths': [],
            'weaknesses': []
        }
        
        return comparison
    
    def literature_review(self, topic):
        """文献综述"""
        relevant = [p for p in self.read_papers 
                   if topic in p['title'].lower() or 
                   topic in p['problem'].lower()]
        
        return {
            'total_papers': len(relevant),
            'key_findings': self._synthesize_findings(relevant),
            'research_gaps': self._identify_gaps(relevant),
            'future_directions': self._suggest_directions(relevant)
        }
```

### 117.2 实验设计

```python
class ExperimentDesign:
    """实验设计"""
    
    def __init__(self):
        self.hypotheses = []
        self.experiments = []
    
    def formulate_hypothesis(self, variable1, relationship, variable2):
        """提出假设"""
        hypothesis = {
            'independent_var': variable1,
            'dependent_var': variable2,
            'relationship': relationship,
            'testable': True
        }
        self.hypotheses.append(hypothesis)
        return hypothesis
    
    def design_experiment(self, hypothesis, control_variables):
        """设计实验"""
        experiment = {
            'hypothesis': hypothesis,
            'control_variables': control_variables,
            'treatment_group': None,
            'control_group': None,
            'metrics': [],
            'sample_size': self._calculate_sample_size(hypothesis)
        }
        self.experiments.append(experiment)
        return experiment
    
    def _calculate_sample_size(self, hypothesis, alpha=0.05, power=0.8):
        """计算样本量"""
        effect_size = 0.5  # Cohen's d
        n = 2 * ((1.96 + 0.84) / effect_size) ** 2
        return int(n)
```

### 117.3 复现实验

```python
class ReproducibilityCheck:
    """复现性检查"""
    
    def __init__(self):
        self.replications = []
    
    def attempt_replication(self, paper, code_path, dataset):
        """尝试复现"""
        try:
            # 运行原始代码
            original_results = self._run_experiment(paper, code_path, dataset)
            
            # 在新数据集上测试
            new_results = self._run_experiment(paper, code_path, dataset, new_data=True)
            
            # 比较结果
            replication = {
                'paper': paper,
                'original_results': original_results,
                'new_results': new_results,
                'reproducible': self._compare_results(original_results, new_results),
                'differences': self._analyze_differences(original_results, new_results)
            }
            
            self.replications.append(replication)
            return replication
        
        except Exception as e:
            return {'error': str(e)}
    
    def report_reproducibility(self, replication_results):
        """报告复现性"""
        successful = sum(1 for r in replication_results if r.get('reproducible'))
        total = len(replication_results)
        
        return {
            'success_rate': successful / total if total > 0 else 0,
            'common_issues': self._identify_common_issues(replication_results),
            'recommendations': self._make_recommendations(replication_results)
        }
```

---

## 118. AI职业发展

### 118.1 技能矩阵

```python
class AISkillMatrix:
    """AI技能矩阵"""
    
    SKILLS = {
        'foundational': {
            'Python': 5,
            'Mathematics': 5,
            'Statistics': 4,
            'Data Structures': 3
        },
        'machine_learning': {
            'Supervised Learning': 5,
            'Unsupervised Learning': 4,
            'Deep Learning': 5,
            'Reinforcement Learning': 3
        },
        'engineering': {
            'MLOps': 4,
            'ML System Design': 4,
            'Optimization': 4,
            'Deployment': 3
        },
        'soft_skills': {
            'Communication': 4,
            'Problem Solving': 5,
            'Collaboration': 4,
            'Business Acumen': 3
        }
    }
    
    def assess_skills(self, current_skills):
        """评估技能"""
        gaps = {}
        
        for category, skills in self.SKILLS.items():
            category_gaps = {}
            for skill, required_level in skills.items():
                current_level = current_skills.get(f"{category}_{skill}", 0)
                if current_level < required_level:
                    category_gaps[skill] = {
                        'current': current_level,
                        'required': required_level,
                        'gap': required_level - current_level
                    }
            if category_gaps:
                gaps[category] = category_gaps
        
        return gaps
    
    def create_learning_plan(self, gaps, timeline_months=12):
        """创建学习计划"""
        total_gap = sum(
            sum(g['gap'] for g in category.values())
            for category in gaps.values()
        )
        
        months_per_level = timeline_months / total_gap
        
        plan = []
        for category, skill_gaps in gaps.items():
            for skill, gap_info in skill_gaps.items():
                for _ in range(gap_info['gap']):
                    plan.append({
                        'skill': skill,
                        'category': category,
                        'duration_months': months_per_level
                    })
        
        return plan
```

### 118.2 面试准备

```python
class MLInterviewPrep:
    """ML面试准备"""
    
    TOPICS = {
        'coding': [
            '数组和字符串操作',
            '链表和树',
            '动态规划',
            '图算法',
            '系统设计'
        ],
        'ml_theory': [
            '偏差-方差权衡',
            '正则化',
            '损失函数',
            '优化算法',
            '评估指标'
        ],
        'deep_learning': [
            '反向传播',
            '正则化技术',
            '架构选择',
            '训练技巧',
            '部署考虑'
        ],
        'case_studies': [
            '推荐系统',
            '搜索排名',
            '欺诈检测',
            '定价策略',
            '用户增长'
        ]
    }
    
    def generate_questions(self, topic, difficulty='medium'):
        """生成面试题"""
        questions = {
            'coding': [
                '实现K-means聚类',
                '设计一个神经网络类',
                '实现梯度下降变体',
                '写一个注意力机制'
            ],
            'ml_theory': [
                '解释偏差-方差 tradeoff',
                'L1 vs L2正则化的区别',
                '为什么使用ReLU而不是sigmoid',
                '如何处理类别不平衡'
            ],
            'deep_learning': [
                'Transformer的注意力机制',
                'Batch Normalization的作用',
                '如何防止过拟合',
                '解释学习率调度'
            ]
        }
        
        return questions.get(topic, [])
    
    def mock_interview(self, role='ML Engineer'):
        """模拟面试"""
        interview = {
            'role': role,
            'rounds': [
                {'type': 'coding', 'questions': self.generate_questions('coding')},
                {'type': 'ml_theory', 'questions': self.generate_questions('ml_theory')},
                {'type': 'deep_learning', 'questions': self.generate_questions('deep_learning')},
                {'type': 'system_design', 'questions': ['设计一个推荐系统']}
            ],
            'duration_minutes': 60,
            'tips': [
                '边说边做',
                '先讲思路再写代码',
                '考虑边界情况',
                '讨论复杂度'
            ]
        }
        return interview
```

### 118.3 职业路径

```python
class CareerPath:
    """职业路径"""
    
    PATHS = {
        'research': {
            'phd_required': True,
            'steps': [
                '研究实习',
                '发表论文',
                '博士后',
                '研究员',
                '教授/首席科学家'
            ],
            'salary_range': '$150