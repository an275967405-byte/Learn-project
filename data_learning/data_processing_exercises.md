# AI数据处理分析师学习路径 - JD评论情感分析数据集

## 数据集介绍
- **数据集名称**: DAMO_NLP/jd
- **数据来源**: ModelScope
- **数据规模**: 约45,366条京东商品评论
- **数据字段**:
  - `sentence`: 评论文本（中文）
  - `label`: 情感标签（0.0=负面，1.0=正面）
  - `dataset`: 数据集标识（固定为"jd"）

---

## 📚 学习路径总览

| 阶段 | 主题 | 主要包 | 难度 | 预计时间 |
|------|------|--------|------|----------|
| 1 | 数据下载与加载 | modelscope + pandas | ⭐ | 2小时 |
| 2 | 数据探索与统计 | pandas | ⭐⭐ | 3小时 |
| 3 | 数据清洗 | pandas | ⭐⭐ | 4小时 |
| 4 | 数据可视化 | matplotlib | ⭐⭐⭐ | 4小时 |
| 5 | 文本预处理 | jieba | ⭐⭐⭐ | 5小时 |
| 6 | 特征工程 | pandas + numpy | ⭐⭐⭐⭐ | 6小时 |
| 7 | 数据标注与增强 | pandas | ⭐⭐⭐ | 4小时 |
| 8 | 数据预处理（为模型准备）| pandas + numpy | ⭐⭐⭐⭐ | 5小时 |
| 9 | 简单文本分类模型 | pytorch | ⭐⭐⭐⭐⭐ | 8小时 |

---

## 🎯 阶段一：数据下载与加载（pandas基础）

### 学习目标
- 学会使用 ModelScope 下载数据集
- 掌握 pandas 的基本操作
- 理解 DataFrame 数据结构

### 练习1.1：下载并加载数据集
**难度**: ⭐  
**时间**: 30分钟

**任务**:
1. 使用 `modelscope` 下载 DAMO_NLP/jd 数据集
2. 将数据转换为 pandas DataFrame
3. 查看数据的基本信息（形状、列名、数据类型）

**提示代码框架**:
```python
from modelscope import MsDataset
import pandas as pd

# 下载数据集
dataset = MsDataset.load('DAMO_NLP/jd', split='train')

# 转换为 DataFrame
df = pd.DataFrame(list(dataset))

# 查看基本信息
print(df.shape)  # 数据形状
print(df.columns)  # 列名
print(df.dtypes)  # 数据类型
print(df.head())  # 前5行
```

**检查点**:
- [ ] 成功下载数据集
- [ ] DataFrame 有3列：sentence, label, dataset
- [ ] 数据量约45,366行

---

### 练习1.2：数据基本信息统计
**难度**: ⭐  
**时间**: 30分钟

**任务**:
1. 统计数据集的基本信息
2. 查看缺失值情况
3. 统计标签分布

**要求输出**:
```
数据集基本信息:
- 总记录数: 45,366
- 缺失值统计: ...
- 标签分布: 正面 X 条, 负面 Y 条
```

**提示**:
- 使用 `df.isnull().sum()` 统计缺失值
- 使用 `df['label'].value_counts()` 统计标签分布

---

### 练习1.3：数据筛选与切片
**难度**: ⭐⭐  
**时间**: 1小时

**任务**:
1. 筛选出所有正面评论（label=1.0）
2. 筛选出所有负面评论（label=0.0）
3. 随机抽取100条数据进行快速分析
4. 按标签分组，每组各取10条样本

**提示**:
- 使用布尔索引：`df[df['label'] == 1.0]`
- 使用 `df.sample(n=100)` 随机抽样
- 使用 `df.groupby('label').head(10)` 分组取样

---

## 🔍 阶段二：数据探索与统计（pandas进阶）

### 学习目标
- 深入理解数据分布
- 掌握 pandas 的统计函数
- 发现数据中的模式和异常

### 练习2.1：文本长度分析
**难度**: ⭐⭐  
**时间**: 1小时

**任务**:
1. 计算每条评论的字符数
2. 统计文本长度的分布（最小值、最大值、平均值、中位数）
3. 找出最长和最短的评论
4. 分析不同标签的文本长度差异

**提示**:
```python
# 计算文本长度
df['text_length'] = df['sentence'].str.len()

# 统计信息
print(df['text_length'].describe())

# 按标签分组统计
df.groupby('label')['text_length'].describe()
```

---

### 练习2.2：数据质量检查
**难度**: ⭐⭐  
**时间**: 1.5小时

**任务**:
1. 检查重复数据
2. 检查空值/缺失值
3. 检查异常标签值（不是0.0或1.0的）
4. 检查文本是否为空或只包含空格
5. 生成数据质量报告

**要求输出**:
```
数据质量报告:
- 重复数据: X 条
- 缺失值: sentence X 条, label Y 条
- 异常标签: X 条
- 空文本: X 条
```

---

### 练习2.3：标签平衡性分析
**难度**: ⭐⭐  
**时间**: 1小时

**任务**:
1. 计算正负样本的比例
2. 判断数据集是否平衡
3. 如果不平衡，计算不平衡比例
4. 分析不平衡可能带来的影响

**提示**:
- 使用 `df['label'].value_counts(normalize=True)` 计算比例
- 平衡数据集：正负样本比例接近 1:1
- 不平衡数据集：比例差异较大（如 1:3 或更大）

---

## 🧹 阶段三：数据清洗（pandas实战）

### 学习目标
- 掌握数据清洗的常用方法
- 处理缺失值、重复值、异常值
- 文本标准化处理

### 练习3.1：处理缺失值
**难度**: ⭐⭐  
**时间**: 1.5小时

**任务**:
1. 识别所有缺失值的位置
2. 分析缺失值的模式（随机缺失还是系统缺失）
3. 选择合适的处理策略：
   - 删除包含缺失值的行
   - 填充缺失值（文本用空字符串，标签用众数）
4. 对比清洗前后的数据量

**策略选择**:
- 如果缺失值很少（<1%），可以删除
- 如果缺失值较多，需要填充或标记

---

### 练习3.2：去除重复数据
**难度**: ⭐⭐  
**时间**: 1小时

**任务**:
1. 找出完全重复的记录
2. 找出文本相同但标签不同的记录（数据冲突）
3. 处理重复数据：
   - 保留第一条或最后一条
   - 对于冲突数据，选择更合理的标签或删除
4. 保存清洗后的数据到CSV文件

**提示**:
```python
# 找出完全重复
duplicates = df[df.duplicated()]

# 找出文本重复但标签不同
text_duplicates = df[df.duplicated(subset=['sentence'], keep=False)]
conflicts = text_duplicates.groupby('sentence')['label'].nunique()
conflicts = conflicts[conflicts > 1]

# 去重（保留第一条）
df_clean = df.drop_duplicates(subset=['sentence'], keep='first')
```

---

### 练习3.3：文本标准化
**难度**: ⭐⭐⭐  
**时间**: 1.5小时

**任务**:
1. 去除文本首尾空格
2. 去除多余的空格（多个空格合并为一个）
3. 去除特殊字符（保留中文、英文、数字、常用标点）
4. 统一标点符号（全角转半角或反之）
5. 处理异常长的文本（如超过500字符的评论）

**提示**:
```python
# 去除首尾空格
df['sentence'] = df['sentence'].str.strip()

# 去除多余空格
df['sentence'] = df['sentence'].str.replace(r'\s+', ' ', regex=True)

# 全角转半角（可选）
import unicodedata
def to_halfwidth(text):
    return unicodedata.normalize('NFKC', text)
```

---

## 📊 阶段四：数据可视化（matplotlib）

### 学习目标
- 掌握 matplotlib 基础绘图
- 学会选择合适的图表类型
- 制作清晰美观的数据可视化

### 练习4.1：标签分布可视化
**难度**: ⭐⭐  
**时间**: 1小时

**任务**:
1. 绘制标签分布的柱状图
2. 绘制标签分布的饼图
3. 添加标题、标签、图例
4. 保存图片到文件

**要求**:
- 使用中文标签（正面/负面）
- 显示具体数值
- 图片清晰，适合报告使用

**提示**:
```python
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 柱状图
label_counts = df['label'].value_counts()
plt.bar(['负面', '正面'], [label_counts[0.0], label_counts[1.0]])
plt.title('标签分布')
plt.ylabel('数量')
plt.show()
```

---

### 练习4.2：文本长度分布可视化
**难度**: ⭐⭐⭐  
**时间**: 1.5小时

**任务**:
1. 绘制文本长度的直方图
2. 分别绘制正面和负面评论的长度分布（叠加或分图）
3. 绘制箱线图比较不同标签的文本长度
4. 分析长度与情感的关系

**要求**:
- 使用合适的bins数量
- 添加统计信息（均值、中位数）
- 使用不同颜色区分正负面

---

### 练习4.3：数据质量可视化
**难度**: ⭐⭐⭐  
**时间**: 1.5小时

**任务**:
1. 绘制缺失值热力图（如果有多个字段）
2. 绘制数据清洗前后的对比图
3. 绘制文本长度随时间/索引的变化趋势（如果有时间信息）
4. 创建一个综合的数据概览仪表板（多个子图）

**提示**:
```python
# 多个子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# 在axes上绘制不同的图
```

---

## 📝 阶段五：文本预处理（jieba分词）

### 学习目标
- 掌握中文分词
- 理解停用词的作用
- 文本预处理流程

### 练习5.1：中文分词基础
**难度**: ⭐⭐⭐  
**时间**: 1.5小时

**任务**:
1. 安装并使用 jieba 进行中文分词
2. 对数据集中的评论进行分词
3. 统计词频（找出最常见的词）
4. 分析正面和负面评论的高频词差异

**提示**:
```python
import jieba

# 分词
text = "质量很好 料子很不错"
words = jieba.cut(text)
word_list = list(words)

# 统计词频
from collections import Counter
word_freq = Counter(word_list)
```

---

### 练习5.2：停用词处理
**难度**: ⭐⭐⭐  
**时间**: 1.5小时

**任务**:
1. 下载或创建中文停用词表
2. 去除停用词
3. 对比去除停用词前后的词频变化
4. 分析哪些词对情感分析最重要

**提示**:
```python
# 停用词列表（示例）
stopwords = ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', ...]

# 过滤停用词
filtered_words = [w for w in words if w not in stopwords]
```

---

### 练习5.3：文本特征提取
**难度**: ⭐⭐⭐⭐  
**时间**: 2小时

**任务**:
1. 提取文本的统计特征：
   - 词数
   - 平均词长
   - 标点符号数量
   - 感叹号数量（可能表示强烈情感）
2. 提取情感词特征：
   - 正面词数量
   - 负面词数量
   - 情感词比例
3. 创建特征DataFrame，为后续分析做准备

**提示**:
```python
# 创建简单的情感词典
positive_words = ['好', '棒', '赞', '满意', '推荐', ...]
negative_words = ['差', '坏', '烂', '失望', '不推荐', ...]

def count_sentiment_words(text, word_list):
    words = jieba.cut(text)
    return sum(1 for w in words if w in word_list)
```

---

## 🔧 阶段六：特征工程（pandas + numpy）

### 学习目标
- 理解特征工程的重要性
- 创建数值特征
- 特征选择和转换

### 练习6.1：创建数值特征
**难度**: ⭐⭐⭐  
**时间**: 2小时

**任务**:
1. 基于文本创建特征：
   - 文本长度
   - 词数
   - 标点符号数量
   - 数字数量
   - 大写字母数量（如果有英文）
2. 使用 numpy 进行特征计算
3. 分析这些特征与标签的相关性

**提示**:
```python
import numpy as np

# 计算特征
df['char_count'] = df['sentence'].str.len()
df['word_count'] = df['sentence'].apply(lambda x: len(jieba.cut(x)))
df['punct_count'] = df['sentence'].str.count(r'[，。！？、；：]')

# 相关性分析
correlation = df[['char_count', 'word_count', 'punct_count', 'label']].corr()
```

---

### 练习6.2：特征标准化
**难度**: ⭐⭐⭐⭐  
**时间**: 2小时

**任务**:
1. 使用 numpy 实现特征标准化（Z-score标准化）
2. 使用 numpy 实现特征归一化（Min-Max标准化）
3. 对比标准化前后的特征分布
4. 理解为什么需要标准化

**提示**:
```python
# Z-score标准化
mean = df['feature'].mean()
std = df['feature'].std()
df['feature_normalized'] = (df['feature'] - mean) / std

# Min-Max归一化
min_val = df['feature'].min()
max_val = df['feature'].max()
df['feature_scaled'] = (df['feature'] - min_val) / (max_val - min_val)
```

---

### 练习6.3：特征选择
**难度**: ⭐⭐⭐⭐  
**时间**: 2小时

**任务**:
1. 计算所有特征与标签的相关性
2. 选择相关性高的特征
3. 使用特征重要性排序
4. 创建特征重要性可视化

**要求**:
- 至少创建10个不同的特征
- 分析哪些特征最有用
- 解释特征选择的理由

---

## 🏷️ 阶段七：数据标注与增强（pandas）

### 学习目标
- 理解数据标注的重要性
- 处理不平衡数据
- 数据增强技术

### 练习7.1：数据标注质量检查
**难度**: ⭐⭐⭐  
**时间**: 1.5小时

**任务**:
1. 检查标签的一致性
2. 找出可能有问题的标注（如正面词但标为负面）
3. 手动检查一些样本，评估标注质量
4. 统计标注准确率（如果有验证集）

**提示**:
```python
# 检查标签与文本的一致性
positive_keywords = ['好', '棒', '赞']
negative_keywords = ['差', '坏', '烂']

def check_label_consistency(row):
    text = row['sentence']
    label = row['label']
    has_positive = any(kw in text for kw in positive_keywords)
    has_negative = any(kw in text for kw in negative_keywords)
    # 检查逻辑...
```

---

### 练习7.2：处理数据不平衡
**难度**: ⭐⭐⭐  
**时间**: 2小时

**任务**:
1. 分析当前数据的平衡性
2. 如果数据不平衡，使用下采样（undersampling）平衡数据
3. 或者使用上采样（oversampling）平衡数据
4. 对比平衡前后的模型效果（如果有模型）

**提示**:
```python
# 下采样：从多数类中随机抽取与少数类相同数量的样本
from sklearn.utils import resample

# 分离多数类和少数类
majority = df[df['label'] == 1.0]
minority = df[df['label'] == 0.0]

# 下采样多数类
majority_downsampled = resample(majority, 
                                replace=False,
                                n_samples=len(minority),
                                random_state=42)

# 合并
df_balanced = pd.concat([majority_downsampled, minority])
```

---

### 练习7.3：数据增强（简单方法）
**难度**: ⭐⭐⭐  
**时间**: 1.5小时

**任务**:
1. 对于少数类样本，使用简单的数据增强：
   - 同义词替换（使用词典）
   - 随机删除非关键词
   - 文本重组（保持语义）
2. 生成增强后的数据集
3. 验证增强数据的质量

**注意**: 由于没有GPU，使用简单的规则方法，不使用深度学习模型

---

## 🎯 阶段八：数据预处理（为模型准备）

### 学习目标
- 准备训练/验证/测试集
- 数据格式转换
- 为PyTorch准备数据

### 练习8.1：数据集划分
**难度**: ⭐⭐⭐  
**时间**: 1.5小时

**任务**:
1. 将数据集划分为训练集、验证集、测试集（70%/15%/15%）
2. 确保每个集合中正负样本比例一致（分层采样）
3. 保存划分后的数据集
4. 统计各集合的基本信息

**提示**:
```python
from sklearn.model_selection import train_test_split

# 分层划分
X = df['sentence']
y = df['label']
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
```

---

### 练习8.2：文本向量化（简单方法）
**难度**: ⭐⭐⭐⭐  
**时间**: 2小时

**任务**:
1. 使用简单的词袋模型（Bag of Words）将文本转换为向量
2. 或者使用TF-IDF向量化
3. 将向量保存为numpy数组
4. 理解向量化的原理

**提示**:
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 词袋模型
vectorizer = CountVectorizer(max_features=1000)  # 限制特征数，避免内存问题
X_train_vec = vectorizer.fit_transform(X_train)
X_train_array = X_train_vec.toarray()  # 转为numpy数组
```

---

### 练习8.3：创建PyTorch DataLoader
**难度**: ⭐⭐⭐⭐  
**时间**: 2小时

**任务**:
1. 创建自定义的PyTorch Dataset类
2. 实现`__len__`和`__getitem__`方法
3. 创建DataLoader，设置合适的batch_size
4. 测试数据加载是否正常

**提示**:
```python
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 创建DataLoader
dataset = TextDataset(X_train_vec, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 🤖 阶段九：简单文本分类模型（PyTorch）

### 学习目标
- 理解神经网络基础
- 实现简单的文本分类模型
- 训练和评估模型

### 练习9.1：构建简单神经网络
**难度**: ⭐⭐⭐⭐  
**时间**: 2小时

**任务**:
1. 使用PyTorch构建一个简单的全连接神经网络
2. 输入：文本向量（如TF-IDF向量）
3. 输出：二分类（正面/负面）
4. 模型结构：输入层 -> 隐藏层 -> 输出层

**提示**:
```python
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

---

### 练习9.2：训练模型
**难度**: ⭐⭐⭐⭐⭐  
**时间**: 3小时

**任务**:
1. 定义损失函数（交叉熵损失）
2. 定义优化器（Adam）
3. 实现训练循环
4. 记录训练过程中的损失和准确率
5. 在验证集上评估模型

**要求**:
- 使用CPU训练（不需要GPU）
- 设置合适的epoch数（如10-20）
- 使用小batch_size（如32）避免内存问题
- 保存最佳模型

**提示**:
```python
import torch.optim as optim

model = SimpleClassifier(input_size=1000, hidden_size=128, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(10):
    model.train()
    for texts, labels in train_loader:
        # 前向传播
        outputs = model(texts)
        loss = criterion(outputs, labels.long())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

### 练习9.3：模型评估与改进
**难度**: ⭐⭐⭐⭐⭐  
**时间**: 3小时

**任务**:
1. 在测试集上评估模型性能
2. 计算准确率、精确率、召回率、F1分数
3. 绘制混淆矩阵
4. 分析模型的错误案例
5. 尝试改进模型（调整超参数、增加层数等）

**提示**:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 评估
model.eval()
with torch.no_grad():
    predictions = []
    true_labels = []
    for texts, labels in test_loader:
        outputs = model(texts)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.numpy())
        true_labels.extend(labels.numpy())

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
```

---

## 📋 学习检查清单

### 基础技能
- [ ] 能够使用pandas加载和处理数据
- [ ] 理解DataFrame的基本操作
- [ ] 能够进行数据清洗和预处理

### 进阶技能
- [ ] 能够进行数据可视化
- [ ] 掌握文本预处理方法
- [ ] 能够进行特征工程

### 高级技能
- [ ] 能够准备训练数据
- [ ] 理解PyTorch基础
- [ ] 能够训练简单的分类模型

---

## 💡 学习建议

1. **循序渐进**: 按照阶段顺序学习，不要跳步
2. **动手实践**: 每个练习都要亲自完成，不要只看答案
3. **理解原理**: 不仅要会写代码，还要理解为什么这样做
4. **记录笔记**: 记录遇到的问题和解决方案
5. **反复练习**: 对于难点，可以多练习几次
6. **查阅文档**: 学会查阅官方文档解决问题

---

## 📚 推荐资源

- **pandas**: https://pandas.pydata.org/docs/
- **matplotlib**: https://matplotlib.org/stable/contents.html
- **jieba**: https://github.com/fxsjy/jieba
- **PyTorch**: https://pytorch.org/tutorials/

---

## 🎓 项目实战建议

完成所有阶段后，可以尝试：
1. 整合所有技能，完成一个完整的情感分析项目
2. 尝试不同的模型架构
3. 优化模型性能
4. 撰写项目报告，总结学习成果

---

**祝学习顺利！** 🚀
