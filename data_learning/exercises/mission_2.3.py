from modelscope import MsDataset
from pandas import DataFrame

def load_dataset(dataset_name):
    dataset = MsDataset.load(dataset_name, split='train') 
    return dataset

ds = load_dataset('DAMO_NLP/jd')

df = DataFrame(list(ds))

# 1. 计算正负样本的比例
proportion_ = df['label'].value_counts(normalize=True)
counts_ = df['label'].value_counts()

# 2. 判断数据集是否平衡
for label, count in counts_.items():
    label_name = '正面' if label == 1.0 else '负面'
    proportion = proportion_[label]
    print(f'{label_name} 数量: {count},占比{proportion*100:.2f}%')

# 3. 如果不平衡，计算不平衡比例
# 4. 分析不平衡可能带来的影响
balance_standards = 1.5  
max_prop = proportion_.max()
min_prop = proportion_.min()
balance_ratio = max_prop / min_prop
if balance_ratio <= balance_standards:
    print("数据集是平衡的")
else:
    print("数据集是不平衡的")
    majority_label = proportion_.idxmax()
    minority_label = proportion_.idxmin()
    majority_count = counts_[majority_label]
    minority_count = counts_[minority_label]
    balance_ratio_counts = majority_count / minority_count
    print(f"多数类: {majority_label}, {majority_count} 条")
    print(f"少数类: {minority_label}, {minority_count} 条")
    print(f"不平衡比例 (多数类/少数类): {balance_ratio_counts:.2f}:1")

