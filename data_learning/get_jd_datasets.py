"""
使用 ModelScope 下载并预览 DAMO_NLP/jd 数据集
"""
from modelscope import MsDataset
import pandas as pd
from textwrap import fill

def print_section(title, char="=", width=80):
    """打印格式化的章节标题"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")

def print_info(key, value, indent=2):
    """打印格式化的键值对"""
    print(f"{' ' * indent}• {key:15s}: {value}")

# 下载数据集
print_section("📥 正在下载数据集", "=")
print("正在从 ModelScope 下载 DAMO_NLP/jd 数据集...")
dataset = MsDataset.load('DAMO_NLP/jd', split='train')

# 数据集基本信息
print_section("📊 数据集基本信息", "=")
print_info("数据集名称", "DAMO_NLP/jd")
print_info("数据集类型", str(type(dataset).__name__))

# 转换为列表以便预览
print("\n正在加载数据...")
data_list = list(dataset)

print_info("总记录数", f"{len(data_list):,} 条")

# 数据结构
if data_list:
    print_section("📋 数据结构", "-")
    sample = data_list[0]
    for i, key in enumerate(sample.keys(), 1):
        print_info(f"字段 {i}", key)

# 预览前几条数据
print_section("👀 数据预览 (前 5 条)", "=")
for i, item in enumerate(data_list[:5], 1):
    print(f"\n{'─' * 80}")
    print(f"📝 记录 #{i}")
    print(f"{'─' * 80}")
    for key, value in item.items():
        if key == 'sentence':
            # 对长文本进行格式化
            formatted_value = fill(str(value), width=70, initial_indent="  ", 
                                   subsequent_indent="  ")
            print(f"  {key:12s}: {formatted_value}")
        else:
            print(f"  {key:12s}: {value}")

if len(data_list) > 5:
    print(f"\n{'─' * 80}")
    print(f"  ... (共 {len(data_list):,} 条记录)")

# 转换为 DataFrame 并显示统计信息
try:
    print_section("📈 数据统计分析", "=")
    
    # 从已加载的数据列表创建 DataFrame
    df = pd.DataFrame(data_list)
    
    print_info("DataFrame 形状", f"{df.shape[0]:,} 行 × {df.shape[1]} 列")
    print_info("内存使用", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 缺失值统计
    print(f"\n{'─' * 80}")
    print("  📉 缺失值统计:")
    missing = df.isnull().sum()
    for col in df.columns:
        count, pct = missing[col], (missing[col] / len(df)) * 100
        print(f"    {'✓' if count == 0 else '⚠'} {col:12s}: {count:5,} ({pct:5.2f}%)")
    
    # 标签分布
    if 'label' in df.columns:
        print(f"\n{'─' * 80}")
        print("  📊 标签分布:")
        for label, count in df['label'].value_counts().sort_index().items():
            pct = (count / len(df)) * 100
            name = "正面" if label == 1.0 else "负面" if label == 0.0 else str(label)
            print(f"    • {name:6s} (label={label}): {count:6,} ({pct:5.2f}%)")
    
    # 统计摘要 - 简化显示
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        print(f"\n{'─' * 80}")
        print("  📊 数值列统计摘要:")
        print(f"{'─' * 80}")
        with pd.option_context('display.precision', 3):
            print(df[numeric_cols].describe().to_string())
    
except Exception as e:
    print(f"\n❌ 无法转换为 DataFrame: {e}")

print_section("✅ 数据预览完成", "=")
