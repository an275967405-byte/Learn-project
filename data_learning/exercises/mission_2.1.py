from modelscope import MsDataset
from pandas import DataFrame

def load_dataset(dataset_name):
    dataset = MsDataset.load(dataset_name, split='train') 
    return dataset

ds = load_dataset('DAMO_NLP/jd')

df = DataFrame(list(ds))

# 1. 计算每条评论的字符数
df['text_length'] = df['sentence'].str.len()
print(f'文本长度:\n {df['text_length']}') 

# 2. 统计文本长度的分布（最小值、最大值、平均值、中位数）
print(f'文本长度统计信息:\n {df['text_length'].describe()}')

# 3. 找出最长和最短的评论
for i , length in df['text_length'].nlargest(1).items():
    print(f'最长评论: {df.at[i, 'sentence']} (长度: {length})')
for i , length in df['text_length'].nsmallest(1).items():
    print(f'最短评论: {df.at[i, 'sentence']} (长度: {length})')

# 4. 分析不同标签的文本长度差异
print(df.groupby('label')['text_length'].describe())

