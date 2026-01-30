from modelscope import MsDataset
from pandas import DataFrame

def load_dataset(dataset_name):
    dataset = MsDataset.load(dataset_name, split='train')
    return dataset

ds = load_dataset('DAMO_NLP/jd')

df = DataFrame(list(ds))

# 1. 检查重复数据
df_duplicated = df[df.duplicated()]
# print(f'重复数据数量: {len(df_duplicated)}')

text_duplicated = df[df['sentence'].duplicated(keep=False)]
conflicts = text_duplicated.groupby('sentence')['label'].nunique()
conflict_sentences = conflicts[conflicts > 1]
# print(f'文本重复标签是：{conflict_sentences}\n有{len(conflict_sentences)}组')

# 2. 检查空值/缺失值
df_null = df.isnull().sum()
# print(f'缺失值数量: {df_null.to_dict()}')


# 3. 检查异常标签值（不是0.0或1.0的）
valid_labels = df['label'].isin([0.0, 1.0])
# print(f'异常标签值数量: {len(df[~valid_labels])}')
# print(f'异常标签值: {(df[~valid_labels])}')

# 4. 检查文本是否为空或只包含空格
empty_texts = df['sentence'].str.strip() == ''
# print(f'空文本数量: {len(df[empty_texts])}')


# 5. 生成数据质量报告
data_quality_report = {
    'total_records': len(df),
    'duplicate_records': len(df_duplicated),
    'missing_values': df_null.to_dict(),
    'invalid_labels': len(df[~valid_labels]),
    'empty_texts': len(df[empty_texts])
}
print(f'数据质量报告:\n {data_quality_report}')