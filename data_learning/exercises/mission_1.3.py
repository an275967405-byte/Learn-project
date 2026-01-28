from modelscope import MsDataset
from pandas import DataFrame

def load_dataset(dataset_name):
    dataset = MsDataset.load(dataset_name, split='train') 
    return dataset

ds = load_dataset('DAMO_NLP/jd')

df = DataFrame(list(ds))

good_ = df[df['label'] == 1.0]
bad_ = df[df['label'] == 0.0]
print(f'正面评论数量: {len(good_)}')
print(f'负面评论数量: {len(bad_)}')

sample_100 = df.sample(n=100,random_state=24)
print(f'随机抽取100条数据:{sample_100}')

group_sample = df.groupby('label').apply(lambda x: x.sample(n=10, random_state=24))
print(f'按标签分组，每组各取10条样本:{group_sample.groupby('label').size()}')

# 1. 筛选出所有正面评论（label=1.0）
# 2. 筛选出所有负面评论（label=0.0）
# 3. 随机抽取100条数据进行快速分析
# 4. 按标签分组，每组各取10条样本