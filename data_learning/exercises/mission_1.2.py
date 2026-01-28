from modelscope import MsDataset
from pandas import DataFrame

def load_dataset(dataset_name):
    dataset = MsDataset.load(dataset_name, split='train') 
    return dataset

ds = load_dataset('DAMO_NLP/jd')

df = DataFrame(list(ds))

total = len(df)
print(f'数据总量{total}')

null_counts = df.isnull().sum()
for col, count in null_counts.items():
    print(f'列 {col} ,缺失值数量: {count},缺失值占比{(count/total)*100:.2f}%')

label_count = df['label'].value_counts()
for label, count in label_count.items():
    label_name = '正面' if label == 1 else '负面'
    print(f'{label_name} 数量: {count},占比{(count/total)*100:.2f}%')

