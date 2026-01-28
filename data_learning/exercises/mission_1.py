from modelscope import MsDataset
from pandas import DataFrame

def load_dataset(dataset_name):
    dataset = MsDataset.load(dataset_name, split='train') 
    return dataset

ds = load_dataset('DAMO_NLP/jd')

df = DataFrame(list(ds))


print(f'数据形状{df.shape}')
print(f'列名{df.columns}')
print(f'数据类型{df.dtypes}')  

