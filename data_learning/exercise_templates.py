"""
AI数据处理分析师练习 - 代码模板
这些模板可以帮助你开始每个练习，但需要你完成TODO部分
"""

# ============================================================================
# 阶段一：数据下载与加载
# ============================================================================

def exercise_1_1_download_data():
    """练习1.1：下载并加载数据集"""
    from modelscope import MsDataset
    import pandas as pd
    
    # TODO: 下载数据集
    dataset = MsDataset.load('DAMO_NLP/jd', split='train')
    
    # TODO: 转换为DataFrame
    df = pd.DataFrame(list(dataset))
    
    # TODO: 查看基本信息
    print("数据形状:", df.shape)
    print("列名:", df.columns.tolist())
    print("数据类型:\n", df.dtypes)
    print("\n前5行数据:")
    print(df.head())
    
    return df


def exercise_1_2_basic_stats(df):
    """练习1.2：数据基本信息统计"""
    import pandas as pd
    
    # TODO: 统计总记录数
    total = len(df)
    print(f"总记录数: {total:,}")
    
    # TODO: 统计缺失值
    missing = df.isnull().sum()
    print("\n缺失值统计:")
    for col, count in missing.items():
        print(f"  {col}: {count} ({count/total*100:.2f}%)")
    
    # TODO: 统计标签分布
    label_counts = df['label'].value_counts()
    print("\n标签分布:")
    for label, count in label_counts.items():
        label_name = "正面" if label == 1.0 else "负面"
        print(f"  {label_name}: {count} ({count/total*100:.2f}%)")
    
    return missing, label_counts


def exercise_1_3_data_filtering(df):
    """练习1.3：数据筛选与切片"""
    import pandas as pd
    
    # TODO: 筛选正面评论
    positive = df[df['label'] == 1.0]
    print(f"正面评论: {len(positive)} 条")
    
    # TODO: 筛选负面评论
    negative = df[df['label'] == 0.0]
    print(f"负面评论: {len(negative)} 条")
    
    # TODO: 随机抽取100条
    sample_100 = df.sample(n=100, random_state=42)
    print(f"\n随机抽取100条数据")
    
    # TODO: 按标签分组，每组取10条
    grouped_sample = df.groupby('label').head(10)
    print(f"\n每组10条样本:")
    print(grouped_sample.groupby('label').size())
    
    return positive, negative, sample_100


# ============================================================================
# 阶段二：数据探索与统计
# ============================================================================

def exercise_2_1_text_length_analysis(df):
    """练习2.1：文本长度分析"""
    import pandas as pd
    
    # TODO: 计算文本长度
    df['text_length'] = df['sentence'].str.len()
    
    # TODO: 统计长度分布
    print("文本长度统计:")
    print(df['text_length'].describe())
    
    # TODO: 找出最长和最短的评论
    max_idx = df['text_length'].idxmax()
    min_idx = df['text_length'].idxmin()
    print(f"\n最长评论 (长度={df.loc[max_idx, 'text_length']}):")
    print(df.loc[max_idx, 'sentence'])
    print(f"\n最短评论 (长度={df.loc[min_idx, 'text_length']}):")
    print(df.loc[min_idx, 'sentence'])
    
    # TODO: 按标签分组统计长度
    print("\n不同标签的文本长度:")
    print(df.groupby('label')['text_length'].describe())
    
    return df


def exercise_2_2_data_quality_check(df):
    """练习2.2：数据质量检查"""
    import pandas as pd
    
    # TODO: 检查重复数据
    duplicates = df.duplicated().sum()
    print(f"完全重复的数据: {duplicates} 条")
    
    # TODO: 检查文本重复但标签不同
    text_duplicates = df[df.duplicated(subset=['sentence'], keep=False)]
    conflicts = text_duplicates.groupby('sentence')['label'].nunique()
    conflicts = conflicts[conflicts > 1]
    print(f"文本重复但标签不同: {len(conflicts)} 组")
    
    # TODO: 检查空文本
    empty_text = df['sentence'].isna() | (df['sentence'].str.strip() == '')
    print(f"空文本: {empty_text.sum()} 条")
    
    # TODO: 检查异常标签
    valid_labels = df['label'].isin([0.0, 1.0])
    invalid_labels = (~valid_labels).sum()
    print(f"异常标签: {invalid_labels} 条")
    
    # 生成报告
    print("\n数据质量报告:")
    print(f"  总记录数: {len(df):,}")
    print(f"  重复数据: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    print(f"  数据冲突: {len(conflicts)} 组")
    print(f"  空文本: {empty_text.sum()} ({empty_text.sum()/len(df)*100:.2f}%)")
    print(f"  异常标签: {invalid_labels}")
    
    return {
        'duplicates': duplicates,
        'conflicts': len(conflicts),
        'empty_text': empty_text.sum(),
        'invalid_labels': invalid_labels
    }


# ============================================================================
# 阶段三：数据清洗
# ============================================================================

def exercise_3_1_handle_missing_values(df):
    """练习3.1：处理缺失值"""
    import pandas as pd
    
    print("清洗前数据量:", len(df))
    
    # TODO: 策略1 - 删除包含缺失值的行
    df_drop = df.dropna()
    print(f"删除缺失值后: {len(df_drop)} 条 (删除了 {len(df)-len(df_drop)} 条)")
    
    # TODO: 策略2 - 填充缺失值
    df_fill = df.copy()
    df_fill['sentence'] = df_fill['sentence'].fillna('')
    df_fill['label'] = df_fill['label'].fillna(df_fill['label'].mode()[0] if len(df_fill['label'].mode()) > 0 else 0.0)
    print(f"填充缺失值后: {len(df_fill)} 条")
    
    # 选择策略（这里选择删除，因为缺失值通常很少）
    df_clean = df_drop
    
    return df_clean


def exercise_3_2_remove_duplicates(df):
    """练习3.2：去除重复数据"""
    import pandas as pd
    
    print("去重前数据量:", len(df))
    
    # TODO: 去除完全重复
    df_no_dup = df.drop_duplicates()
    print(f"去除完全重复后: {len(df_no_dup)} 条")
    
    # TODO: 去除文本重复（保留第一条）
    df_final = df_no_dup.drop_duplicates(subset=['sentence'], keep='first')
    print(f"去除文本重复后: {len(df_final)} 条")
    
    # TODO: 保存到CSV
    df_final.to_csv('jd_cleaned.csv', index=False, encoding='utf-8')
    print("已保存清洗后的数据到 jd_cleaned.csv")
    
    return df_final


def exercise_3_3_text_normalization(df):
    """练习3.3：文本标准化"""
    import pandas as pd
    import re
    
    df_clean = df.copy()
    
    # TODO: 去除首尾空格
    df_clean['sentence'] = df_clean['sentence'].str.strip()
    
    # TODO: 去除多余空格
    df_clean['sentence'] = df_clean['sentence'].str.replace(r'\s+', ' ', regex=True)
    
    # TODO: 去除异常长的文本（超过500字符）
    long_text_mask = df_clean['sentence'].str.len() > 500
    print(f"发现 {long_text_mask.sum()} 条超长文本，将截断")
    df_clean.loc[long_text_mask, 'sentence'] = df_clean.loc[long_text_mask, 'sentence'].str[:500]
    
    # TODO: 去除空文本
    df_clean = df_clean[df_clean['sentence'].str.strip() != '']
    
    print(f"文本标准化后: {len(df_clean)} 条")
    
    return df_clean


# ============================================================================
# 阶段四：数据可视化
# ============================================================================

def exercise_4_1_label_distribution_plot(df):
    """练习4.1：标签分布可视化"""
    import matplotlib.pyplot as plt
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    label_counts = df['label'].value_counts().sort_index()
    
    # TODO: 绘制柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 柱状图
    labels = ['负面', '正面']
    values = [label_counts[0.0], label_counts[1.0]]
    ax1.bar(labels, values, color=['#ff6b6b', '#51cf66'])
    ax1.set_title('标签分布 - 柱状图', fontsize=14)
    ax1.set_ylabel('数量')
    for i, v in enumerate(values):
        ax1.text(i, v, str(v), ha='center', va='bottom')
    
    # TODO: 绘制饼图
    ax2.pie(values, labels=labels, autopct='%1.1f%%', 
            colors=['#ff6b6b', '#51cf66'], startangle=90)
    ax2.set_title('标签分布 - 饼图', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
    print("图表已保存为 label_distribution.png")
    plt.show()


def exercise_4_2_text_length_distribution(df):
    """练习4.2：文本长度分布可视化"""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if 'text_length' not in df.columns:
        df['text_length'] = df['sentence'].str.len()
    
    # TODO: 绘制直方图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 整体分布
    axes[0, 0].hist(df['text_length'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('文本长度分布（整体）')
    axes[0, 0].set_xlabel('字符数')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].axvline(df['text_length'].mean(), color='red', linestyle='--', label=f'均值: {df["text_length"].mean():.1f}')
    axes[0, 0].legend()
    
    # 正负面对比
    positive = df[df['label'] == 1.0]['text_length']
    negative = df[df['label'] == 0.0]['text_length']
    
    axes[0, 1].hist([negative, positive], bins=30, label=['负面', '正面'], 
                    color=['#ff6b6b', '#51cf66'], alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('文本长度分布（按标签）')
    axes[0, 1].set_xlabel('字符数')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].legend()
    
    # TODO: 箱线图
    data_to_plot = [negative, positive]
    bp = axes[1, 0].boxplot(data_to_plot, labels=['负面', '正面'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff6b6b')
    bp['boxes'][1].set_facecolor('#51cf66')
    axes[1, 0].set_title('文本长度箱线图')
    axes[1, 0].set_ylabel('字符数')
    
    # 统计信息
    stats_text = f"""
    整体统计:
    均值: {df['text_length'].mean():.1f}
    中位数: {df['text_length'].median():.1f}
    
    正面评论:
    均值: {positive.mean():.1f}
    中位数: {positive.median():.1f}
    
    负面评论:
    均值: {negative.mean():.1f}
    中位数: {negative.median():.1f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                    verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('统计信息')
    
    plt.tight_layout()
    plt.savefig('text_length_distribution.png', dpi=150, bbox_inches='tight')
    print("图表已保存为 text_length_distribution.png")
    plt.show()


# ============================================================================
# 阶段五：文本预处理
# ============================================================================

def exercise_5_1_chinese_segmentation(df):
    """练习5.1：中文分词基础"""
    import jieba
    from collections import Counter
    
    # TODO: 对评论进行分词
    sample_text = df['sentence'].iloc[0]
    print(f"原文: {sample_text}")
    
    words = jieba.cut(sample_text)
    word_list = list(words)
    print(f"分词结果: {' / '.join(word_list)}")
    
    # TODO: 统计词频
    all_words = []
    for text in df['sentence']:
        words = jieba.cut(str(text))
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    print(f"\n最常见的20个词:")
    for word, freq in word_freq.most_common(20):
        print(f"  {word}: {freq}")
    
    # TODO: 分析正负面评论的高频词差异
    positive_words = []
    negative_words = []
    
    for idx, row in df.iterrows():
        words = list(jieba.cut(str(row['sentence'])))
        if row['label'] == 1.0:
            positive_words.extend(words)
        else:
            negative_words.extend(words)
    
    pos_freq = Counter(positive_words)
    neg_freq = Counter(negative_words)
    
    print("\n正面评论高频词（前10）:")
    for word, freq in pos_freq.most_common(10):
        print(f"  {word}: {freq}")
    
    print("\n负面评论高频词（前10）:")
    for word, freq in neg_freq.most_common(10):
        print(f"  {word}: {freq}")
    
    return word_freq, pos_freq, neg_freq


# ============================================================================
# 主函数 - 运行所有练习
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AI数据处理分析师练习 - 代码模板")
    print("=" * 80)
    
    # 阶段一：数据下载与加载
    print("\n【阶段一：数据下载与加载】")
    df = exercise_1_1_download_data()
    
    missing, label_counts = exercise_1_2_basic_stats(df)
    
    positive, negative, sample = exercise_1_3_data_filtering(df)
    
    print("\n✅ 阶段一完成！")
    print("\n" + "=" * 80)
