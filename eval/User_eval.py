import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

correctness_map = {
    'Incorrect': 1,
    'Partially Correct': 2,
    'Mostly Correct': 3,
    'Correct': 4,
    'Perfect': 5
}

logic_map = {
    'Illogical': 1,
    'Weak Logic': 2,
    'Moderate Logic': 3,
    'Strong Logic': 4,
    'Flawless Logic': 5
}

sensitivity_map = {
    'Insensitive': 1,
    'Low Sensitivity': 2,
    'Moderate Sensitivity': 3,
    'High Sensitivity': 4,
    'Exceptional Sensitivity': 5
}

usefulness_map = {
    'Useless': 1,
    'Limited Use': 2,
    'Moderately Useful': 3,
    'Very Useful': 4,
    'Indispensable': 5
}

# 将 N/A 替换为 NaN，并统一格式
def map_labels(column, mapping):
    return column.replace({'N/A': np.nan}).map(mapping).astype('float')

def To_score(df):
    # 对每一列进行处理
    return_df = df
    return_df['Correctness'] = map_labels(return_df['Correctness'], correctness_map)
    return_df['Logicality'] = map_labels(return_df['Logicality'], logic_map)
    return_df['Sensitivity'] = map_labels(return_df['Sensitivity'], sensitivity_map)
    return_df['Practicality'] = map_labels(return_df['Practicality'], usefulness_map)
    # 查看结果
    return return_df


def read_and_split_excel(file_path):
    # 读取Excel文件，跳过前两行
    df = pd.read_excel(file_path, header=None, skiprows=2)
    # 获取第三行的标题（现在是第0行，因为跳过了前两行）
    headers = df.iloc[0].tolist()
    # 从第四行开始是实际数据（现在是第1行及以后）
    data = df.iloc[1:]
    # 去除第一列（序号列）
    data = data.iloc[:, 1:]  # 保留所有行，从第2列开始
    headers = headers[1:]  # 标题也去除第一个
    # 确定每组有多少列（假设每组3列，因为去除了序号列）
    group_size = 4
    num_groups = len(headers) // group_size
    # 拆分DataFrame
    dfs = []
    for i in range(num_groups):
        start_col = i * group_size
        end_col = (i + 1) * group_size
        # 提取当前组的列
        group_df = data.iloc[:, start_col:end_col].copy()
        # 设置列名（使用原始标题）
        group_headers = headers[start_col:end_col]
        group_df.columns = group_headers
        # 添加到结果列表
        dfs.append(group_df)
    return dfs


def plot_boxplots(score_df):
    # 设置绘图风格
    sns.set(style="whitegrid", font_scale=1.2)

    # 创建子图（4个维度）
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    dimensions = score_df['维度'].unique()

    # 自定义中位线样式（红色加粗）
    medianprops = dict(linestyle='-', linewidth=3, color='red')

    # 学术风格浅色配色（蓝、灰、绿）
    palette = {
        'Model1': '#a2cffe',   # 浅蓝色
        'Model2': '#d3d3d3',   # 浅灰色
        'Model3': '#b3e6cc'    # 浅绿色
    }

    for i, dim in enumerate(dimensions):
        ax = axes[i]
        subset = score_df[score_df['维度'] == dim]

        # 绘制箱线图 + 自定义样式
        sns.boxplot(
            x='Model',
            y='Score',
            data=subset,
            ax=ax,
            palette=palette,             # 使用自定义浅色配色
            medianprops=medianprops,     # 红色加粗中位线
            showfliers=False,            # 隐藏异常值点
            linewidth=1.5                # 箱体边框线宽
        )
        ax.set_title(dim, fontsize=14, pad=10)
        ax.set_ylabel('Score (1–5)', fontsize=12)
        ax.set_ylim(0.5, 5.5)            # 固定 Y 轴范围
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=11)

    # 自动调整布局，防止重叠
    plt.tight_layout(pad=3.0)
    plt.show()


def main():
    file_dir = '../results/RQ4'
    files = os.listdir(file_dir)

    # 存储所有评分者对每个模型的聚合数据
    all_model_results = []
    all_scores = []  # 用于存储所有原始评分，用于绘图

    for file in files:
        file_path = f'{file_dir}/{file}'
        result_dfs = read_and_split_excel(file_path)  # 返回 15 个 df，顺序为 model1×5, model2×5, model3×5

        # 拆分成三个模型的 df 列表
        model1_dfs = result_dfs[0:5]
        model2_dfs = result_dfs[5:10]
        model3_dfs = result_dfs[10:15]

        # 定义处理模型数据的函数
        def process_model_dfs(model_dfs):
            scored_dfs = []
            for df in model_dfs:
                scored_df = To_score(df)  # 标签 → 数值分数
                scored_dfs.append(scored_df)
            if not scored_dfs:
                return pd.DataFrame()  # 防止空列表报错
            return pd.concat(scored_dfs, ignore_index=True)  # 合并成一个大 df

        # 获取每个模型的合并 df（含所有5轮的数据）
        model1_df = process_model_dfs(model1_dfs)
        model2_df = process_model_dfs(model2_dfs)
        model3_df = process_model_dfs(model3_dfs)

        # 收集原始评分用于绘图（长格式）
        def collect_scores(df, model_name):
            for dim in df.columns:
                scores = df[dim].dropna()
                for score in scores:
                    all_scores.append({
                        'Model': model_name,
                        '维度': dim,
                        'Score': score
                    })

        collect_scores(model1_df, 'Model1')
        collect_scores(model2_df, 'Model2')
        collect_scores(model3_df, 'Model3')

        # 统计函数
        def get_stats(df, model_name):
            stats = {}
            for col in df.columns:
                scores = df[col]
                valid_scores = scores.dropna()
                stats[col] = {
                    'Model': model_name,
                    'Dim': col,
                    'Mean': round(scores.mean(), 2),
                    'Median':round(scores.median(),2),
                    'Std': round(scores.std(), 2),
                    'High_Rate': round((valid_scores >= 4).mean() * 100, 2),
                    'Median_Rate':round((valid_scores >= 3).mean() * 100, 2),
                    'Low_Rate':round((valid_scores < 2).mean() * 100, 2),
                    # 'High_Rate': round(((scores >= 4) & pd.notna(scores)).mean() * 100, 2),
                    'Missing_Rate': round(scores.isna().mean() * 100, 2)
                }
            return stats

        # 提取每个模型的统计信息
        model1_stats = get_stats(model1_df, 'Model1')
        model2_stats = get_stats(model2_df, 'Model2')
        model3_stats = get_stats(model3_df, 'Model3')

        # 合并成一个 DataFrame 并保存
        all_stats = []
        for stat_dict in [model1_stats, model2_stats, model3_stats]:
            for dim, values in stat_dict.items():
                all_stats.append(values)

        stats_df = pd.DataFrame(all_stats)
        all_model_results.append(stats_df)

    # 合并所有评分者的统计结果
    final_stats = pd.concat(all_model_results, axis=0)

    # 对每个 (Model, 维度) 取平均
    final_stats = final_stats.groupby(['Model', 'Dim']).mean(numeric_only=True).reset_index()

    print("📊 统计结果：")
    print(final_stats)

    # 将统计结果保存到 CSV
    final_stats.to_csv('../results/RQ4_summary.csv', index=False)

    # 构建绘图用的 DataFrame
    score_df = pd.DataFrame(all_scores)

    # 确保有数据才绘图
    if not score_df.empty:
        plot_boxplots(score_df)
    else:
        print("⚠️ 没有评分数据，跳过绘图。")

# def test():
#     file_path = '../results/RQ4/answer_0.xlsx'  # 替换为你的Excel文件路径
#     result_dfs = read_and_split_excel(file_path)
#     test_df = result_dfs[14]
#     print(test_df.head())
#     To_score(test_df)
main()

