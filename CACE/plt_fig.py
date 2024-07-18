import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def to_percent(temp, position):
    return "0" if temp == 0 else '%2.f'%(temp) + '%'

# 定义文件路径
file_path_ori = '/home/hexiang/CACE-Complexity/Exps/Supv/expComplexity_Seed3917_guide_Audio-Guide_psai_0.3_Contrastive_False_contras-coeff_1.0_lambda_0.0/202406140231_seed_3917_contrastive_False_guide_Audio-Guide_lambda_0.0_contras-coeff_1.0_psai_0.3.log'
file_path_ours = '/home/hexiang/CACE/Exps/Supv_supp/expComplexity_Seed3917_guide_Co-Guide_psai_0.3_Contrastive_True_contras-coeff_1.0_lambda_0.6/202406140231_seed_3917_contrastive_True_guide_Co-Guide_lambda_0.6_contras-coeff_1.0_psai_0.3.log'

# 解析日志数据
def parse_log(file_path):
    epochs_data = []
    accuracy_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "completed in" in line:
                parts = line.split()
                time_completed = float(parts[-2])
                epochs_data.append(time_completed)
            elif "Evaluation results (acc):" in line:
                parts = line.split()
                accuracy = float(parts[-1][:-2])  # Remove the last % character
                accuracy_data.append(accuracy)
    return epochs_data, accuracy_data

# 聚合数据，计算15分钟间隔的准确率
def aggregate_data(accuracy_data, interval):
    aggregated_accuracies = []
    for idx, duration in enumerate(accuracy_data):
        if idx % interval == 0:  # 1min的idx
            aggregated_accuracies.append(accuracy_data[idx])
    aggregated_accuracies.insert(0, 0.0)
    return aggregated_accuracies

# 绘图
def plot_data(aggregated_accuracies_ori, aggregated_accuracies_ours):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots(figsize=(18, 5))
    legend_list = ['Vanilla training method (Baseline)', "Co-guidance and Contrastive Enhancement (Ours)"]

    for i in range(2):
        if i == 0:
            ax.plot(range(0, len(aggregated_accuracies_ori)), aggregated_accuracies_ori, linewidth=4, label=legend_list[i])
        else:
            ax.plot(range(0, len(aggregated_accuracies_ours)), aggregated_accuracies_ours, linewidth=4, label=legend_list[i])

    ax.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0, fontsize=24)
    # ax.legend(loc='lower right', frameon=True, framealpha=0.9, shadow=True, borderpad=1)

    # 放大开始阶段
    axins = ax.inset_axes([0.1, 0.47, 0.3, 0.25])  # 调整这些参数以适应你的需要
    axins.plot(range(0, len(aggregated_accuracies_ori)), aggregated_accuracies_ori, linewidth=2)
    axins.plot(range(0, len(aggregated_accuracies_ours)), aggregated_accuracies_ours, linewidth=2)
    axins.set_xlim(1, 12.4)  # 放大到训练的前0.2小时
    axins.set_ylim(60, 84)  # 调整为关注的y轴区间
    # 设置内嵌图的x轴刻度
    axins.set_xticks(list(np.linspace(1, 12, num=5)))  # 显示75%和80%
    axins.set_xticklabels(['0.0', '0.05', '0.1', '0.15', '0.2'])  # 为刻度标签添加%符号
    # 设置内嵌图的y轴刻度
    axins.set_yticks([60, 70, 80])  # 显示75%和80%
    axins.set_yticklabels(['60%', '70%', '80%'])  # 为刻度标签添加%符号
    axins.set_title('Zoom on initial epochs', fontsize=20)
    # Set tick parameters for both axes
    axins.tick_params(axis='x', labelsize=23)  # Set x axis tick label size
    axins.tick_params(axis='y', labelsize=15)  # Set y axis tick label size
    ax.indicate_inset_zoom(axins)



    # Set tick parameters for both axes
    ax.tick_params(axis='x', labelsize=30)  # Set x axis tick label size
    ax.tick_params(axis='y', labelsize=30)  # Set y axis tick label size

    plt.xticks(np.linspace(0, 31, num=6), ['0', '0.1', '0.2', '0.3', '0.4', '0.5'])

    # 在axins上标注红色星号
    ax.plot([10], [aggregated_accuracies_ours[9]], 'r*', markersize=15)  # 'r*' 表示红色星号
    # 在axins上标注色绿色号
    ax.plot([6], [aggregated_accuracies_ori[5]], 'g*', markersize=15)  # 'r*' 表示红色星号

    # 在axins上标注红色星号
    ax.plot([20], 63, 'r*', markersize=15)  # 'r*' 表示红色星号
    # 在axins上标注色绿色号
    ax.plot([20], 53, 'g*', markersize=15)  # 'r*' 表示红色星号

    # 添加注释
    ax.annotate('Highest accuracy: 80.80%',  # 显示的文本
                   xy=(20, 62),  # 点的坐标
                   xytext=(20.3, 61.3),  # 文本的坐标
                   textcoords='data',  # 坐标系
                   fontsize=20,  # 文本大小
                   color='red')  # 文本颜色

    ax.annotate('Highest accuracy: 77.83%',  # 显示的文本
                   xy=(20, 52),  # 点的坐标
                   xytext=(20.3, 51),  # 文本的坐标
                   textcoords='data',  # 坐标系
                   fontsize=20,  # 文本大小
                   color='green')  # 文本颜色

    ax.set_ylim([0, 83.5])  # Set the limits for the y-axis
    ax.set_xlim([0, 31])  # Set the limits for the x-axis
    plt.xlabel('Training hours', fontsize=42)
    plt.ylabel('Test Accuracy', fontsize=42)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # plt.show()
    plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
    plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
    plt.tight_layout()
    plt.savefig('re.pdf', bbox_inches='tight', dpi=300)

# 主函数
def main():
    show_epoch = 31
    epochs_data, accuracy_data = parse_log(file_path_ori)
    aggregated_accuracies_ori = aggregate_data(accuracy_data, 6)
    aggregated_accuracies_ori = aggregated_accuracies_ori[:show_epoch]


    epochs_data, accuracy_data = parse_log(file_path_ours)
    aggregated_accuracies_ours = aggregate_data(accuracy_data, 5)
    aggregated_accuracies_ours = aggregated_accuracies_ours[:show_epoch]

    plot_data(aggregated_accuracies_ori, aggregated_accuracies_ours)

# 调用主函数
main()
