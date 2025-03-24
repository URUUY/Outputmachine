import numpy as np

# 读取结果文件
result_file = "accuracy_results2.txt"

# 初始化存储每个模型准确率的列表
model1_accuracies = []
model2_accuracies = []
model3_accuracies = []
model4_accuracies = []
majority_voting_accuracies = []
weighted_voting_accuracies = []

# 读取文件并提取准确率
with open(result_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines[1:]:  # 跳过标题行
        parts = line.strip().split("\t")
        if len(parts) != 7:  # 确保每行有 7 列数据
            continue
        model1_accuracies.append(float(parts[1]))
        model2_accuracies.append(float(parts[2]))
        model3_accuracies.append(float(parts[3]))
        model4_accuracies.append(float(parts[4]))
        majority_voting_accuracies.append(float(parts[5]))
        weighted_voting_accuracies.append(float(parts[6]))

# 将列表转换为 numpy 数组
model1_accuracies = np.array(model1_accuracies)
model2_accuracies = np.array(model2_accuracies)
model3_accuracies = np.array(model3_accuracies)
model4_accuracies = np.array(model4_accuracies)
majority_voting_accuracies = np.array(majority_voting_accuracies)
weighted_voting_accuracies = np.array(weighted_voting_accuracies)

# 计算统计指标
def calculate_statistics(accuracies, model_name):
    mean_accuracy = np.mean(accuracies)
    max_accuracy = np.max(accuracies)
    min_accuracy = np.min(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"{model_name}:")
    print(f"  - 平均准确率: {mean_accuracy:.2f}%")
    print(f"  - 最高准确率: {max_accuracy:.2f}%")
    print(f"  - 最低准确率: {min_accuracy:.2f}%")
    print(f"  - 标准差: {std_accuracy:.2f}%")
    print()

# 输出各个模型的统计结果
calculate_statistics(model1_accuracies, "Model 1")
calculate_statistics(model2_accuracies, "Model 2")
calculate_statistics(model3_accuracies, "Model 3")
calculate_statistics(model4_accuracies, "Model 4")
calculate_statistics(majority_voting_accuracies, "Majority Voting")
calculate_statistics(weighted_voting_accuracies, "Weighted Voting")

# 将统计结果保存到文件
with open("model_statistics.txt", "w", encoding="utf-8") as f:
    f.write("Model\tAverage Accuracy\tMax Accuracy\tMin Accuracy\tStd Deviation\n")
    f.write(f"Model 1\t{np.mean(model1_accuracies):.2f}\t{np.max(model1_accuracies):.2f}\t{np.min(model1_accuracies):.2f}\t{np.std(model1_accuracies):.2f}\n")
    f.write(f"Model 2\t{np.mean(model2_accuracies):.2f}\t{np.max(model2_accuracies):.2f}\t{np.min(model2_accuracies):.2f}\t{np.std(model2_accuracies):.2f}\n")
    f.write(f"Model 3\t{np.mean(model3_accuracies):.2f}\t{np.max(model3_accuracies):.2f}\t{np.min(model3_accuracies):.2f}\t{np.std(model3_accuracies):.2f}\n")
    f.write(f"Model 4\t{np.mean(model4_accuracies):.2f}\t{np.max(model4_accuracies):.2f}\t{np.min(model4_accuracies):.2f}\t{np.std(model4_accuracies):.2f}\n")
    f.write(f"Majority Voting\t{np.mean(majority_voting_accuracies):.2f}\t{np.max(majority_voting_accuracies):.2f}\t{np.min(majority_voting_accuracies):.2f}\t{np.std(majority_voting_accuracies):.2f}\n")
    f.write(f"Weighted Voting\t{np.mean(weighted_voting_accuracies):.2f}\t{np.max(weighted_voting_accuracies):.2f}\t{np.min(weighted_voting_accuracies):.2f}\t{np.std(weighted_voting_accuracies):.2f}\n")