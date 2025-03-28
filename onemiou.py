import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_miou(pred_mask, true_mask, num_classes, class_names=None):
    """
    计算并可视化单张图片的mIoU指标
    参数:
        pred_mask: 预测的掩码图像（numpy数组，值为类别索引）
        true_mask: 真实的掩码图像（numpy数组，值为类别索引）
        num_classes: 类别总数（包含背景）
        class_names: 可选的类别名称列表
    返回:
        miou: 平均交并比
        class_iou: 每个类别的IoU
        metrics: 包含详细指标的字典
    """
    # 验证输入
    assert pred_mask.shape == true_mask.shape, "预测掩码和真实掩码尺寸不一致"
    assert pred_mask.dtype == true_mask.dtype, "数据类型不一致"
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_mask.flatten(), pred_mask.flatten(), labels=range(num_classes))
    
    # 计算各类IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
    iou = np.zeros_like(intersection, dtype=float)
    np.divide(intersection, union, out=iou, where=union!=0)  # 修正这里
    
    miou = np.nanmean(iou)
    
    # 计算其他指标
    pixel_accuracy = np.diag(cm).sum() / cm.sum()
    class_accuracy = np.zeros_like(np.diag(cm), dtype=float)
    np.divide(np.diag(cm), np.sum(cm, axis=1), out=class_accuracy, where=np.sum(cm, axis=1)!=0)
    
    # 组织结果
    metrics = {
        'miou': miou,
        'pixel_accuracy': pixel_accuracy,
        'class_iou': dict(zip(range(num_classes), iou)),
        'class_accuracy': dict(zip(range(num_classes), class_accuracy)),
        'confusion_matrix': cm
    }
    
    # 可视化结果
    visualize_results(pred_mask, true_mask, metrics, num_classes, class_names)
    
    return miou, iou, metrics

def visualize_results(pred_mask, true_mask, metrics, num_classes, class_names=None):
    """可视化预测结果与评估指标"""
    plt.figure(figsize=(18, 12))
    
    # 1. 显示原始预测和真实掩码
    plt.subplot(2, 3, 1)
    plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=num_classes-1)
    plt.title('Predicted Mask')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(2, 3, 2)
    plt.imshow(true_mask, cmap='jet', vmin=0, vmax=num_classes-1)
    plt.title('True Mask')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # 2. 显示差异图
    diff = np.where(pred_mask != true_mask, 1, 0)
    plt.subplot(2, 3, 3)
    plt.imshow(diff, cmap='gray')
    plt.title(f'Difference (Error Rate: {np.mean(diff)*100:.2f}%)')
    
    # 3. 显示各类IoU
    plt.subplot(2, 3, 4)
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    plt.bar(range(num_classes), list(metrics['class_iou'].values()))
    plt.xticks(range(num_classes), class_names, rotation=90)
    plt.title(f'Per-Class IoU (mIoU: {metrics["miou"]:.4f})')
    plt.ylim(0, 1)
    
    # 4. 显示混淆矩阵（归一化）
    plt.subplot(2, 3, 5)
    cm_normalized = metrics['confusion_matrix'].astype('float') / metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def load_and_preprocess_mask(path, resize_to=None):
    """加载并预处理掩码图像"""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法加载图像: {path}")
    
    if resize_to is not None:
        mask = cv2.resize(mask, resize_to, interpolation=cv2.INTER_NEAREST)
    
    return mask

if __name__ == "__main__":
    # 用户输入
    pred_path = input("请输入预测掩码路径: ").strip()
    true_path = input("请输入真实掩码路径: ").strip()
    num_classes = int(input("请输入类别数量(包含背景): "))
    
    # 可选: 输入类别名称
    class_names = None
    if input("是否提供类别名称? (y/n): ").lower() == 'y':
        class_names = input("请输入类别名称(逗号分隔): ").split(',')
        class_names = [name.strip() for name in class_names]
        assert len(class_names) == num_classes, "类别数量不匹配"
    
    # 加载图像
    try:
        pred_mask = load_and_preprocess_mask(pred_path)
        true_mask = load_and_preprocess_mask(true_path, resize_to=(pred_mask.shape[1], pred_mask.shape[0]))
        
        # 计算指标
        miou, class_iou, metrics = calculate_miou(pred_mask, true_mask, num_classes, class_names)
        
        # 打印结果
        print("\n" + "="*50)
        print(f"mIoU: {miou:.4f}")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print("\n各类别IoU:")
        for i, iou in enumerate(class_iou):
            name = class_names[i] if class_names else f'Class {i}'
            print(f"{name}: {iou:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")