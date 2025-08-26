#!/usr/bin/env python3
"""
读取TensorBoard日志文件并提取训练信息
"""
import os
import sys
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt

def read_tensorboard_logs(log_dir):
    """读取TensorBoard日志文件"""
    if not os.path.exists(log_dir):
        print(f"日志目录不存在: {log_dir}")
        return None
    
    # 获取最新的日志文件
    log_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not log_files:
        print("未找到TensorBoard日志文件")
        return None
    
    # 按修改时间排序，获取最新的
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
    latest_log = os.path.join(log_dir, log_files[0])
    
    print(f"正在读取日志文件: {latest_log}")
    
    try:
        ea = event_accumulator.EventAccumulator(latest_log)
        ea.Reload()
        
        # 获取所有可用的标签
        tags = ea.Tags()
        print(f"可用的标签: {tags}")
        
        # 提取训练数据
        data = {}
        for tag in tags['scalars']:
            events = ea.Scalars(tag)
            data[tag] = {
                'steps': [event.step for event in events],
                'values': [event.value for event in events],
                'wall_times': [event.wall_time for event in events]
            }
        
        return data
    
    except Exception as e:
        print(f"读取日志文件时出错: {e}")
        return None

def analyze_training_logs(data):
    """分析训练日志数据"""
    if not data:
        return
    
    print("\n=== 训练日志分析 ===")
    
    # 分析损失函数
    if 'Loss/Train' in data and 'Loss/Val' in data:
        train_loss = data['Loss/Train']['values']
        val_loss = data['Loss/Val']['values']
        epochs = data['Loss/Train']['steps']
        
        print(f"\n训练轮数: {len(epochs)}")
        print(f"最终训练损失: {train_loss[-1]:.4f}")
        print(f"最终验证损失: {val_loss[-1]:.4f}")
        print(f"最佳验证损失: {min(val_loss):.4f} (第{val_loss.index(min(val_loss))+1}轮)")
        
        # 检查过拟合
        if len(train_loss) > 10:
            recent_train = train_loss[-10:]
            recent_val = val_loss[-10:]
            if min(recent_val) > max(recent_train):
                print("⚠️  可能存在过拟合现象")
    
    # 分析准确率
    if 'AccuracyTop1/Train' in data and 'AccuracyTop1/Val' in data:
        train_acc = data['AccuracyTop1/Train']['values']
        val_acc = data['AccuracyTop1/Val']['values']
        
        print(f"\n最终训练Top1准确率: {train_acc[-1]:.6f}")
        print(f"最终验证Top1准确率: {val_acc[-1]:.6f}")
        print(f"最佳验证Top1准确率: {max(val_acc):.6f} (第{val_acc.index(max(val_acc))+1}轮)")
    
    # 分析ArcFace损失
    if 'ArcFace/Train' in data and 'ArcFace/Val' in data:
        train_arcface = data['ArcFace/Train']['values']
        val_arcface = data['ArcFace/Val']['values']
        
        print(f"\n最终训练ArcFace损失: {train_arcface[-1]:.4f}")
        print(f"最终验证ArcFace损失: {val_arcface[-1]:.4f}")
    
    # 分析Triplet损失
    if 'Triplet/Train' in data and 'Triplet/Val' in data:
        train_triplet = data['Triplet/Train']['values']
        val_triplet = data['Triplet/Val']['values']
        
        print(f"\n最终训练Triplet损失: {train_triplet[-1]:.4f}")
        print(f"最终验证Triplet损失: {val_triplet[-1]:.4f}")
    
    # 分析学习率
    if 'LR' in data:
        lr_values = data['LR']['values']
        print(f"\n初始学习率: {lr_values[0]:.6f}")
        print(f"最终学习率: {lr_values[-1]:.6f}")
        print(f"学习率衰减次数: {len([i for i in range(1, len(lr_values)) if lr_values[i] < lr_values[i-1]])}")

def plot_training_curves(data, save_path=None):
    """绘制训练曲线"""
    if not data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Iris Recognition Model Training Curves', fontsize=16)
    
    # 损失曲线
    if 'Loss/Train' in data and 'Loss/Val' in data:
        axes[0, 0].plot(data['Loss/Train']['steps'], data['Loss/Train']['values'], label='Train Loss')
        axes[0, 0].plot(data['Loss/Val']['steps'], data['Loss/Val']['values'], label='Val Loss')
        axes[0, 0].set_title('Loss Function')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # 准确率曲线
    if 'AccuracyTop1/Train' in data and 'AccuracyTop1/Val' in data:
        axes[0, 1].plot(data['AccuracyTop1/Train']['steps'], data['AccuracyTop1/Train']['values'], label='Train Top1 Acc')
        axes[0, 1].plot(data['AccuracyTop1/Val']['steps'], data['AccuracyTop1/Val']['values'], label='Val Top1 Acc')
        axes[0, 1].set_title('Top1 Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # ArcFace损失曲线
    if 'ArcFace/Train' in data and 'ArcFace/Val' in data:
        axes[1, 0].plot(data['ArcFace/Train']['steps'], data['ArcFace/Train']['values'], label='Train ArcFace Loss')
        axes[1, 0].plot(data['ArcFace/Val']['steps'], data['ArcFace/Val']['values'], label='Val ArcFace Loss')
        axes[1, 0].set_title('ArcFace Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 学习率曲线
    if 'LR' in data:
        axes[1, 1].plot(data['LR']['steps'], data['LR']['values'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线图已保存到: {save_path}")
    
    plt.show()

def main():
    log_dir = "outputs/recognition/logs"
    
    print("正在读取虹膜识别训练日志...")
    data = read_tensorboard_logs(log_dir)
    
    if data:
        analyze_training_logs(data)
        
        # 保存训练曲线图
        plot_save_path = "outputs/recognition/training_curves.png"
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plot_training_curves(data, plot_save_path)
        
        # 保存数据到CSV
        csv_save_path = "outputs/recognition/training_data.csv"
        df_data = {}
        for tag, values in data.items():
            df_data[f'{tag}_step'] = values['steps']
            df_data[f'{tag}_value'] = values['values']
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_save_path, index=False)
        print(f"训练数据已保存到: {csv_save_path}")
    else:
        print("无法读取训练日志")

if __name__ == "__main__":
    main()
