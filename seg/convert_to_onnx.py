import torch
import torch.onnx
import argparse
import os
from model import build_model


def convert_pth_to_onnx(pth_path, onnx_path, img_size=256, num_classes=1, base_channels=32, use_boundary=False, backbone='mobilenet_v3'):
    # 创建模型
    model = build_model(num_classes=num_classes, pretrained=False, base_channels=base_channels, 
                       use_boundary=use_boundary, backbone=backbone)
    
    # 加载模型权重
    device = torch.device('cpu')  # ONNX转换通常在CPU上进行
    ck = torch.load(pth_path, map_location=device)
    state = ck.get('model', ck)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    # 创建示例输入张量
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # 导出模型到ONNX格式
    torch.onnx.export(
        model,                          # model being run
        dummy_input,                    # model input (or a tuple for multiple inputs)
        onnx_path,                      # where to save the model
        export_params=True,             # store the trained parameter weights inside the model file
        opset_version=11,               # the ONNX version to export the model to
        do_constant_folding=True,       # whether to execute constant folding for optimization
        input_names=['input'],          # the model's input names
        output_names=['output'],        # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},     # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f'Model successfully converted to ONNX format and saved at {onnx_path}')


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX format')
    parser.add_argument('--pth-path', type=str, required=True, help='Path to the PyTorch model (.pth file)')
    parser.add_argument('--onnx-path', type=str, required=True, help='Path where the ONNX model will be saved')
    parser.add_argument('--img-size', type=int, default=256, help='Image size for the model input')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--base-channels', type=int, default=32, help='Base channels for the model')
    parser.add_argument('--use-boundary', action='store_true', help='Whether the model uses boundary head')
    parser.add_argument('--backbone', type=str, default='mobilenet_v3', choices=['mobilenet_v3', 'mobilenet_v2'], 
                        help='Backbone architecture')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)
    
    # 转换模型
    convert_pth_to_onnx(
        args.pth_path,
        args.onnx_path,
        img_size=args.img_size,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        use_boundary=args.use_boundary,
        backbone=args.backbone
    )


if __name__ == '__main__':
    main()