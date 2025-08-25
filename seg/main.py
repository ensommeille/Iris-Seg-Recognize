#!/usr/bin/env python3
import argparse, os
from train import train
from inference import run_inference
from test import run_test  # 若没有 test.py，可删除或实现

def parse_args():
    parser = argparse.ArgumentParser('Iris Segmentation (improved)')
    sub = parser.add_subparsers(dest='mode', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data-roots', type=str, nargs='+', required=True,
                   help='One or more dataset roots. Each root should contain train/images & train/masks (or images/masks).')
    t.add_argument('--save-dir', type=str, default='checkpoints')
    t.add_argument('--batch-size', type=int, default=16)
    t.add_argument('--epochs', type=int, default=50)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--img-size', type=int, default=256)
    t.add_argument('--num-workers', type=int, default=4)
    t.add_argument('--resume', type=str, default=None)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--warmup-epochs', type=int, default=3)
    t.add_argument('--freeze-backbone-epochs', type=int, default=0,
                   help='If >0, freeze backbone for these many epochs to stabilize training.')
    t.add_argument('--amp', action='store_true', help='Enable mixed precision training (recommended)')
    t.add_argument('--base-channels', type=int, default=32)
    t.add_argument('--val-split', type=float, default=0.05)
    t.add_argument('--backbone', type=str, default='mobilenet_v3', choices=['mobilenet_v3','mobilenet_v2'])
    t.add_argument('--boundary-weight', type=float, default=0.0, help='Auxiliary boundary loss weight (0 = disabled)')

    i = sub.add_parser('infer')
    i.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth or .onnx)')
    i.add_argument('--input', type=str, required=True)
    i.add_argument('--output', type=str, default='results')
    i.add_argument('--batch-size', type=int, default=8)
    i.add_argument('--img-size', type=int, default=256)
    i.add_argument('--device', type=str, default='cuda')
    i.add_argument('--visualize', action='store_true')
    i.add_argument('--tensorrt', action='store_true')
    i.add_argument('--overlay', action='store_true', help='Whether to generate overlay image. Default is False, only output mask with same name as original image. If True, generate overlay image with _mask and _overlay suffixes.')

    test_parser = sub.add_parser('test')
    test_parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth or .onnx)')
    test_parser.add_argument('--data', type=str, required=True,
                             help='Path to data folder containing images and masks subfolders')
    test_parser.add_argument('--output', type=str, default='test_results')
    test_parser.add_argument('--batch-size', type=int, default=8)
    test_parser.add_argument('--img-size', type=int, default=256)
    test_parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True) if getattr(args,'save_dir',None) else None
    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        run_inference(args)
    elif args.mode == 'test':
        run_test(args)

if __name__ == '__main__':
    main()
