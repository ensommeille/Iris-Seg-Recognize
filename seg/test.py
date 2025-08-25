import torch
import cv2
import numpy as np
import os
from model import build_model
from dataset import IrisDataset, collect_pairs
from torch.utils.data import DataLoader
from utils import iou_score, dice_score

# Try to import onnxruntime, but handle the case where it's not available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False


def run_test(args):
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    
    # Check if using ONNX model
    use_onnx = args.weights.endswith('.onnx')
    
    if use_onnx:
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX model specified but onnxruntime is not installed. Please install onnxruntime to use ONNX models.")
        # Use ONNX Runtime
        session = ort.InferenceSession(args.weights)
        input_name = session.get_inputs()[0].name
    else:
        # Build PyTorch model
        model = build_model(num_classes=1, pretrained=False, base_channels=32)
        ck = torch.load(args.weights, map_location=device)
        state = ck.get('model', ck)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
    
    # Collect image-mask pairs from data folder
    images, masks = collect_pairs(args.data)
    if len(images) == 0:
        raise FileNotFoundError(f'No image-mask pairs found in {args.data}')
    
    # Create dataset and dataloader
    dataset = IrisDataset(images, masks, img_size=args.img_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize metrics
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, (imgs, gt_masks) in enumerate(loader):
            imgs = imgs.to(device)
            gt_masks = gt_masks.to(device)
            
            # Run inference
            if use_onnx:
                # ONNX inference
                imgs_np = imgs.cpu().numpy()
                preds = session.run(None, {input_name: imgs_np})[0]
                preds = torch.from_numpy(preds).to(device)
                probs = torch.sigmoid(preds)
                pred_masks = (probs > 0.5).float()
            else:
                # PyTorch inference
                preds = model(imgs)
                probs = torch.sigmoid(preds)
                pred_masks = (probs > 0.5).float()
            
            # Calculate metrics
            batch_iou = iou_score(pred_masks, gt_masks)
            batch_dice = dice_score(pred_masks, gt_masks)
            total_iou += batch_iou.item() * imgs.size(0)
            total_dice += batch_dice.item() * imgs.size(0)
            num_samples += imgs.size(0)
            
            # Process each sample in the batch
            for i in range(imgs.size(0)):
                # Get original image path
                sample_idx = batch_idx * args.batch_size + i
                original_img_path = images[sample_idx]
                base_name = os.path.splitext(os.path.basename(original_img_path))[0]
                
                # Get ground truth mask and prediction mask
                gt_mask = gt_masks[i].cpu().numpy().squeeze()
                pred_mask = pred_masks[i].cpu().numpy().squeeze()
                
                # Convert to uint8 format (0-255)
                gt_mask_vis = (gt_mask * 255).astype('uint8')
                pred_mask_vis = (pred_mask * 255).astype('uint8')
                
                # Read original image
                original_img = cv2.imread(original_img_path)
                if original_img is None:
                    original_img = cv2.imdecode(np.fromfile(original_img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if len(original_img.shape) == 2 or original_img.shape[2] == 1:
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
                
                # Resize masks to original image size
                resized_gt_mask = cv2.resize(gt_mask_vis, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                resized_pred_mask = cv2.resize(pred_mask_vis, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Create overlay for ground truth
                gt_overlay = original_img.copy()
                gt_overlay[resized_gt_mask > 0] = [0, 255, 0]  # Green for ground truth
                gt_combined = cv2.addWeighted(original_img, 0.6, gt_overlay, 0.4, 0)
                
                # Add "GT mask" text to ground truth image
                cv2.putText(gt_combined, 'GT mask', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Create overlay for prediction
                pred_overlay = original_img.copy()
                pred_overlay[resized_pred_mask > 0] = [255, 0, 0]  # Red for prediction
                pred_combined = cv2.addWeighted(original_img, 0.6, pred_overlay, 0.4, 0)
                
                # Add "Predicted mask" text to prediction image
                cv2.putText(pred_combined, 'Predicted mask', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                
                # Combine both overlays side by side
                combined_height = max(gt_combined.shape[0], pred_combined.shape[0]) + 40  # Extra space for text
                combined_width = gt_combined.shape[1] + pred_combined.shape[1]
                combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
                combined_img[20:20+gt_combined.shape[0], :gt_combined.shape[1]] = gt_combined
                combined_img[20:20+pred_combined.shape[0], gt_combined.shape[1]:gt_combined.shape[1]+pred_combined.shape[1]] = pred_combined
                
                # Calculate current sample metrics
                sample_iou = iou_score(pred_masks[i].unsqueeze(0), gt_masks[i].unsqueeze(0)).item()
                sample_dice = dice_score(pred_masks[i].unsqueeze(0), gt_masks[i].unsqueeze(0)).item()
                
                # Add current sample metrics text
                text = f'Current IoU: {sample_iou:.4f} | Current Dice: {sample_dice:.4f}'
                cv2.putText(combined_img, text, (10, combined_img.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Save combined image
                output_path = os.path.join(args.output, f'{base_name}_comparison.png')
                cv2.imwrite(output_path, combined_img)
    
    # Print final metrics
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    print(f'Test Results - Avg IoU: {avg_iou:.4f}, Avg Dice: {avg_dice:.4f}')