# utils.py - metrics, losses, checkpoint helpers
import torch, os, cv2, numpy as np, time
import torch.nn.functional as F

def iou_score(preds, targets, eps=1e-7):
    preds = preds.float(); targets = targets.float()
    inter = (preds * targets).sum(dim=[1,2,3])
    union = (preds + targets - preds*targets).sum(dim=[1,2,3])
    iou = (inter + eps) / (union + eps)
    return iou.mean()

def dice_score(preds, targets, eps=1e-7):
    preds = preds.float(); targets = targets.float()
    inter = (preds * targets).sum(dim=[1,2,3])
    denom = preds.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3])
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()

def dice_loss(preds, targets, eps=1e-7):
    preds = preds.float(); targets = targets.float()
    inter = (preds * targets).sum(dim=[1,2,3])
    denom = preds.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3])
    loss = 1.0 - ((2 * inter + eps) / (denom + eps))
    return loss.mean()

def boundary_target(mask, connectivity=1):
    """
    Compute a binary boundary/edge target from a binary mask using Sobel-like filters.
    Args:
        mask: torch.Tensor with shape (B,1,H,W) or (B,H,W), values in {0,1} or {0.,1.}
    Returns:
        edge: torch.Tensor same shape (B,1,H,W) with values 0/1 (float)
    Implementation using simple gradient magnitude with conv kernels (works on GPU).
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    # define sobel kernels
    kernel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], dtype=torch.float32, device=mask.device).view(1,1,3,3)
    kernel_y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], dtype=torch.float32, device=mask.device).view(1,1,3,3)
    # pad reflect
    grad_x = F.conv2d(mask, kernel_x, padding=1)
    grad_y = F.conv2d(mask, kernel_y, padding=1)
    grad = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
    # normalize per-sample
    maxv = grad.view(grad.size(0), -1).max(dim=1)[0].view(-1,1,1,1)
    maxv = torch.clamp(maxv, min=1e-6)
    grad = grad / maxv
    # threshold to binary edge
    edge = (grad > 0.05).float()
    return edge

def save_checkpoint(model, optimizer, epoch, best_iou, path):
    ck = {'model': model.state_dict(), 'optimizer': getattr(optimizer,'state_dict', lambda: None)(), 'epoch': epoch, 'best_iou': best_iou}
    torch.save(ck, path)

def save_mask(mask_np, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    ext = os.path.splitext(path)[1][1:] or 'png'
    _, buf = cv2.imencode('.' + ext, mask_np)
    buf.tofile(path)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    params_mb = total * 4 / (1024**2)
    return f'{total} params ({params_mb:.2f} MB)'

def measure_inference_fps(model, loader, device, warmup=5, runs=100):
    model.to(device); model.eval()
    
    # Warmup
    it = iter(loader)
    with torch.no_grad():
        for _ in range(warmup):
            try:
                imgs, _ = next(it)
            except StopIteration:
                break
            imgs = imgs.to(device)
            _ = model(imgs)
    
    # Measure FPS
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            _ = model(imgs)
            n += imgs.size(0)
            if n >= runs:
                break
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.time()
    elapsed = t1 - t0
    
    return n / (elapsed + 1e-12) if elapsed > 0 else 0
