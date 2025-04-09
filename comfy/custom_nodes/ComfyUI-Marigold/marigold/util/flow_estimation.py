import torch
from ...gmflow.gmflow import GMFlow
import numpy as np
import os
from ...gmflow.utils import InputPadder
import torch.nn.functional as F
import cv2

class FlowEstimator:
    def __init__(self, model_path, device):
        self.model = self.load_model(model_path, device)
        self.device = device
    
    def load_model(self, model_path, device):
        loc = 'cuda:{}'.format(0)
        checkpoint = torch.load(model_path, map_location=device)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model = GMFlow().to(device)
        model_without_ddp = model
        model_without_ddp.load_state_dict(weights)
        model_without_ddp.eval()

        return model_without_ddp
            
    def estimate_flow(self, img0, img1):
        # Obtain original image size
        og_size = (img0.shape[1], img0.shape[2]) 

        # Run the model to get flow predictions
        results_dict = self.model(img0, img1, [2], [-1], [-1])
        flow_preds = results_dict['flow_preds']

        # Resize the flow prediction to the original image size
        flow_pred = F.interpolate(flow_preds[0], size=og_size, mode='bilinear', align_corners=True)

        return flow_pred
    
    def warp_with_flow(flow, curImg):
        
        curImg = curImg.unsqueeze(0).unsqueeze(0)
        device = curImg.device
        dtype = curImg.dtype
        N, C, H, W = flow.shape
        flow = -flow

        # Convert to numpy, add the grid, then convert back to torch tensor
        flow_np = flow.cpu().numpy()
        flow_np[:, 0, :, :] += np.arange(W)  # Add x-coordinates to the flow's x component
        flow_np[:, 1, :, :] += np.arange(H)[:, np.newaxis]  # Add y-coordinates to the flow's y component
        flow = torch.from_numpy(flow_np).to(flow.device)

        # Now permute and normalize the flow to get a grid in the range [-1, 1]
        flow = flow.permute(0, 2, 3, 1).to(dtype).to(device)
        flow[:, :, :, 0] = (flow[:, :, :, 0] / (W - 1) * 2) - 1
        flow[:, :, :, 1] = (flow[:, :, :, 1] / (H - 1) * 2) - 1

        # Warp the image by the flow
        nextImg = F.grid_sample(curImg, flow, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Remove batch and channel dimensions before returning
        nextImg = nextImg.squeeze(0).squeeze(0)

        return nextImg
