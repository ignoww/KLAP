import numpy as np
import torch
class Masker_color():
    """Object for masking and demasking"""

    def __init__(self, width=4, mode='zero', infer_single_pass=False, include_mask_as_input=False):
        self.grid_size = width
        self.n_masks = width ** 2

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, X, i, mask_partial):

        phasex = i % self.grid_size
        phasey = (i // self.grid_size) % self.grid_size
#        phasey = (i // 2) % 2 +1
        mask = pixel_grid_mask(X[0, 0].shape, self.grid_size, phasex, phasey)
        mask = mask.to(X.device)

        mask_inv = torch.ones(mask.shape).to(X.device) - mask

        masked = interpolate_mask_color(X, mask, mask_inv,mask_partial)

           
        return masked, mask

    def __len__(self):
        return self.n_masks


def pixel_grid_mask(shape, patch_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if (i % patch_size == phase_x and j % patch_size == phase_y):
                A[i, j] = 1
    return torch.Tensor(A)



def interpolate_mask(tensor, mask, mask_inv):
    device = tensor.device

    mask = mask.to(device)

    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)

    return filtered_tensor * mask + tensor * mask_inv



def interpolate_mask_color(tensor, mask, mask_inv,mask_partial):
    device = tensor.device

    mask = mask.to(device)

    kernel = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], (1.0, 1.0, 1.0)],dtype=np.float32)
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor_1 = torch.nn.functional.conv2d(tensor[:,0:1,:,:], kernel, stride=1, padding=1)
    filtered_tensor_2 = torch.nn.functional.conv2d(tensor[:,1:2,:,:], kernel, stride=1, padding=1 )
    filtered_tensor_3 = torch.nn.functional.conv2d(tensor[:,2:3,:,:], kernel, stride=1, padding=1)
    filtered_tensor = torch.cat([filtered_tensor_1,filtered_tensor_2,filtered_tensor_3],1)
    
    ####
    mask_partial = mask_partial*255
    kernel = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], (1.0, 1.0, 1.0)],dtype=np.float32)
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    mask_partial_1 = torch.nn.functional.conv2d(mask_partial[:,0:1,:,:], kernel, stride=1, padding=1)
    mask_partial_2 = torch.nn.functional.conv2d(mask_partial[:,1:2,:,:], kernel, stride=1, padding=1)
    mask_partial_3 = torch.nn.functional.conv2d(mask_partial[:,2:3,:,:], kernel, stride=1, padding=1)
    mask_partial_ = torch.cat([mask_partial_1,mask_partial_2,mask_partial_2],1)
#    print(mask_partial_.shape)
#    print(mask_partial.shape)
    
    mask_partial_ = mask_partial_ * mask_partial
    filtered_tensor = filtered_tensor * mask_partial

    filtered_tensor = torch.nan_to_num(filtered_tensor * (8 /mask_partial_), nan=0.0, posinf=0.0)
    
#    torch.save(torch.sum(filtered_tensor * mask + tensor * mask_inv,1).unsqueeze(1), "ori.pt")
#    torch.save(tensor, "tensor.pt")
#    torch.save(filtered_tensor, "filtered_tensor.pt")
#    torch.save(filtered_tensor * mask, "mask.pt")
#    torch.save(tensor * mask_inv, "mask_inv.pt")
        
    return torch.sum(filtered_tensor * mask + tensor * mask_inv,1).unsqueeze(1)