import torch
import torch.nn.functional as F
from pcdet.ops.clocs.clocs_utils import cos_similarity

from torch.autograd import gradcheck

def extract(g):
    global features_grad
    features_grad = g


class CustomCosineSimilarity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lidar_features, camera_features):
        lidar_num = lidar_features.shape[0]
        camera_num = camera_features.shape[0]
        lidar_features = lidar_features.view(lidar_num, -1, lidar_features.shape[-1])
        camera_features = camera_features.view(-1, camera_num, camera_features.shape[-1])
        
        cos_sim = torch.zeros([lidar_num, camera_num], device = lidar_features.device)
        cos_similarity(lidar_features, camera_features, cos_sim)
        ctx.save_for_backward(lidar_features, camera_features, cos_sim)
        return cos_sim

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output -- [B, N]
        lidar_features, camera_features, cos_sim = ctx.saved_tensors
        lidar_features = lidar_features.view(lidar_features.shape[0], -1)
        camera_features = camera_features.view(camera_features.shape[1], -1)
        
        norm_lidar_features = torch.norm(lidar_features, dim=-1)
        norm_camera_features = torch.norm(camera_features, dim=-1)
        
        lidar_num = lidar_features.shape[0]
        camera_num = camera_features.shape[0]
        lidar_first_below = (norm_lidar_features.view(-1, 1) * norm_camera_features.view(1, -1)).unsqueeze(-1)
        camera_ext = camera_features.view(1, camera_num, -1).expand(lidar_num, camera_num, camera_features.shape[-1])
        lidar_second_below = norm_lidar_features.view(lidar_num, 1).expand(lidar_num, camera_num).unsqueeze(-1)
        lidar_ext = lidar_features.view(lidar_num, 1, -1).expand(lidar_num, camera_num, lidar_features.shape[-1])
        lidar_grad = camera_ext/lidar_first_below - cos_sim.view(lidar_num, camera_num, -1) * lidar_ext / lidar_second_below**2
        lidar_grad = torch.sum(lidar_grad, dim=1)
        
        camera_first_below = (norm_camera_features.view(-1, 1) * norm_lidar_features.view(1, -1)).unsqueeze(-1)
        lidar_ext = lidar_features.view(1, lidar_num, -1).expand(camera_num, lidar_num, lidar_features.shape[-1])
        camera_second_below = norm_camera_features.view(camera_num, 1).expand(camera_num, lidar_num).unsqueeze(-1)
        camera_ext = camera_features.view(camera_num, 1, -1).expand(camera_num, lidar_num, camera_features.shape[-1])
        camera_grad = lidar_ext/camera_first_below - cos_sim.T.view(camera_num, lidar_num, -1) * camera_ext / camera_second_below**2
        camera_grad = torch.sum(camera_grad, dim=1)
        print(lidar_grad)
        print(camera_grad)
        print(grad_output)
        
        return  grad_output @ camera_grad, grad_output.T @ lidar_grad



def print_grad(grad):
    print("grad is ", grad)
    print("grad is ", grad.requires_grad)
    
# input1 = torch.tensor([[ 0.5079, -1.6995, -1.2796],
#         [ 0.8876, -0.4778, -0.7058]], device='cuda:0', requires_grad=True)
input1 = torch.randn(3, 4, device='cuda:0', requires_grad=True)
input2 = torch.randn(5, 4, device='cuda:0', requires_grad=True)

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1)[:, None], b.norm(dim=-1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
output = sim_matrix(input1, input2)
loss1 = output.sum()/4
input1.register_hook(extract)
loss1.backward()
print(features_grad)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, lidar_features, camera_features):
        sim_feature = CustomCosineSimilarity.apply(input1, input2)
        return sim_feature

model = Model()
model.train()
input4 = input1.detach().clone()
input4.requires_grad = True
input4.register_hook(lambda grad:store(grad,input4))
sim_feature = model(input4, input2)
loss2 = sim_feature.sum()/4
loss2.backward()

print("Done")


    