import torch
x = torch.tensor([[2,2],[1,.7],[.5,0]]).float()
norms = torch.linalg.norm(x,dim=-1).unsqueeze(-1)
x_norm = x/norms
clamped_norm = torch.clamp(norms,min=0,max=1)
print(x_norm*clamped_norm)


