import torch

a = torch.tensor([[1, 2, 3, 4],[5, 6, 7, 8]], dtype=torch.long)    # (2, 4)
b = torch.tensor([2, 3, 4, 5], dtype=torch.long)                   # (4,)

c = torch.einsum("d,nd->n", b, a)
d = torch.einsum("xy,y->x", a, b)
print(a)
print(b)
print(c)
print(c.shape)