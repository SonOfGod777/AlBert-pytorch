import sys
import torch
# sys.path.append('..')
# from config import *
# print(args.batch_size)
# from embedding import *
from RelPosAttention import MultiAttention

# Attention = MultiAttention()
# print(Attention)

i = 0
k = 2
for x in range(10):
    i = x + i + k
    print(x, k, i)
