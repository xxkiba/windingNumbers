import open_clip
import torch
from torchvision import datasets
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
from tqdm import tqdm

# 加载 MNIST 数据集
trainset = datasets.MNIST('data', download=True, train=True)

# 加载 OpenCLIP 模型和预处理工具
model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')

# 初始化列表以存储特征和标签
features = []
labels = []

# 处理数据集的一个小子集，例如前1000个图像
for i, (image, label) in enumerate(tqdm(trainset)):
    if i >= 100:  # 限制处理的图像数量
        break
    preprocessed_image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_feature = model.encode_image(preprocessed_image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

    features.append(image_feature.squeeze().numpy())  # 移除批次维度并转换为 numpy
    labels.append(label)

# 转换为 numpy 数组
features = np.array(features)

# 使用 k-NN 算法找到最近邻
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(features)
distances, indices = nbrs.kneighbors(features)

# 创建图
G = nx.Graph()

# 添加节点和边
for i in range(len(features)):
    G.add_node(i, label=labels[i])

for i, neighbors in enumerate(indices):
    for neighbor in neighbors:
        if i < neighbor:  # 避免重复添加
            G.add_edge(i, neighbor)

# 准备绘图
pos = nx.random_layout(G)  # 随机布局
color_map = [G.nodes[i]['label'] for i in G.nodes]  # 根据标签设置颜色

# 绘制图
plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_color=color_map, node_size=20, cmap=cm.get_cmap('tab10'), with_labels=False)
plt.title('MNIST k-NN Graph Visualization')
plt.show()
