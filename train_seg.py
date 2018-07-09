import torch
from models.kcnet import KCNetSegment
from data.shapenet import ShapeNetDataset

trainset = ShapeNetDataset('/opt/shapenetcore_partanno_segmentation_benchmark_v0', split='val')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = ShapeNetDataset('/opt/shapenetcore_partanno_segmentation_benchmark_v0', split='test')
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

net = KCNetSegment(trainset.class_nums, trainset.category_nums)
for epcho in range(1, 100):
    net.fit(trainloader, epcho)
net.score(testloader)
