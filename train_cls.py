import torch
from models.kcnet import KCNetClassify
from data.modelnet import ModelNetDataset

trainset = ModelNetDataset('/opt/modelnet40_normal_resampled', train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = ModelNetDataset('/opt/modelnet40_normal_resampled', train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

net = KCNetClassify(trainset.class_nums)
for epcho in range(1, 400):
    net.fit(trainloader, epcho)
net.score(testloader)
