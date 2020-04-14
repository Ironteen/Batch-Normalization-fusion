import os
import copy
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from pytorchcv.model_provider import get_model as ptcv_get_model

from bn_fusion import fuse_bn_recursively

def main():
    # prepare input data
    trf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    inputs = trf(Image.open('./data/dog.jpg')).unsqueeze(0)

    # load existing model from pytorchcv lib (pretrain=False)
    net = ptcv_get_model('mobilenetv2_w1', pretrained=False)
    net.eval()
    output  = net(inputs)
    # print("original model:",net)

    # fuse the Batch Normalization
    net1 = fuse_bn_recursively(copy.deepcopy(net))
    net1.eval()
    output1 = net1(inputs)
    # print("BN fusion  model:",net1)

    # compare the output
    print("output of original model:",output.size())
    print("=> min : %.4f, mean : %.4f max : %.4f"%(output.min().detach().numpy(),output.mean().detach().numpy(),output.max().detach().numpy()))
    print("output of BNfusion model:",output1.size())
    print("=> min : %.4f, mean : %.4f max : %.4f"%(output1.min().detach().numpy(),output1.mean().detach().numpy(),output1.max().detach().numpy()))

    # transform to ONNX format for  visualization
    dummy_input = Variable(torch.randn([1, 3, 224, 224]))
    torch.onnx.export(net, dummy_input, "./data/mobilenet_v2.onnx")
    torch.onnx.export(net1, dummy_input, "./data/mobilenet_v2_nobn.onnx")

if __name__ == '__main__':
    main()
