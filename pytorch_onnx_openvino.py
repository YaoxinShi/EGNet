'''
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# https://zhuanlan.zhihu.com/p/129879495

#########################################################
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

#########################################################
print(">>> Create the pytorch model using the above model definition")
torch_model = SuperResolutionNet(upscale_factor=3)

#########################################################
print(">>> Initialize the pytorch model with the pretrained weights")
# model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
# torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=torch.device('cpu')))
torch_model.load_state_dict(torch.load("superres_epoch100-44c6958e.pth", map_location=torch.device('cpu')))

# set the model to inference mode
torch_model.eval()

# Input to the model
batch_size = 1    # just a random number
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

###########################################################
print(">>> Export the pytorch model to onnx")
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
'''

import argparse
from dataset import get_loader
from solver import Solver
import torch
import torch.nn as nn

# Convert PyTorch to ONNX
# https://github.com/ngeorgis/pytorch_onnx_openvino/blob/master/pytorch_onnx_openvino.ipynb
# https://docs.aws.amazon.com/zh_cn/dlami/latest/devguide/tutorial-onnx-pytorch-mxnet.html

#####################################################
# (1) create pytorch model
#####################################################

# 保存pth文件有两种方式：(https://zhuanlan.zhihu.com/p/38056115)
# 1. 保存整个网络: torch.save(net, PATH) 
# 2. 保存网络中的参数, 速度快，占空间少: torch.save(net.state_dict(),PATH)
# 由于源程序中使用第二种方式保存pth文件，所以epoch_resnet.pth不能直接使用torch.load打开
# net50 = torch.load('epoch_resnet.pth', map_location=torch.device('cpu'))
# 否则到onnx.export的时候会报错：AttributeError: 'collections.OrderedDict' object has no attribute 'state_dict'

print(">>> create pytorch model")

vgg_path = '/home/liuj/code/Messal/weights/vgg16_20M.pth'
resnet_path = '/home/liuj/code/Messal/weights/resnet50_caffe.pth'
parser = argparse.ArgumentParser()
parser.add_argument('--n_color', type=int, default=3)
parser.add_argument('--cuda', type=bool, default=False)
# Training settings
parser.add_argument('--vgg', type=str, default=vgg_path)
parser.add_argument('--resnet', type=str, default=resnet_path)
parser.add_argument('--epoch', type=int, default=30) # 12, now x3
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--num_thread', type=int, default=4)
parser.add_argument('--load_bone', type=str, default='')
parser.add_argument('--save_fold', type=str, default='./EGNet')
parser.add_argument('--epoch_save', type=int, default=1) # 2, now x3
parser.add_argument('--epoch_show', type=int, default=1)
parser.add_argument('--pre_trained', type=str, default=None)
# Testing settings
parser.add_argument('--model', type=str, default='./epoch_resnet.pth')
parser.add_argument('--test_fold', type=str, default='./results/test')
parser.add_argument('--test_mode', type=int, default=1)
parser.add_argument('--sal_mode', type=str, default='g')
# Misc
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
parser.add_argument('--visdom', type=bool, default=False)

config = parser.parse_args()
test_loader, dataset = get_loader(config.test_batch_size, mode='test',num_thread=config.num_thread, test_mode=config.test_mode, sal_mode=config.sal_mode)
test = Solver(None, test_loader, config, dataset.save_folder())
#print(test.net_bone)
#test.test(test_mode=config.test_mode)

#####################################################
# (2) load pretrained weights
#####################################################

#------------------------------------------
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
#------------------------------------------

print(">>> load pretrained weights")
use_sample_model = False
if use_sample_model:
    torch_model = SuperResolutionNet(upscale_factor=3)
    # model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    # torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=torch.device('cpu')))
    torch_model.load_state_dict(torch.load("superres_epoch100-44c6958e.pth", map_location=torch.device('cpu')))
else:
    torch_model = test.net_bone
    torch_model.load_state_dict(torch.load("epoch_resnet.pth", map_location=torch.device('cpu')))

torch_model.eval() # set the model to inference mode

#####################################################
# (3) convert pytorch to onnx
#####################################################

print(">>> convert pytorch to onnx")
if use_sample_model:
    x = torch.randn(1, 1, 224, 224, requires_grad=True) # Input to the model. (batch,channel,width,height)
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "out.onnx",                # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      verbose=False)             # print out a human-readable representation of the network
else:
    x = torch.randn(1, 3, 224, 224, requires_grad=True) # Input to the model. (batch,channel,width,height)
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "out.onnx",                # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      verbose=False)             # print out a human-readable representation of the network

# For "epoch_resnet.pth"
# If use opset_version=11, pytorch-onnx can pass, but later onnx-openvino will fail.
#   "[ ERROR ]  There is no registered "infer" function for node "Resize_350" with op = "Resize"."
#   Because mo.py only support Resize with Opset-10 version
#   https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html
#   https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit/topic/848533
# If use opset_version=10, pytorch-onnx will fail.
#   "RuntimeError: ONNX export failed: Couldn't export operator aten::upsample_bilinear2d"
#   https://github.com/pytorch/pytorch/issues/29980
#   If replace "bilinear" to "nearest" in model.py, pytorch-onnx (with opset_version=10) can pass. But later onnx-openvino still fail with "Resize" issue.

#####################################################
# (4) verify onnx model
#####################################################

# pip install onnxruntime
# pip install onnxruntime-gpu

print(">>> verify onnx model")
import onnx
onnx_model = onnx.load("out.onnx")
onnx.checker.check_model(onnx_model)

#####################################################
# (5.a) convert onnx to OpenVINO
#####################################################

onnx_openvino_directly = True
if onnx_openvino_directly:
    print(">>> convert onnx to OpenVINO")
    import os
    os.system('python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py" --input_model out.onnx')

#####################################################
# (5.b.1) convert onnx to TF
#####################################################

else:
    # pip install onnx-tf
    print(">>> convert onnx to TF")
    from onnx_tf.backend import prepare
    tf_rep = prepare(onnx_model, strict=False)
    tf_rep.export_graph("out.pb")

# If use opset_version=11, onnx-tf fails.
#   "NotImplementedError: Gather version 11 is not implemented."
#   This issue is fixed recently: https://github.com/onnx/onnx-tensorflow/issues/527
#   If you are on Tensorflow 1.x, you need to do a source build from the "tf-1.x" branch. Please try the following and see if it helps.
#   //python
#   //import tensorflow as tf
#   //tf.__version__
#   git clone https://github.com/onnx/onnx-tensorflow.git
#   cd onnx-tensorflow
#   git checkout tf-1.x
#   pip install -e .
# After fixing above, new failure:
#   "NotImplementedError: Resize version 11 is not implemented."
#   This issue hasn't been fixed yet: https://github.com/onnx/onnx-tensorflow/issues/591

#####################################################
# (5.b.2) convert onnx to OpenVINO
#####################################################

    print(">>> convert TF to OpenVINO")
    import os
    os.system('python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py" --input_model out.pb')
