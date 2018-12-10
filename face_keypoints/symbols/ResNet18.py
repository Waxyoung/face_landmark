import torch
import torch.nn as nn
import math
from momo_pytorch2caffe import *
import torch.optim as optim
import remove_caffe_model_bn as remove_bn
import sys

#define ResNet18 Frame
class ResNet18(nn.Module):
    def __init__(self, name, in_channels, out_channels):
        super(ResNet18, self).__init__()
        self.op_name = name

        op_list = []

        op_list += [ conv_bn_relu(name + '/first_conv', in_channels, 64, kernel_size = 5, stride = 2, padding = 2) ]

        ch_num = [64, 64, 128, 256, 512]
        
        for i in range(len(ch_num) - 1):
            stride = 1 if (i == 0) else 2
            op_list += [ BasicResnetBlock(name + '/stage%d_1'%(i+1), ch_num[i], ch_num[i+1], stride = stride, direct_plus = True),
                         BasicResnetBlock(name + '/stage%d_2'%(i+1), ch_num[i+1], ch_num[i+1], stride = 1, direct_plus = True)  ]

        op_list += [op_name(name + '/AvgPooling', nn.AvgPool2d(8))]

        op_list += [ flatten(name + '/flatten', 1) ]
        
        op_list += [ linear_bn_relu(name + '/FC1', 512, 256),
                     op_name(name + '/FC2', nn.Linear(256, out_channels))
                   ]

        self.conv_block = nn.Sequential(*op_list)

        
    def forward(self, x):
        return self.conv_block(x)

    def generate_caffe_prototxt(self, caffe_net, x):
        feature = generate_caffe_prototxt(self.conv_block, caffe_net, x)
        return generate_caffe_prototxt(self.landmark_block, caffe_net, feature), generate_caffe_prototxt(self.trackprobe_block, caffe_net, feature)

def get_symbol(symbol_name = 'ResNet18', input_nc = 3, output_nc = 214):
    return ResNet18(symbol_name, input_nc, output_nc)

def load_model(model, model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except:
        pretrained_dict = torch.load(model_path, map_location='cpu')
        model_dic = model.state_dict()
        for k,v in pretrained_dict.items():
            if (k in model_dic) and  v.size() == model_dic[k].size():
                model_dic[k] = v
            else:
                key_split = k.split('.')
                key_split[0] = 'landmark_block'
                key_split[1] = str( int(key_split[1]) - 6 ) 
                key_new = '.'.join(key_split)
                if (key_new in model_dic) and (v.size() == model_dic[key_new].size()):
                    model_dic[key_new] = v
                else:
                    print 'Unknow keys:', key_new
                    # sys.exit()
        model.load_state_dict(model_dic)

def pytorch2caffe(proto_path, caffemodel_path):
    model = ResNet18('ResNet18', 3, 214)
    load_model(model, '../../checkpoints/ResNet18_latest.pth')
    model.eval()
    
    caffe_net = caffe.NetSpec()
    layer = L.Input(shape=dict(dim=[1, 3, 128, 128]))
    caffe_net.tops['data'] = layer
    model.generate_caffe_prototxt(caffe_net, layer)
    # print(caffe_net.to_proto())
    with open(proto_path, 'w') as f:
        f.write(str(caffe_net.to_proto()))

    caffe_net = caffe.Net(proto_path, caffe.TEST)
    convert_weight_from_pytorch_to_caffe(model, caffe_net)
    caffe_net.save(caffemodel_path)

    caffe_prototxt_nobn = proto_path.replace('.prototxt', '_nobn.prototxt')
    caffe_caffemodel_nobn = caffemodel_path.replace('.caffemodel', '_nobn.caffemodel')
    remove_bn.zrnRemoveProtoCaffemodelBN(proto_path, caffemodel_path, caffe_prototxt_nobn, caffe_caffemodel_nobn)
    print '----------converted finished--------------'

    print 'Begin to Test the Model Created!'    
    pytorch_input = torch.randn(1, 3, 128, 128)
    pytorch_output, track_probe = model(pytorch_input)

    caffe.set_mode_cpu()
    caffe_net = caffe.Net(caffe_prototxt_nobn, caffe_caffemodel_nobn, caffe.TEST)
    forward_kwargs = {'data': pytorch_input.numpy()}
    caffe_output = caffe_net.forward(**forward_kwargs).items()[0][1]

    criterion = nn.MSELoss()
    loss = criterion(pytorch_output, torch.FloatTensor(caffe_output))

    print 'RMSE LOSS:', math.sqrt(loss.item())
    print 'pytorch output:   ', pytorch_output.detach().numpy()[0,:6]
    print 'caffe_nobn output:', caffe_output[0,:6]
    if (math.sqrt(loss.item()) < 0.001):
        print 'Convert Successful!' 
    else:
        print 'Convert failed!' 

    sys.exit()

if __name__=='__main__':
    # pytorch2caffe('../../caffemodel/ResNet18.prototxt',  '../../caffemodel/ResNet18.caffemodel')

    net = ResNet18('ResNet18', 3, 214)
    net.eval()
    
    # for name, m in net.named_modules():
    #     print name
    #     print m
    # sys.exit()

    # criterion = nn.MSELoss()
    # target = torch.randn(1,10)
    input = torch.randn(1, 3, 128, 128)

    output = net(input)
    print output.size()
    sys.exit()

    optimizer = optim.SGD(net.parameters(), lr=0.001)
    
    for i in range(1000):
        output = net(input)
        loss = criterion(output, target)
        print loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

