import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DNetV3']  # model_registry will add each entrypoint fn to this


def conv33(in_channel, out_channel, stride=1, groups=1, bias=False):
    if groups != 0 and in_channel % groups != 0:
        raise ValueError('In channel "{}" is not a multiple of groups: "{}"'.format(
            in_channel, groups))
    if out_channel % groups != 0:
        raise ValueError('Out channel "{}" is not a multiple of groups: "{}"'.format(
            out_channel, groups))

    return nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1,
                     stride=stride, groups=groups, bias=bias)


def conv11(in_channel, out_channel, stride=1, bias=False):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=bias)


def conv33_base(in_channel, out_channel, stride=1, base_channel=1):
    return conv33(in_channel, out_channel, stride, in_channel // base_channel)


def conv33_sep(in_channel, out_channel, stride):
    return nn.Sequential(
        conv33(in_channel, in_channel, stride, groups=in_channel),
        conv11(in_channel, out_channel))


OPS = {
    'conv3': lambda in_channel, out_channel, stride: conv33(in_channel, out_channel, stride),
    'conv1': lambda in_channel, out_channel, stride: conv11(in_channel, out_channel, stride),
    'conv3_grp2': lambda in_channel, out_channel, stride: conv33(in_channel, out_channel, stride, groups=2),
    'conv3_grp4': lambda in_channel, out_channel, stride: conv33(in_channel, out_channel, stride, groups=4),
    'conv3_base1': lambda in_channel, out_channel, stride: conv33_base(in_channel, out_channel, stride, base_channel=1),
    'conv3_base16': lambda in_channel, out_channel, stride: conv33_base(in_channel, out_channel, stride,
                                                                        base_channel=16),
    'conv3_base32': lambda in_channel, out_channel, stride: conv33_base(in_channel, out_channel, stride,
                                                                        base_channel=32),
    'conv3_sep': lambda in_channel, out_channel, stride: conv33_sep(in_channel, out_channel, stride)
}


def create_op(opt_name, in_channel, out_channel, stride=1):
    layer = OPS[opt_name](in_channel, out_channel, stride)
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    bn = nn.BatchNorm2d(out_channel)
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()
    return nn.Sequential(layer, bn)


class AddBlock(nn.Module):
    def __init__(self, layer_sizes, strides, num1, num2):
        super(AddBlock, self).__init__()
        self.num1 = num1
        self.num2 = num2
        self.conv = None
        stride = 1
        if strides[num1] != strides[num2]:
            stride = 2
        if stride != 1 or layer_sizes[num1] != layer_sizes[num2]:
            self.conv = create_op('conv1', layer_sizes[num1], layer_sizes[num2], stride)

    def forward(self, x):
        x1, x2 = x[self.num1], x[self.num2]
        if self.conv is not None:
            x1 = self.conv(x1)
        x[self.num2] = x1 + x2
        return x


class ConcatBlock(nn.Module):
    def __init__(self, layer_sizes, strides, num1, num2):
        super(ConcatBlock, self).__init__()
        self.num1 = num1
        self.num2 = num2
        self.conv = None
        stride = 1
        if strides[num1] != strides[num2]:
            stride = 2
        if stride != 1:
            self.conv = create_op('conv1', layer_sizes[num1], layer_sizes[num1], stride)
        layer_sizes[self.num2] += layer_sizes[self.num1]

    def forward(self, x):
        x1, x2 = x[self.num1], x[self.num2]
        if self.conv is not None:
            x1 = self.conv(x1)
        x[self.num2] = torch.cat([x1, x2], 1)
        return x


class EncodedBlock(nn.Module):
    def __init__(self, block_str, in_channel, op_names, stride=1, channel_increase=1):
        super(EncodedBlock, self).__init__()

        if "-" in block_str:
            layer_str, connect_str = block_str.split('-')
        else:
            layer_str, connect_str = block_str, ""

        layer_str = layer_str + "2"
        base_channel = in_channel * channel_increase
        layer_sizes = [in_channel]
        connect_parts = [connect_str[i:i + 3] for i in range(0, len(connect_str), 3)]
        connect_parts = sorted(connect_parts, key=lambda x: x[2])
        connect_index = 0

        self.module_list = nn.ModuleList()
        length = len(layer_str) // 2
        stride_place = 0
        while (stride_place + 1) * 2 < len(layer_str) and layer_str[stride_place * 2] == '1':
            stride_place += 1
        
        strides = [1] * (stride_place + 1) + [stride] * (length - stride_place)
        connect_parts.append("a0{}".format(length))

        for i in range(length):
            layer_module_list = nn.ModuleList()
            layer_opt_name = op_names[int(layer_str[i * 2])]
            layer_in_channel = layer_sizes[-1]
            layer_out_channel = base_channel * 2 ** int(layer_str[i * 2 + 1]) // 4
            layer_stride = stride if i == stride_place else 1
            layer = create_op(layer_opt_name, layer_in_channel, layer_out_channel, layer_stride)
            if i + 1 == len(layer_str) // 2:
                layer[-1].weight.data.fill_(0)
            layer_module_list.append(layer)
            layer_sizes.append(layer_out_channel)

            while connect_index < len(connect_parts) and int(connect_parts[connect_index][2]) == i + 1:
                block_class = AddBlock if connect_parts[connect_index][0] == 'a' else ConcatBlock
                block = block_class(
                    layer_sizes, strides, int(connect_parts[connect_index][1]), int(connect_parts[connect_index][2]))
                layer_module_list.append(block)
                connect_index += 1

            self.module_list.append(layer_module_list)

    def forward(self, x):
        outs = [x]
        current = x

        for layer_num, module_layer in enumerate(self.module_list):
            for i, layer in enumerate(module_layer):
                if i == 0:
                    outs.append(layer(current))
                else:
                    outs = layer(outs)
            current = F.relu(outs[-1], inplace=True)

        return current


class DNetV3(nn.Module):
    def __init__(self, arch, op_names=None, num_classes=1000, **kwargs):
        super(DNetV3, self).__init__()
        if op_names is None:
            op_names = ["conv3", "conv1", "conv3_grp2", "conv3_grp4", "conv3_base1", "conv3_base32", "conv3_sep"]

        block_str, num_channel, macro_str = arch.split('_')
        block_encoding_list = block_str.split('*')
        self.macro_str = macro_str
        curr_channel, index = int(num_channel), 0
        layers = [
            create_op('conv3', 3, curr_channel // 2, stride=2),
            nn.ReLU(inplace=True),
            create_op('conv3', curr_channel // 2, curr_channel // 2),
            nn.ReLU(inplace=True),
            create_op('conv3', curr_channel // 2, curr_channel, stride=2),
            nn.ReLU(inplace=True)
        ]

        block_encoding_index = 0
        while index < len(macro_str):
            stride = 1
            if macro_str[index] == '-':
                stride = 2
                index += 1
                block_encoding_index += 1

            channel_increase = int(macro_str[index])
            block_encoding = block_encoding_list[block_encoding_index]
            block = EncodedBlock(block_encoding, curr_channel, op_names, stride, channel_increase)
            layers.append(block)
            curr_channel *= channel_increase
            index += 1

        layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features=curr_channel, out_features=num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x
