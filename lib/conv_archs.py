from lib.conv_layers import ConvLayer, EiConvLayer, MaxPool, Flatten, Dropout
from lib.dense_layers import DenseLayer, EiDenseWithShunt
from lib.init_policies import EiDenseWithShunt_WeightInitPolicy_ICLR
from lib.update_policies import DalesANN_cSGD_UpdatePolicy
from torch.nn import functional as F

def test_mnist_layers():
    layers = [
        EiConvLayer(1,32,10,3,3),
        MaxPool(2,2,0),
        EiConvLayer(32,64,10,3,3),
        MaxPool(2,2,0),
        EiConvLayer(64,64,10,3,3),
        Flatten(),
        #Dropout(drop_prob=dp),
        #DenseLayer(1024, 500, nonlinearity=F.relu), # for cifar10
        # 576 for mnist, 1024 for cifar 10
        EiDenseWithShunt(576, 500, 50, nonlinearity=F.relu,
                         weight_init_policy=EiDenseWithShunt_WeightInitPolicy_ICLR(),
                         update_policy=DalesANN_cSGD_UpdatePolicy()), 
        #Dropout(drop_prob=dp),
        EiDenseWithShunt(500, 10,1, nonlinearity=None,
                         weight_init_policy=EiDenseWithShunt_WeightInitPolicy_ICLR(),
                         update_policy=DalesANN_cSGD_UpdatePolicy()),
    ]

    return layers

def vgg_layers(vgg19=False, dropout_prob=0):
    """
    Args:
        vgg19 : bool, return either vgg16 or vgg19.
    """
    kwargs={'stride': 1, 'padding': 1, 'dilation': 1,
            'groups': 1, 'padding_mode': 'zeros', 'bias':True}
    layers = [
            ConvLayer(3,64,3,conv2d_kwargs=kwargs),
            ConvLayer(64,64,3,conv2d_kwargs=kwargs),
            MaxPool(2,2,0),
            ConvLayer(64,128,3,conv2d_kwargs=kwargs),
            ConvLayer(128,128,3,conv2d_kwargs=kwargs),
            MaxPool(2,2,0),
            ConvLayer(128,256,3,conv2d_kwargs=kwargs),
            ConvLayer(256,256,3,conv2d_kwargs=kwargs),
            ConvLayer(256,256,3,conv2d_kwargs=kwargs),
            ConvLayer(256,256,3,conv2d_kwargs=kwargs),
            MaxPool(2,2,0),
            ConvLayer(256,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            MaxPool(2,2,0),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            MaxPool(2,2,0),
            #AveragePool((4,4)), TODO: Implement AveragePool
            Flatten(),
            DenseLayer(512*4*4, 4096, nonlinearity=F.relu),
            Dropout(drop_prob=dropout_prob),
            DenseLayer(4096, 4096, nonlinearity=F.relu),
            Dropout(drop_prob=dropout_prob),
            DenseLayer(4096, 10, nonlinearity=None),
        ]

    if vgg19:
        return layers

    # Below we convert vgg19 to vgg16 (this is probs a clunky way to do it)

    # we need to remove the last conv layer in the last three filterbanks
    # (which we locate using the max pool layers)
    to_remove = []
    for index,l in enumerate(layers):
        if isinstance(l,MaxPool): to_remove.append(index-1)
    to_remove = to_remove[-3:] # slice to only removing the last three convs
    for conv_i in to_remove[::-1]: # index from reverse to not change indexes
        del layers[conv_i]
    return layers

#export
def ei_vgg_layers(vgg19=False,dropout_prob=0):
    """
    Args:
        vgg19 : bool, return either vgg16 or vgg19.
    """
    conv2d_kwargs={'stride': 1, 'padding': 1, 'dilation': 1,
                   'groups': 1, 'padding_mode': 'zeros', 'bias':False}

    layers = [
            EiConvLayer(3,64,10,3,3, e_param_dict=conv2d_kwargs),
            EiConvLayer(64,64,10,3,3,e_param_dict=conv2d_kwargs),
            MaxPool(2,2,0),
            EiConvLayer(64,128,20,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(128,128,20,3,3,e_param_dict=conv2d_kwargs),
            MaxPool(2,2,0),
            EiConvLayer(128,256,40,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(256,256,40,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(256,256,40,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(256,256,40,3,3,e_param_dict=conv2d_kwargs),
            MaxPool(2,2,0),
            EiConvLayer(256,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            MaxPool(2,2,0),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            MaxPool(2,2,0),
            #AveragePool((4,4)), TODO: Implement AveragePool
            Flatten(),
            EiDenseWithShunt(512*4*4, (4096,409), nonlinearity=F.relu,
                             weight_init_policy=EiDenseWithShunt_WeightInitPolicy_ICLR(),
                             update_policy=DalesANN_cSGD_UpdatePolicy()),
            Dropout(drop_prob=dropout_prob),
            EiDenseWithShunt(4096, (4096,409), nonlinearity=F.relu,
                             weight_init_policy=EiDenseWithShunt_WeightInitPolicy_ICLR(),
                             update_policy=DalesANN_cSGD_UpdatePolicy()),
            Dropout(drop_prob=dropout_prob),
            EiDenseWithShunt(4096, (10,1), nonlinearity=None,
                             weight_init_policy=EiDenseWithShunt_WeightInitPolicy_ICLR(),
                             update_policy=DalesANN_cSGD_UpdatePolicy()),
        ]

    if vgg19:
        return layers

    # Below we convert vgg19 to vgg16 (this is probs a clunky way to do it)

    # we need to remove the last conv layer in the last three filterbanks
    # (which we locate using the max pool layers)
    to_remove = []
    for index,l in enumerate(layers):
        if isinstance(l,MaxPool): to_remove.append(index-1)
    to_remove = to_remove[-3:] # slice to only removing the last three convs
    for conv_i in to_remove[::-1]: # index from reverse to not change indexes
        del layers[conv_i]
    return layers # here
