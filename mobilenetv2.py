"""
Available for tensorflow 1.4, keras 2.1.4
Add pretrained weight: imagenet
"""

from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, Activation, Add, Reshape
from keras.layers import ZeroPadding2D, BatchNormalization, GlobalAveragePooling2D
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils import get_file
from keras import backend as K

def MobileNet_v2(input_shape, nb_class, weights='imagenet', alpha=1.0):
    """
    Build MobileNet_v2 model
    
    :param input_shape - tuple: image input size, (224,224,3)
    :param nb_class - int: the number of classes
    :param weights - string: load weight
    :return model: mobileNet v2 model
    """
        
    BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/')    

    def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
        in_channels = K.int_shape(inputs)[-1]
        pointwise_conv_filters = int(filters*alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        x = inputs
        prefix = 'block_{}_'.format(block_id)

        if block_id:
            # Expand
            x = Conv2D(expansion*in_channels, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
            x = Activation(relu6, name=prefix + 'expand_relu')(x)
        else:
            prefix = 'expanded_conv_'

        # Depthwise
        if stride == 2:
            x = ZeroPadding2D(padding=_correct_pad(x, (3, 3)), name=prefix + 'pad')(x)
            
        x = DepthwiseConv2D(kernel_size=3,
                            strides=stride,
                            activation=None,
                            use_bias=False,
                            padding='same' if stride == 1 else 'valid',
                            name=prefix + 'depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
        x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

        # Project
        x = Conv2D(pointwise_filters,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

        if in_channels == pointwise_filters and stride == 1:
            return Add(name=prefix + 'add')([inputs, x])
        return x

    def _correct_pad(inputs, kernel_size):
        """Returns a tuple for zero-padding for 2D convolution with downsampling.
            # Arguments
                input_size: An integer or tuple/list of 2 integers.
                kernel_size - tuple: kernel size, (3,3)
            # Returns
                A tuple.
        """
        img_dim = 1
        input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

        if input_size[0] is None:
            adjust = (1, 1)
        else:
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

        correct = (kernel_size[0] // 2, kernel_size[1] // 2)

        return ((correct[0] - adjust[0], correct[0]),
                (correct[1] - adjust[1], correct[1]))
    
    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    
    # build model
    inputs = Input(shape=input_shape)
    
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D(padding=_correct_pad(inputs, (3, 3)))(inputs)
    x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='valid', use_bias=False, name='Conv1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = Activation(relu6, name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)
    
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)
    
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)
    
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)
    
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)
    
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)
    
    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    x = Conv2D(1280, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = Activation(relu6, name='out_relu')(x)

    x = GlobalAveragePooling2D()(x)
    
   # Create model.
    model = Model(inputs=inputs, outputs=x, name='mobilenetv2_%0.2f_%s' % (alpha, 224))

    if weights == 'imagenet':
        model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + str(alpha) + '_' + str(224) + '_no_top' + '.h5')
        weigh_path = BASE_WEIGHT_PATH + model_name
        weights_path = get_file(model_name, weigh_path, cache_subdir='models')
        
        model.load_weights(weights_path)


    x = Reshape((1, 1, 1280))(model.output)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(nb_class, (1, 1), padding='same')(x)

    x = Activation('softmax', name='softmax')(x)
    output = Reshape((nb_class,))(x)
    model = Model(inputs=inputs, outputs=output)
    
    return model

# run function
model = MobileNet_v2(input_shape=(224,224,3), nb_class=14, weights='imagenet')
