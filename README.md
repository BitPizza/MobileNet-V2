# MobileNet - V2

MobileNet V2 is based on an **inverted residue** structure where the shortcut connections are between the thin **bottleneck layers**.

The inverted residual with linear bottleneck takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. Features are subsequently projected back to a low-dimensional representation with a linear convolution.

## Inverted residuals 
###### Standard residual block
A standard residual blocks compress the wide channels to narrow channels with 1*1 plus 3*3 convolutions, and then expand the channels to wide again woth 1*1 convolutions for a concatenation with the skip connection. (wide - nattow - wide)

###### Inverted residual block
A inverted residual block using a 1*1 conv to expand channels and then adding the depthwise separable convolution for the following computation. This can reduce the parameters significatly. (narrow - wide - narrow)

![screen shot 2019-03-06 at 2 45 43 pm](https://user-images.githubusercontent.com/18547241/53912465-997fd900-401e-11e9-82b0-c0be0f2abf93.png)

## Linear bottlenects
Inverted residual block could hurt the performance if we use non-linear activations at the last convolution stage for a concatenation with the skip connection, since informations are squeezed instead of expanded, compared to the standard residual blocks. Thus, MobileNetV2 uses linear bottlenecks to provide a linear output before it's added to the initial activations. This provides a better performance compared to non-linearity in the bottleneck layer.

![screen shot 2019-03-06 at 2 47 59 pm](https://user-images.githubusercontent.com/18547241/53912603-e4015580-401e-11e9-84b8-3dc00a8e65ce.png)

## Performance
MobileNetV2 performs higher accuracy, less computation, and less memory usage during inference. 

# MobileNet - V1
The big idea behind MobileNet V1 is that convolutional layers, which are essential to computer vision tasks but are quite expensive to compute, can be replaced by so-called depthwise separable convolutions which used in Inception models to reduce the computation in the first few layers.

## Depthwise separable convolutions
* The MobileNet model is based on depthwise separable convolutions which is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution and a 1*1 convolution called a pointwise convolution. This factorization has the effect of drastically reducing computation and model size
* The depthwise convolution applies a single filter to each input channel, doesn’t create new features.
* The pointwise convolution applies a 1*1 convolution to create a linear combination of the outputs from the depthwise layer.

## Standard Convolutions vs. Depthwise Separable Convolutions
Standard convolutions have the computational cost of:

> DK • DK • M • N • DF • DF

Where the computational cost depends multiplicatively on the number of input channels M, the number of output channels N the kernel size DK * DK and the feature map size DF * DF. 

Depthwise convolution has a computational cost of:

> DK • DK • M • DF • DF

Depthwise convolution is only filters input channels, it doesn’t combine them to create new features. So an additional layer that computes a linear combination of the output of depthwise convolution via 1*1 convolution is needed in order to generate these new features.

Depthwise Separable convolutions cost:

> DK • DK • M • DF • DF + M • N • DF • DF

Which is the sum of the depthwise and 1*1 pointwise convolution

MobileNet uses 3*3 depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy. 

## Full Architecture of MobileNet V1
* The full architecture of MobileNet V1 consists of a regular 3×3 convolution as the very first layer, followed by 13 times depthwise separable convolutions block.
* No pooling layers in between these depthwise separable blocks. Instead, some of the depthwise layers have a stride of 2 to reduce the spatial dimensions of the data. When that happens, the corresponding pointwise layer also doubles the number of output channels. If the input image is 224×224×3 then the output of the network is a 7×7×1024 feature map.
* The convolution layers are followed by batch normalization.
* The activation function used by MobileNet is ReLU6 which like the well-known ReLU but it prevents activations from becoming too big.
```
y = min(max(0, x), 6)
```
* In a classifier based on MobileNet, there is typically a global average pooling layer at the very end, followed by a fully-connected classification layer or an equivalent 1×1 convolution, and a softmax.

## Two Global Hyperparameters
These hyperparameters allow the model builder to choose the right sized model for their application based on the constraints of the problem.
* Width Multiplier: α

The role of the width multiplier α is to thin a network uniformly at each layer. Width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly α2. α ∈ (0, 1], α = 1 is baseline MobileNet and α < 1 are reduced MobileNets

* Resolution Multiplier: ρ

We apply resolution multiplier to the input image and the internal representation of every layer is subsequently reduced by the same multiplier. Resolution multiplier has the effect of reducing computational cost by ρ2. ρ ∈ (0, 1], ρ = 1 is baseline MobileNet and ρ < 1 are reduced MobileNets

# Differences between MobileNet and MobileNet-V2

1. MoblieNet V2 still uses depthwise separable convolutions, but there are three convolutional layers in the block. The last two are the ones in MobileNet: a depthwise convolution that filters the inputs, followed by a 1*1 pointwise convolution layer.

2. In the MobileNet, the pointwise convolution either kept the number of channels the same or double them. In the MobileNetV2, it makes the number of channels smaller. So this layer is known as the projection layer in the MobileNetV2. It projects data with a high number of dimensions (channels) into a tensor with a much lower number of dimensions.

3. MobileNet V2 add a new layer in the block: expansion layer which is a 1*1 convolution. Its purpose is to expand the number of channels in the data before it goes into the depthwise convolution.

4. MobileNet V2 also adds the residual connection which helps with the flow of gradients through the network.

# Paper link
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
* [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)


