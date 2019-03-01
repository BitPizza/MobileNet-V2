# MobileNet - V2

MobileNet V2 is based on an inverted residue structure where the shortcut connections are between the thin bottleneck layers.


The inverted residual with linear bottleneck takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. Features are subsequently projected back to a low-dimensional representation with a linear convolution.

**Differences between MobileNet and MobileNet-V2** 

1. MoblieNet V2 still uses depthwise separable convolutions, but there are three convolutional layers in the block. The last two are the ones in MobileNet: a depthwise convolution that filters the inputs, followed by a 1*1 pointwise convolution layer.

2. In the MobileNet, the pointwise convolution either kept the number of channels the same or double them. In the MobileNetV2, it makes the number of channels smaller. So this layer is known as the projection layer in the MobileNetV2. It projects data with a high number of dimensions (channels) into a tensor with a much lower number of dimensions.

3. MobileNet V2 add a new layer in the block: expansion layer which is a 1*1 convolution. Its purpose is to expand the number of channels in the data before it goes into the depthwise convolution.

4. MobileNet V2 also adds the residual connection which helps with the flow of gradients through the network.
