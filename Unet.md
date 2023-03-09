Определение маски Unet сеть для нахождения и выделения контура 


# Math 
1. Convolution

[Convolution explained](https://www.youtube.com/watch?v=KuXjwB4LzSA&ab_channel=3Blue1Brown)

	(1,2,3) * (4,5,6) = (4,12,28,27,18)
```	
    123      123    123
654        654     654
            
           1x4   1x5+2x4 

```



For blurring an image 
For detecting variation in the pixel value 
For sharpening 

Kernel is a table which we use as the second term in convolution

Kernel for detecting color variation for example may look like

|-0.25|-0.5|-0.25|
| :---        |    :----:   |          ---: |
|0|0|0|
|0.25|0.5|0.25|



 Image * Kernel 

All depends on different Kernel s

So convolutional neural network use data to figure out what the kernels should be in the first place!

Fast way to compute convolution fftconvolve 

Let a = [], b = [] - lists which convolution we want ot calculate 

Denote f(x) as a polynom with a<sub>i</sub> as coefficients and g(x) as a polynom with b<sub>i</sub> as a coefficients. Than calculate fast fourier transform(evaluating polynoms at some specific imaginary points) than multiply to results and than implement inverst FFT. O(n*log(n)).


2. ReLU 
[ReLu explanation](https://iq.opengenus.org/relu-activation)
ReLU - Rectified Linear Unit

f(x) = max(0, x)

We use ReLU as an activation function. So such function can activate the computation  when our data have reached specific values. 

Important remark - it must be non linear in order to  to make the network learn complex patterns in the data.



# UNET - U shape NET

## Why use UNET?

- U-Net learns segmentation in an end-to-end setting.
You input a raw image and get a segmentation map as the output.

- U-Net is able to precisely localize and distinguish borders.
Performs classification on every pixel so that the input and output share the same size.

- U-Net uses very few annotated images.
Data augmentation with elastic deformations reduces the number of annotated images required for training.

## Architecture of UNET 

![basic architecture](Unet_arc.png)

[Uncomplicated explanation of UNET](https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5)
[Another Explanation](https://www.youtube.com/watch?v=GAYJ81M58y8&ab_channel=DigitalSreeni)

<dl>
<dt>contracting </dt>
<dd>Сжимающй<dd>
<dt>expansive</dt>
<dd>Расширительный</dd>
</dl>


"The architecture is symmetric and consists of two major parts — the left part is called contracting path, which is constituted **by the general convolutional process**; the right part is expansive path, which is constituted by **transposed 2d convolutional layers**"

So, the process:
1. Layer 1  
From 1 image do a convolution, which will produce 64 another images, and do a convolution on them. We are using 3 x 3 Kernel. Than max pool all of them into 64 image with a resolution of 2 times lower 

2. Repeat the process n steps 

3. Upsample m images in order to get 2 times more resolution but with half of images.  


4. While we are upscaling images on the n-th step we are also Concatinatate filters from the n-th step when we compressed them. So when double our number of "features" of images. 

5. Repeat until we've got 1 image.

### Useful links


[Original source code, trained network](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)


