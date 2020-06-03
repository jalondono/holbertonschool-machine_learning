0x04. Convolutions and Pooling
==============================

Specializations - Machine Learning ― Math

_by Alexa Orrico, Software Engineer at Holberton School_

Ongoing project - started 06-03-2020, must end by 06-06-2020 (in 3 days) - you're done with 0% of tasks.

Checker will be released at 06-04-2020 12:00 PM

QA review fully automated.

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/ed9ca14839ad0201f19e.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4b9e83ba782db1b20f20fd1db3780a823304d8bc519348cf53f734ad89ee4134)

Resources
---------

**Read or watch**:

*   [Convolution](/rltoken/xbzvTRaBX2LUOM7A1NazVQ "Convolution")
*   [Kernel (image processing)](/rltoken/lsI2xbijDWAiKDFuCYkcAA "Kernel (image processing)")
*   [Image Kernels](/rltoken/Qeq8i5dhkR9Tlp-IgFDzQw "Image Kernels")
*   [Undrestanding Convolutional Layers](/rltoken/g8kHsJFzC51whRSEupvidw "Undrestanding Convolutional Layers")
*   [What is max pooling in convolutional neural networks?](/rltoken/crEEAb4sDHc30ntPwY-qsQ "What is max pooling in convolutional neural networks?")
*   [Edge Detection Examples](/rltoken/nV4RcnhzFvjLfl7z2k5-Cw "Edge Detection Examples") (_Note: I suggest watching this video at 1.5x - 2x speed_)
*   [Padding](/rltoken/WZ_a9ntwdJ_AU51W46KOlw "Padding") (_Note: I suggest watching this video at 1.5x - 2x speed_)
*   [Strided Convolutions](/rltoken/yupMT890fCjD5XVyogDkmg "Strided Convolutions") (_Note: I suggest watching this video at 1.5x - 2x speed_)
*   [Convolutions over Volumes](/rltoken/vdFQg1m-0BJ_s0lg8b3fkg "Convolutions over Volumes") (_Note: I suggest watching this video at 1.5x - 2x speed_)
*   [Pooling Layers](/rltoken/Z0dPond1Oi9a04MiWsbgXA "Pooling Layers") (_Note: I suggest watching this video at 1.5x - 2x speed_)
*   [numpy.pad](/rltoken/QkWjIyjvPImhaA4HJGGz-w "numpy.pad")
*   [A guide to convolution arithmetic for deep learning](/rltoken/ZJItcZYPPp4e6bAV-xaMkw "A guide to convolution arithmetic for deep learning")

Learning Objectives
-------------------

At the end of this project, you are expected to be able to [explain to anyone](/rltoken/qTkzSfy5DRC1VkBEPRt4Ag "explain to anyone"), **without the help of Google**:

### General

*   What is a convolution?
*   What is max pooling? average pooling?
*   What is a kernel/filter?
*   What is padding?
*   What is “same” padding? “valid” padding?
*   What is a stride?
*   What are channels?
*   How to perform a convolution over an image
*   How to perform max/average pooling over an image

Requirements
------------

### General

*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
*   Your files will be executed with `numpy` (version 1.15)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should use the `pycodestyle` style (version 2.5)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
*   Unless otherwise noted, you are not allowed to import any module except `import numpy as np`
*   You are not allowed to use `np.convolve`
*   All your files must be executable
*   The length of your files will be tested using `wc`

More Info
---------

### Testing

Please download [this dataset](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/animals_1.npz "this dataset") for use in some of the following main files.

* * *

Tasks
-----

Done?!  
Help

×

#### Students who are done with "0. Valid Convolution"

#### 0\. Valid Convolution mandatory

Write a function `def convolve_grayscale_valid(images, kernel):` that performs a valid convolution on grayscale images:

*   `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
*   `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the convolved images

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 0-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale_valid(images, kernel)
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./0-main.py 
    (50000, 28, 28)
    (50000, 26, 26)
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ce4e2c023fa5f946069e7c5bfd59aaf456302419fce5b8c6f416c0c3df57b32e)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6e1b02cc87497f12f17e.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=010f2ef3b565968da0a0903ddef7914734e6cc69d7a28a729f774b1883d676de)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `0-convolve_grayscale_valid.py`

Done?!  
Help

×

#### Students who are done with "1. Same Convolution"

#### 1\. Same Convolution mandatory

Write a function `def convolve_grayscale_same(images, kernel):` that performs a same convolution on grayscale images:

*   `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
*   `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   if necessary, the image should be padded with 0’s
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the convolved images

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 1-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale_same(images, kernel)
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./1-main.py 
    (50000, 28, 28)
    (50000, 28, 28)
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ce4e2c023fa5f946069e7c5bfd59aaf456302419fce5b8c6f416c0c3df57b32e)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/b32bba8fea86011c3372.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=147ae289f70733ddfcc6f5891df17d626a644cd352e3b64669f78e293790aa91)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `1-convolve_grayscale_same.py`

Done?!  
Help

×

#### Students who are done with "2. Convolution with Padding"

#### 2\. Convolution with Padding mandatory

Write a function `def convolve_grayscale_padding(images, kernel, padding):` that performs a convolution on grayscale images with custom padding:

*   `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
*   `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   `padding` is a tuple of `(ph, pw)`
    *   `ph` is the padding for the height of the image
    *   `pw` is the padding for the width of the image
    *   the image should be padded with 0’s
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the convolved images

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 2-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale_padding(images, kernel, (2, 4))
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./2-main.py 
    (50000, 28, 28)
    (50000, 30, 34)
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ce4e2c023fa5f946069e7c5bfd59aaf456302419fce5b8c6f416c0c3df57b32e)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/3f178b675c1e2fdc86bd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=83616635a38406aef49e981fff564d648ede293ea5225d462d7fc0d38e2a604a)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `2-convolve_grayscale_padding.py`

Done?!  
Help

×

#### Students who are done with "3. Strided Convolution"

#### 3\. Strided Convolution mandatory

Write a function `def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):` that performs a convolution on grayscale images:

*   `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
*   `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   `padding` is either a tuple of `(ph, pw)`, ‘same’, or ‘valid’
    *   if ‘same’, performs a same convolution
    *   if ‘valid’, performs a valid convolution
    *   if a tuple:
        *   `ph` is the padding for the height of the image
        *   `pw` is the padding for the width of the image
    *   the image should be padded with 0’s
*   `stride` is a tuple of `(sh, sw)`
    *   `sh` is the stride for the height of the image
    *   `sw` is the stride for the width of the image
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed _Hint: loop over `i` and `j`_
*   Returns: a `numpy.ndarray` containing the convolved images

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 3-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(2, 2))
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./3-main.py 
    (50000, 28, 28)
    (50000, 13, 13)
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ce4e2c023fa5f946069e7c5bfd59aaf456302419fce5b8c6f416c0c3df57b32e)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/036ccba7dccf211dab76.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=160095eb7e630886a2e57fe7e38f7c079b34a83137ffb92ced9561eff9ba8428)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `3-convolve_grayscale.py`

Done?!  
Help

×

#### Students who are done with "4. Convolution with Channels"

#### 4\. Convolution with Channels mandatory

Write a function `def convolve_channels(images, kernel, padding='same', stride=(1, 1)):` that performs a convolution on images with channels:

*   `images` is a `numpy.ndarray` with shape `(m, h, w, c)` containing multiple images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
    *   `c` is the number of channels in the image
*   `kernel` is a `numpy.ndarray` with shape `(kh, kw, c)` containing the kernel for the convolution
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   `padding` is either a tuple of `(ph, pw)`, ‘same’, or ‘valid’
    *   if ‘same’, performs a same convolution
    *   if ‘valid’, performs a valid convolution
    *   if a tuple:
        *   `ph` is the padding for the height of the image
        *   `pw` is the padding for the width of the image
    *   the image should be padded with 0’s
*   `stride` is a tuple of `(sh, sw)`
    *   `sh` is the stride for the height of the image
    *   `sw` is the stride for the width of the image
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the convolved images

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 4-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_channels = __import__('4-convolve_channels').convolve_channels
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/animals_1.npz')
        images = dataset['data']
        print(images.shape)
        kernel = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]], [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]], [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])
        images_conv = convolve_channels(images, kernel, padding='valid')
        print(images_conv.shape)
    
        plt.imshow(images[0])
        plt.show()
        plt.imshow(images_conv[0])
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./4-main.py 
    (10000, 32, 32, 3)
    (10000, 30, 30)
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4f037eb728f25324db7087adce858f75e9e81e3520fa6e87ac65e2fd77db0a6f)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/8bc039fb38d60601b01a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=72a1e5bd4da2791d65bc7d889d58a0981ea37547016cc8191829ff399a73042e)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `4-convolve_channels.py`

Done?!  
Help

×

#### Students who are done with "5. Multiple Kernels"

#### 5\. Multiple Kernels mandatory

Write a function `def convolve(images, kernels, padding='same', stride=(1, 1)):` that performs a convolution on images using multiple kernels:

*   `images` is a `numpy.ndarray` with shape `(m, h, w, c)` containing multiple images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
    *   `c` is the number of channels in the image
*   `kernels` is a `numpy.ndarray` with shape `(kh, kw, c, nc)` containing the kernels for the convolution
    *   `kh` is the height of a kernel
    *   `kw` is the width of a kernel
    *   `nc` is the number of kernels
*   `padding` is either a tuple of `(ph, pw)`, ‘same’, or ‘valid’
    *   if ‘same’, performs a same convolution
    *   if ‘valid’, performs a valid convolution
    *   if a tuple:
        *   `ph` is the padding for the height of the image
        *   `pw` is the padding for the width of the image
    *   the image should be padded with 0’s
*   `stride` is a tuple of `(sh, sw)`
    *   `sh` is the stride for the height of the image
    *   `sw` is the stride for the width of the image
*   You are only allowed to use three `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the convolved images

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 5-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve = __import__('5-convolve').convolve
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/animals_1.npz')
        images = dataset['data']
        print(images.shape)
        kernels = np.array([[[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[0, -1, 1], [0, -1, 1], [0, -1, 1]]],
                           [[[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]], [[5, 0, 0], [5, 0, 0], [5, 0, 0]], [[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]],
                           [[[0, 1, -1], [0, 1, -1], [0, 1, -1]], [[-1, 0, -1], [-1, 0, -1], [-1, 0, -1]], [[0, -1, -1], [0, -1, -1], [0, -1, -1]]]])
    
        images_conv = convolve(images, kernels, padding='valid')
        print(images_conv.shape)
    
        plt.imshow(images[0])
        plt.show()
        plt.imshow(images_conv[0, :, :, 0])
        plt.show()
        plt.imshow(images_conv[0, :, :, 1])
        plt.show()
        plt.imshow(images_conv[0, :, :, 2])
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./5-main.py 
    (10000, 32, 32, 3)
    (10000, 30, 30, 3)
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4f037eb728f25324db7087adce858f75e9e81e3520fa6e87ac65e2fd77db0a6f)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6d6319bb470e3566e885.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d277054f71bd033f7bed42302a9353afeb6db1fcf1d3cdf543ab05ac780b098c)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/1370dd6200e942eee8f9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=5110ec305d4ae1c5efa343b171c99d0478a6e85266c056d4e852e9de532858ca)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/a24b7d741b3c378f9f89.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d5f9222412641e6afe75d1371af0ad2e3508ab598b2656b5a3716fe7b5d4b6bb)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `5-convolve.py`

Done?!  
Help

×

#### Students who are done with "6. Pooling"

#### 6\. Pooling mandatory

Write a function `def pool(images, kernel_shape, stride, mode='max'):` that performs pooling on images:

*   `images` is a `numpy.ndarray` with shape `(m, h, w, c)` containing multiple images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
    *   `c` is the number of channels in the image
*   `kernel_shape` is a tuple of `(kh, kw)` containing the kernel shape for the pooling
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   `stride` is a tuple of `(sh, sw)`
    *   `sh` is the stride for the height of the image
    *   `sw` is the stride for the width of the image
*   `mode` indicates the type of pooling
    *   `max` indicates max pooling
    *   `avg` indicates average pooling
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the pooled images

    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 6-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    pool = __import__('6-pool').pool
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/animals_1.npz')
        images = dataset['data']
        print(images.shape)
        images_pool = pool(images, (2, 2), (2, 2), mode='avg')
        print(images_pool.shape)
    
        plt.imshow(images[0])
        plt.show()
        plt.imshow(images_pool[0] / 255)
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./6-main.py 
    (10000, 32, 32, 3)
    (10000, 16, 16, 3)
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4f037eb728f25324db7087adce858f75e9e81e3520fa6e87ac65e2fd77db0a6f)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/ab4705f939c3a8e487bb.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T160746Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=b645cf6b350843015975e0b08394c2fdaa1953ebe9204059ac2e9d3c002290d0)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `6-pool.py`