# 使用PyTorch进行图像数据增强的方法
在这个笔记中，我将记录使用PyTorch进行图像数据增强的方法。这个笔记主要来自[李沐老师的视频课程中的内容](https://www.bilibili.com/video/BV17y4y1g76q?spm_id_from=333.999.0.0)和[李沐老师的动手学深度学习教材13.1图像增广](https://zh-v2.d2l.ai/chapter_computer-vision/image-augmentation.html)

首先，我们创建一个空白脚本`test.py`，并启用如下的名为`superglue`的conda虚拟环境。运行`pip list`命令，可以看到，`superglue`虚拟环境中的包的版本如下：
```
Package           Version
----------------- --------
black             22.1.0
click             8.0.3
cycler            0.11.0
fonttools         4.29.1
kiwisolver        1.3.2
matplotlib        3.5.1
mkl-fft           1.3.1
mkl-random        1.2.2
mkl-service       2.4.0
mypy-extensions   0.4.3
numpy             1.21.2
olefile           0.46
opencv-python     4.5.5.62
packaging         21.3
pathspec          0.9.0
Pillow            8.4.0
pip               22.0.3
platformdirs      2.5.0
pyparsing         3.0.7
python-dateutil   2.8.2
scipy             1.8.0
setuptools        60.8.2
six               1.16.0
tomli             2.0.1
torch             1.8.0
torchvision       0.9.0
tqdm              4.62.3
typing_extensions 4.0.1
wheel             0.37.1
```

我们首先在`test.py`里导入需要用的包：
``` python
import torch
import torchvision
from torch import nn
import cv2
```

## 常用的图像增广方法
本节内容参考[李沐老师的教材13.1.1常用的图像增广方法](https://zh-v2.d2l.ai/chapter_computer-vision/image-augmentation.html#id2)

由于我是使用ssh远程连接的开发机，因此，无法使用诸如`cv2.imshow()`这一类的方法。我的办法是：保存增强之后的8张图像，使用`code 1.jpg`、`code 2.jpg`、...、`code 8.jpg`这8个命令用vscode把图像打开。这样每次生成新的图像之后，就可以直接用vscode来查看新生成的图像了。

### 水平翻转
首先我们来试试水平翻转，也就是所谓的左右翻转。左右翻转图像通常不会改变对象的类别，这是最早且最广泛使用的图像增广方法之一。测试下述代码：
``` python
import torch
import torchvision
from torch import nn
from PIL import Image

img = Image.open("/data/XXX/YYY/cat1.jpg")


def apply(img, aug, num_rows=2, num_cols=4):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    temp = 1
    for image in Y:
        image.save(f"{temp}.jpg")
        temp += 1


apply(img, torchvision.transforms.RandomHorizontalFlip())
```
可以看到，8张新生成的图像展示出了随机的左右翻转。

### 上下翻转
上下翻转图像不如左右图像翻转那样常用。但是，至少对于这个示例图像，上下翻转不会妨碍识别。测试下述代码：
``` python
import torch
import torchvision
from torch import nn
from PIL import Image

img = Image.open("/data/XXX/YYY/cat1.jpg")


def apply(img, aug, num_rows=2, num_cols=4):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    temp = 1
    for image in Y:
        image.save(f"{temp}.jpg")
        temp += 1


apply(img, torchvision.transforms.RandomVerticalFlip())
```
可以看到，8张新生成的图像展示出了随机的上下翻转。

### 随机裁剪
我们可以通过对图像进行随机裁剪，使物体以不同的比例出现在图像的不同位置。 这也可以降低模型对目标位置的敏感性。

在下面的代码中，我们随机裁剪一个面积为原始面积10%到100%的区域，该区域的宽高比从0.5到2之间随机取值。 然后，区域的宽度和高度都被缩放到200像素。 在本节中（除非另有说明），$a$和$b$之间的随机数指的是在区间$[a,b]$中通过均匀采样获得的连续值。

测试如下的代码：
``` python
import torch
import torchvision
from torch import nn
from PIL import Image

img = Image.open("/data/XXX/YYY/cat1.jpg")


def apply(img, aug, num_rows=2, num_cols=4):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    temp = 1
    for image in Y:
        image.save(f"{temp}.jpg")
        temp += 1


shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2)
)

apply(img, shape_aug)
```
可以看到，8张新生成的图像被统一缩放成了$200 \times 200$的尺寸，并且经过了随机裁剪。

### 改变颜色
另一种增广方法是改变颜色。 我们可以改变图像颜色的四个方面：亮度、对比度、饱和度和色调。 在下面的示例中，我们随机更改图像的亮度，随机值为原始图像的50%（也就是(1-0.5)）到150%（也就是(1+0.5)）之间。

测试下述代码：
``` python
import torch
import torchvision
from torch import nn
from PIL import Image

img = Image.open("/data/XXX/YYY/cat1.jpg")


def apply(img, aug, num_rows=2, num_cols=4):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    temp = 1
    for image in Y:
        image.save(f"{temp}.jpg")
        temp += 1


apply(
    img,
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
)
```
可以看到，8张新生成的图像被随机调整了亮度。有些图像变得更暗，有些图像变得更亮。

同样，我们可以随机更改图像的色调。测试下述代码：
``` python
import torch
import torchvision
from torch import nn
from PIL import Image

img = Image.open("/data/XXX/YYY/cat1.jpg")


def apply(img, aug, num_rows=2, num_cols=4):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    temp = 1
    for image in Y:
        image.save(f"{temp}.jpg")
        temp += 1


apply(
    img,
    torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
)
```
可以看到，8张新生成的图像被随机调整了色调。本次生成的8张图像中的毛有蓝色、蓝绿色、粉色、紫色等颜色。

我们还可以创建一个RandomColorJitter实例，并设置如何同时随机更改图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。测试下述代码：
``` python
import torch
import torchvision
from torch import nn
from PIL import Image

img = Image.open("/data/XXX/YYY/cat1.jpg")


def apply(img, aug, num_rows=2, num_cols=4):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    temp = 1
    for image in Y:
        image.save(f"{temp}.jpg")
        temp += 1


color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
)

apply(img, color_aug)
```
可以看到，8张新生成的图像被随机调整了亮度和色调。既有亮度的调整，也有色调的调整。


### 同时使用多种图像增强方法
在实践中，我们将结合多种图像增广方法。比如，我们可以通过使用一个Compose实例来综合上面定义的不同的图像增广方法，并将它们应用到每个图像。测试下述代码：
``` python
import torch
import torchvision
from torch import nn
from PIL import Image

img = Image.open("/data/XXX/YYY/cat1.jpg")


def apply(img, aug, num_rows=2, num_cols=4):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    temp = 1
    for image in Y:
        image.save(f"{temp}.jpg")
        temp += 1


color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
)

shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2)
)

augs = torchvision.transforms.Compose(
    [torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug]
)

apply(img, augs)
```
可以看到，8张新生成的图像被缩放到了$200 \times 200$的尺寸，并且被随机左右翻转，随机调整了亮度和色调。我们明白了：**使用torchvision.transforms.Compose()函数来组合使用多种图像增强方法**

**另外，有一点需要注意：在李沐老师的视频教程里提到，一般来说，图片增强torchvision.transforms.Compose()函数的最后，需要跟一个torchvision.transforms.ToTensor()，把图像转换成PyTorch张量，用于训练。**

本次图像增强方法的介绍暂且先到这里，之后或许会更新更多的图像增强方法。