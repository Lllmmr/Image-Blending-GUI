## Image Blending GUI

算法设计与分析课程project

一个能够进行图像融合的图形用户界面

简化版（仅保留泊松编辑功能）见：[Poisson Image Editing UI](https://github.com/Lllmmr/Poisson-Image-Editing-UI)

### 运行

运行该项目需要`python3`环境

并安装`pyqt5`,`pyqt5-tools`, `numpy`, `opencv`, `opencv-contrib` ,  `tensorflow`, `scikit-image` , `chainer`

```
$ pip install PyQt5 pyqt5-tools numpy opencv-python opencv-contrib-python tensorflow==1.14.0 scikit image chainer
```

（注：最好使用1.14.0版本的tensorflow，其他版本可能会存在问题）

并下载模型

[北大网盘](https://disk.pku.edu.cn:443/link/59A1E10766F04EE5CEFAD5C5395B5166)（有效期至2020/7/3）

[百度网盘](https://pan.baidu.com/s/1cEWvf6Op3k4QzVPQzcovKw)（ 提取码: vkwe）

下载后，将目录下`blending_gan.npz`放到`GP_GAN_blending`文件夹中

将目录下`model`文件夹整个放到`deepimageharm`文件夹中

### 操作指南

#### 图片操作：

`File->Open->dst_img/src_img`加载目标（背景）图像

按住空格键不放，可以用鼠标左键对源图像位置进行拖动

使用鼠标滚轮可以对源图像进行缩放

使用鼠标左键圈出源图像要进行编辑的区域

单击鼠标右键可以取消对编辑区域的选择

将图片移动/缩放/选区完毕后，可以使用右侧的按键进行图像编辑

`File->Save As`将图像编辑的结果保存到本地

#### 按键及功能：

右侧上方按键为对各种图像编辑模式的选择，模式都选择完后，即可按右侧最下方`Poisson Image Editing`按键进行图像编辑

最上方三个选项`Normal`, `Mixed`, `Transfer`分别表示“正常”，“混合”和“特征转换”

混合模式可以进行图像中线条的迁移，例如将文字迁移到背景上

特征转换模式可以进行纹路特征的转换，例如橘子和梨表面的特征转换

第四个选项`Local Changes`表示利用泊松编辑对源图像进行局部的改动，该模式下目标图像和源图像为同一张，因此只显示通过`File->Open->src_img`加载的图像

在该模式下，还需至少选择`Flattening`, `Illumination`, `Color`中的至少一项（建议只选一项）

`Flattening`对图像的编辑区域做扁平化处理

`low`, `high`两个滑动条调整扁平化的参数

`Illumination`调整图像所选区域的亮度

`a`, `b`两个滑动条调整亮度变化的参数

`Color`改变所选区域的颜色

通过下方按键选择要变成的颜色，选中`Gray`可将背景变为灰色

#### 图像编辑方法：

通过右下方四个不同的按键使用不同的编辑方法

其中，除了泊松编辑以外不支持对编辑模式的调整

因此若在上方选择了`Local Changes`选项，则点击按键不会做出反应

`Laplacian Pyramids`方法是对泊松编辑的改进，效果无明显变化

`Deep Image Harmonization`对选区要求较高

`GT-GAN`适合对结构较为相近的风景照进行编辑

#### 拓展功能：

`File->Save Mask/Save Src`可将经过缩放、裁剪后的源图像及其单通道mask保存到本地，方便其他未加入GUI的算法获取ROI