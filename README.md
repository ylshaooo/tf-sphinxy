# tf-sphinxy
阿里天池算法大赛 服饰关键点检测任务

基于hourglass的tensorflow实现

## 运行环境
* python >= 3
* tensorflow >= 1.4
* numpy
* opencv >= 3
* matplotlib
* scikit-image

## 代码结构
* preprocess.ipynb : 数据预处理
* config.py : 模型参数
* utils.py : 工具函数和预定义变量
* model.py : 模型结构搭建，训练和测试流程
* datagen.py : 训练和测试数据生成，数据增补
* launcher.py : 训练和测试入口
* demo.ipynb : 可视化训练和测试的结果
## 训练步骤
本模型将数据集分为两个部分进行训练：上装和下装。两个部分的区别在于数据集和模型输出通道数量，在config.py中进行控制。</br></br>
实际训练中，两个部分的训练是在两块gpu上同时进行的。
1. 使用preprocess.ipynb预处理训练集。这将同时生成上装和下装数据
2. 根据需要调整config.py中的参数，主要包括以下几个部分：
    * #dataset：上装或下装控制，数据集位置
    * #model: 模型结构参数
    * #train: 模型训练参数
    * #saver：控制模型参数和summary的保存和载入
3. 运行 launcher.py
    >python launcher.py -m=train
4. 修改config.py，完成另一部分的训练（注意修改saver_dir或name属性，以防之前参数被覆盖）

## 测试步骤
测试同样分为上装和下装两部分。参数设置同训练步骤。
1. 使用preprocess.ipynb预处理测试集。
2. 修改config.py以载入训练好的模型参数。详见[载入训练参数](#load)部分
2. 运行 launcher.py
    > python launcher.py -m=test
3. 修改config.py中的#dataset设置，完成另一部分的测试（改变test_output_file设置，以防前一次操作被覆盖）

## 可视化测试
使用demo.ipynb可以对模型进行单步inference，在图片上显示预测的关键点位置。
<span id="load"></span>
## 载入训练参数
在训练、测试、可视化测试中都可以载入训练好的模型参数。可以通过修改config.py来实现。也可以在实例化Config对象后，修改其属性。载入的参数必须和当前的模型结构一致。</br></br>
参数保存在\<checkpoints>/\<load>位置。例如，参数保存在checkpoints/sphinx_20时，应令
* saver_dir=checkpoints
* load=sphinx_20

## 查看训练日志
训练流程中利用了tensorflow的summary功能。因此可以使用tensorboard查看\<logdir>下的log文件。如：
> tensorboard --logdir=logs
