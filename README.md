## 主流程
##### 1.主程序为main_detection.py
  主要功能：设置一些参数，如训练的文件夹，模型保存路径等，主要功能是调用models/yolo/yolo.py程序，并且传递参数。
##### 2.yolo.py
  主要功能：判断模式为训练还是测试，并调用models/yolo/train.py,test.py程序，并传递参数。
##### 3.train.py
  获取训练数据，并且调用yolo/net/darknet53.py来提取特征,并通调动models/yolo/YOLOv3.py生成loss，然后通过优化器来优化loss
并保存模型文件

### yolo_v3流程：
main（）-->yolov3函数-->>到train方法  --->>读取数据路劲,读取参数
（这类的数据是通过标注工具生成的 ，格式为 图片名，x_y_w_h_类别)
--->>产生一批次数据
--->>darknet53 --->> 把输入的数据 提取为特征图（例如13X13,26x26,52x52）的特征图
接下来把特征图映射到原图上（解码）来与原图的标注图求iou 
--->>求损失 1.这里的损失是在特征图上求取 

---

## 数据加载
数据加载程序位置为models/yolo/dataset/DataSet.py
程序DataSet.py中：
定义了一个DataSet类，包含了__init__构造函数，train_1w_loader，train_b_loader，val_1w_loader，val_b_loader，train_nms_loader，
test_loade六个方法。
定义了两个函数generator_batch，generator_val_batch这两个函数为普通函数不属于DataSet类。
train_1w_loader，train_b_loader两个方法调用了generator_batch函数来加载训练数据。
val_1w_loader，val_b_loader两个方法调用了generator_val_batch函数来加载验证数据。

##### models/yolo/YOLOv3.py
获取darknet53预测结果，并与label计算各种损失。

##### yolo/net/darknet53.py

---

### 定义了darknet53的网络结构
darknet53中53层仅算卷积层和全连接层其中卷积层52层，全连接层1层。softmax relu res relu不计入。

|             |kernel_size   |  filters   |   strides  |   padding   |   out_shape
|:---:        |:---:|:---:|:---:|:---:|:---:	 
conv1        | [3,3]      |     32     |    (1,1)    |    1    |   416X416X32
conv2        | [3,3]      |     64     |    (2,2)    |    1    |   208X208X64
conv3        | [1,1]      |     32     |    (1,1)    |    0    |   208X208X32
conv4        | [3,3]      |     64     |    (1,1)    |    1    |   208X208X64
conv2+conv4`res1`    |    |            |             |         |   208X208X64
conv5        | [3,3]      |     128    |    (1,1)    |    1    |   104X104X128
conv6        | [1,1]      |     64     |    (1,1)    |    0    |   104X104X64
conv7        | [3,3]      |     128    |    (2,2)    |    1    |   104X104X128
conv5+conv7`res2`   |     |            |             |         |   104X104X128
conv8        | [1,1]      |     64     |    (1,1)    |    0    |   104X104X64 
conv9        | [3,3]      |     128    |    (1,1)    |    1    |   104X104X128
conv9+res2`res3`    |     |            |             |         |   104X104X128
conv10       | [3,3]      |     256    |    (2,2)    |    1    |   52X52X256
conv11       | [1,1]      |     128    |    (1,1)    |    0    |   52X52X128
conv12       | [3,3]      |     256    |    (1,1)    |    1    |   52X52X256          
conv10+conv12`res4`|      |            |             |         |   52X52X256

---
|conv13             |[1,1]         | 128         |(1,1)   |0        |52X52X128 
|:---: |:---: |:---:|:---: |:---:|:---:	
conv14              |[3,3]          |256          |(1,1)   |1        |52X52X256 
conv14+res4`res5` |            |              |       |         |52X52X256 
---
##### X5(重复上述虚线框的结构5次)

|conv25          |[1,1]      |128        | (1,1)   |0   |52X52X128
|:---:|:---:|:---:|:---:|:---:|:---:
|conv26          |[3,3]      |256        |(1,1)    |1   |52X52X256
conv56+res10  `res11` |           |           |         |  |52X52X256
conv27           |[3,3]      |512        |(2,2)    |1   |26X26X512
conv28	         |[1,1]      |256        |(1,1)    |0   |26X26X256
conv29	         |[3,3]      |512        |(1,1)    |1   |26X26X512
conv27+conv29 `res12` |           |           |         |  |26X26X512

---
|conv30	     | [1,1]          | 256       |  (1,1)   |   0    |26X26X256
|:---:|:---:|:---:|:---:|:---:|:---:
conv31	     | [3,3]          | 512       |(1,1)    |  1     |26X26X512
conv31+res12 **res13** |           |           |         |  |26X26X512 
---
X5(重复上面的框(conv30-conv31)的结构5次)

|conv42	  |[1,1]            |256         |(1,1)      |0        |26X26X256
|:---:|:---:|:---:|:---:|:---:|:---:
|conv43	  |[3,3]            |512         |(1,1)      |1        |26X26X512
conv43+res18`res19`  |    |          |           |         |
conv44     |[3,3]           |1024      |(2,2)      |2        |13X13X1024
conv45	   |[1,1]           |512       |(1,1)      |0        |13X13X512
conv46	   |[3,3]           |1024      |(1,1)      |1        |13X13X1024
conv44+conv46`res20`  |    |          |           |         |13X13X1024

---
|conv47	      |[1,1]        |512         |(1,1)         |0         |13X13X512
|:---:|:---:|:---:|:---:|:---:|:---:
|conv48	      |[3,3]        |1024        |(1,1)         |1         |13X13X1024
|conv48+res20  `res21`|     |            |              |		   |
conv49	      |[1,1]        |512         |(1,1)         |0         |13X13X512
conv50	      |[3,3]        |1024        |(1,1)         |1         |13X13X1024
conv48+res21  `res22`|      |            |              |		   |
conv51	      |[1,1]        |512         |(1,1)         |0         |13X13X512
conv52	      |[3,3]        |1024        |(1,1)         |1         |13X13X1024
conv52+res22  `res23`|      |            |              |		   |

---
sacla0: all conv stride=1 conv([kernel],channel,outmapsize)     
scale1: all conv stride=1                                        
scale2: all conv stride=1 

         conv1([1,1],512,13X13) 
         conv2([3,3],1024,13X13)                                                                                                    
         conv3([1,1],512,13X13)
         conv4([3,3],1024,13X13)                                上采样，上采样计算公式：https://www.jianshu.com/p/f0674e48894c。
                                                                一般情况stride为几就扩大几倍
         conv5([1,1],512,13X13)-----[conv([1,1],256,13X13)]----->conv2d_transpose(k=[1,1],256,strides=(2,2),26X26)
                                                                 concate(conv43+conv2d_transpose,26X26X768) 512+256
         conv6([1,1],1024,13X13)                                 conv1([1,1],256,26X26)      
         conv7([1,1], ?,13X13) -------predict0                   conv2([3,3],512,26X26)
    ？含义:公式：（类别总数+5[四坐标点+置信度])×每个特征图预测anchor数    conv3([1,1],256,26X26)
                                                                 conv4([3,3],512,26X26)
                                                                 conv5([1,1],256,26X26) -------[conv([1,1],128,26X26)]----------->conv2d_transpose(k=[1,1],128,strides=(2,2),52X52)
                                                                 conv6([3,3],512,26X26)                                           concate(conv26+conv2d_transpose,52X52X384) 256+128
                                                                 conv7([1,1],?,26X26)-----------predict1 mask:3,4,5               conv1([1,1],128,52X52)
                                                                                                                                  conv2([3,3],256,52X52)
                                                                                                                                  conv3([1,1],128,52X52)
                                                                                                                                  conv4([3,3],256,52X52)
                                                                                                                                  conv5([1,1],128,52X52)
                                                                                                                                  conv6([3,3],256,52X52)
                                                                                                                                  conv7([1,1],?,52X52)-----------predict2





