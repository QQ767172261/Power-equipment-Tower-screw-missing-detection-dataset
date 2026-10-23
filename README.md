电塔螺钉缺失检测数据集，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7bafef3db1fc44c88fa2a00b8239d280.png)
989张，yolo，voc，coco三种标注方式
图像尺寸:640*640
类别数量:3类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9b66d0f6aba94038a8a336cd294bb2ae.png)

训练集图像数量:775; 验证集图像数量:140； 测试集图像数量:74
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c02bacd57831486fb3e810abb517b17b.png)

Screw - 螺钉：763, 3864
Noscrew - 漏装螺钉：72, 99
Sidescrew - 侧面螺钉：416, 1638
image num: 989![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aa3398b4142d49cf9cce7c3903e01a31.png)
# 电塔螺钉缺失检测数据集
## 数据集信息表
|项目|详情|
| ---- | ---- |
|数据集名称|电塔螺钉缺失检测数据集|
|总图像数量|989张|
|图像分辨率|640×640|
|标注格式|YOLO、VOC、COCO 三种|
|类别总数|3类|
|训练集|775张|
|验证集|140张|
|测试集|74张|

### 类别分布表
|类别英文|类别中文|包含该类图像数量|标注框总数|
| ---- | ---- | ---- | ---- |
|Screw|螺钉|763|3864|
|Noscrew|漏装螺钉（螺钉缺失）|72|99|
|Sidescrew|侧面螺钉|416|1638|

## 标签
`Screw`、`Noscrew`、`Sidescrew`

## 关键词
电塔螺钉检测、输电铁塔缺陷检测、螺钉缺失识别、无人机电力巡检、YOLO目标检测、铁塔构件缺陷、电力设施巡检

## 数据集yaml配置文件 `tower_screw.yaml`
```yaml
path: ./tower_screw_dataset
train: images/train
val: images/val
test: images/test

names:
  0: Screw
  1: Noscrew
  2: Sidescrew
```

## YOLOv11 训练代码 `train_tower_screw.py`
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv11n预训练权重
    model = YOLO("yolo11n.pt")

    # 开始训练
    results = model.train(
        data="tower_screw.yaml",
        epochs=100,
        imgsz=640,
        batch=12,
        device=0,
        project="tower_screw_output",
        name="yolo11_tower_screw",
        conf=0.25,
        iou=0.45
    )
```

## 推理测试代码 `predict_tower_screw.py`
```python
from ultralytics import YOLO

model = YOLO("./tower_screw_output/yolo11_tower_screw/weights/best.pt")

# 图片推理
res = model.predict("test.jpg", save=True, conf=0.25)
for r in res:
    print(f"检测目标总数：{len(r.boxes)}")
    for box in r.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()[0]
        print(f"类别：{cls_name},置信度:{conf:.3f},坐标:{xyxy}")
```

## 环境安装
```bash
pip install ultralytics torch opencv-python
```


