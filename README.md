(Fork and edit from repo: https://github.com/davidsandberg/facenet)

### 1. Colect data:
Colect data and push it into ```data_face``` as bellow:
```buildoutcfg
data_face/

├── HienHo
│   ├── HienHo_0001.jpg
│   ├── HienHo_0002.jpg
│   ├── ...
│   ├── HienHo_0014.jpg
│   └── HienHo_0015.jpg

├── TrucAnh
│   ├── TrucAnh_00013.jpg
│   ├── TrucAnh_0001.jpg
│   ├── ...
│   ├── TrucAnh_0014.jpg
│   └── TrucAnh_0015.jpg

├── TruongGiang
|  ├── TruongGiang_0001.jpg
|  ├── TruongGiang_0002.jpg
|  ├── ...
|  ├── TruongGiang_0015.jpg
|  └── TruongGiang_0016.jpg
```

### 2. Install requirements:
Install requirements ```pip install -r requirements.txt```

### 3. Download pretrain model:
Download model [VGGFace2](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) and unzip into ```models```, result as bellow:
```python
models
├── 20180402-114759.pb
├── model-20180402-114759.ckpt-275.data-00000-of-00001
├── model-20180402-114759.ckpt-275.index
└── model-20180402-114759.meta

```

### 4. Training:
run ```python train.py```. In this file, we will run 2 module:<br>

```python
align_mtcnn('data_face', 'face_align')
train('face_align/', 'models/20180402-114759.pb', 'models/your_model.pkl')
```

- ```align_mtcnn``` use MTCNN (Multi-task Cascaded Convolutional Networks) to detect face and crop.
![MTCNN](image/mtcnn.png)
  
    (image cre: [kpzhang93.github.io](https://kpzhang93.github.io/MTCNN_face_detection_alignment/))
  

- ```train``` facenet to recognize face
![FaceNet](image/facenet.png)
  
    (image cre: [towardsdatascience](https://towardsdatascience.com/a-facenet-style-approach-to-facial-recognition-dc0944efe8d1))

### 5. Detection:
- run ```python detection_cam.py``` if you want to run webcam or read file .mp4
```video_file=None``` if you want to run internal camera.

```python
if __name__ == '__main__':
    run('models', 'models/your_model.pkl', video_file='input_image/demo.mp4', output_file='output_image/demo.avi')
```


- run ```python detection_img.py``` if you want to open image

```python
if __name__ == '__main__':
    run('models', 'models/your_model.pkl', img_path='input_image/madona-check.jpg', output_file='output_image/result2.jpg')
```
