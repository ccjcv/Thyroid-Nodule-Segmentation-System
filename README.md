# Thyroid-Nodule-Segmentation-System
基于QT，mysql与Transformer的甲状腺图像分割系统。（软著已登记，请勿商用）

项目描述：使用QT进行开发，配置基于mysql的用户登录系统以及子线程进行客户端与服务端之间的socker通信，子线程传输图像文件，后端使用python图像分割神经网络进行分割，分割结果于界面显示。

登录：
![image](https://github.com/ccjcv/Thyroid-Nodule-Segmentation-System/blob/main/%E7%99%BB%E5%BD%95.PNG)

套接字通信：
![image](https://github.com/ccjcv/Thyroid-Nodule-Segmentation-System/blob/main/%E9%80%9A%E4%BF%A1.PNG)

图片传输：
![image](https://github.com/ccjcv/Thyroid-Nodule-Segmentation-System/blob/main/%E6%96%87%E4%BB%B6%E4%BC%A0%E8%BE%932.PNG)

客户端多线程：
![image](https://github.com/ccjcv/Thyroid-Nodule-Segmentation-System/blob/main/%E5%AE%A2%E6%88%B7%E7%AB%AF%E5%A4%9A%E7%BA%BF%E7%A8%8B.PNG)

BPAT-UNet模型的分割与测试：
BPAT-UNet: Boundary Preserving Assembled Transformer UNet for Ultrasound Thyroid Nodule Segmentation(2023 CCJ)
https://github.com/ccjcv/BPAT-UNet
![image](https://github.com/ccjcv/Thyroid-Nodule-Segmentation-System/blob/main/BPAT-UNet%E5%88%86%E5%89%B2%E4%B8%8E%E6%B5%8B%E8%AF%95.PNG)

UNet模型的分割与测试：
![image[(https://github.com/ccjcv/Thyroid-Nodule-Segmentation-System/blob/main/UNet%E5%88%86%E5%89%B2%E4%B8%8E%E6%B5%8B%E8%AF%95.PNG)

UTNet模型的分割与测试：
![image](https://github.com/ccjcv/Thyroid-Nodule-Segmentation-System/blob/main/UTNet%E5%88%86%E5%89%B2%E4%B8%8E%E6%B5%8B%E8%AF%95.PNG)

TransUNet模型的分割与测试：
![image](https://github.com/ccjcv/Thyroid-Nodule-Segmentation-System/blob/main/TransUnet%E5%88%86%E5%89%B2%E4%B8%8E%E6%B5%8B%E8%AF%95.PNG)

Segformer模型的分割与测试：
![image](https://github.com/ccjcv/Thyroid-Nodule-Segmentation-System/blob/main/Segformer%E5%88%86%E5%89%B2%E4%B8%8E%E6%B5%8B%E8%AF%95.PNG)
