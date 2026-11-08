# 离线表情模型

- `face_detection_yunet_2023mar.onnx` 来自 [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。MIT 许可证见 [LICENSE-YuNet](LICENSE-YuNet)。SHA-256 为 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`。
- `facial_expression_recognition_mobilefacenet_2022july.onnx` 来自 [OpenCV Zoo Facial Expression Recognition](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition)。Apache 2.0 许可证见 [LICENSE-MobileFaceNet](LICENSE-MobileFaceNet)。SHA-256 为 `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9`。

对弈程序通过 OpenCV 在本机使用这两个模型。表情分类分数仅供当前棋局的难度控制使用。
