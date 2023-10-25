# YOLOv5-Malaysian-Skin-Type-Reader
## yolov5_skin_condition_detection folder: the source code for training the YOLOv5 model.
## the other folders and files: the source code of the Skin Type Reader system.

This application, skin type reader focuses on analyzing the facial skin of Malaysians with object detection and deep learning algorithms. YOLOv5 is employed to detect users' facial skin conditions, such as acne, pigment, enlarged pores, uneven skin, blackheads, etc. Then, based on the detected skin conditions, it further classifies the user's skin type into the normal, oily, sensitive, or dry groups. Based on the classified skin type, the system can also provide skincare products suitable for the user’s 
skin type. Facial skin images of Malaysians are collected as the datasets used to train the YOLOv5 model. 

YOLOv5 model is trained using Python via Google Colaboratory. The user interface of the system is developed using HTML, CSS, JavaScript via Atom. In order to integrate between the trained YOLOv5 model and the user interface, Flask is used with Visual Studio Code. Flask provides a simple and straightforward approach to render HTML templates and transfer data from Python model to these HTML templates.

## The trained YOLOv5 model was able to achieve approximately 87.3% precision, 79.7% recall and 86.7% mAP50 in detecting all categories of skin conditions.
![image](https://github.com/yinghueygan/YOLOv5-Malaysian-Skin-Type-Reader/assets/90696965/34f81c68-cfb1-4ace-b808-fa1db8ef6b4a)
![image](https://github.com/yinghueygan/YOLOv5-Malaysian-Skin-Type-Reader/assets/90696965/9762f93d-9aaa-4b16-bb28-f2de1bd801dd)

## Skin Type Classification Results:
Dry Skin:
![image](https://github.com/yinghueygan/YOLOv5-Malaysian-Skin-Type-Reader/assets/90696965/8eda55b7-3d6d-425d-bcf3-ff8d22e07bd3)
Normal Skin:
![image](https://github.com/yinghueygan/YOLOv5-Malaysian-Skin-Type-Reader/assets/90696965/0fcd0d30-b043-4b81-bec0-7e9b2c1c3380)
Oily Skin:
![image](https://github.com/yinghueygan/YOLOv5-Malaysian-Skin-Type-Reader/assets/90696965/6eff26ef-6ed9-40a9-8fb4-604141655852)
Sensitive Skin:
![image](https://github.com/yinghueygan/YOLOv5-Malaysian-Skin-Type-Reader/assets/90696965/860b8252-5d52-4a0a-888f-7f61767e5b4b)

## Users can also lively detect their facial skin imperfections:
![image](https://github.com/yinghueygan/YOLOv5-Malaysian-Skin-Type-Reader/assets/90696965/622ab2c6-d62e-4fbf-9568-f4b2283da9fc)
