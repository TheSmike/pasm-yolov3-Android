# pasm-yolov3-Android
YOLOv3 implementation with Tensorflow on Android

This project contains an example of YoloV3 implementation on Android, the YoloV3 model was implemented through the library 
``org.tensorflow:tensorflow-android``.

Below is a list of steps taken to convert the YoloV3 model from darkflow to tensorflow for Android (command launched on Ubuntu inside Anaconda):

* clone DW2TF repository from here https://github.com/jinyu121/DW2TF to local folder
  * ex: /home/user/projects/
* Download (or train) YoloV3 model and weights in darknet format (.cfg and .weights)
* Launch DW2TF conversion as mentioned on the github page of DW2TF: https://github.com/jinyu121/DW2TF:
  * 
  ```
  python3 main.py \
    --cfg 'data/yolov3-tiny.cfg' \
    --weights 'data/yolov3-tiny.weights' \
    --output 'data/' \
    --prefix 'yolov3-tiny/' \
    --gpu 0
  ```
 * launch freeze_graph to have a single bp graph file:
 ```
  freeze_graph  \
  --input_graph yolov3-tiny.pb  \
  --input_checkpoint yolov3-tiny.ckpt  \
  --input_binary=true  \
  --output_graph=ultimate_yolov3.bp  \
  --output_node_names=yolov3-tiny/convolutional10/BiasAdd
  ```

NOTE: For older version of Yolo you can use darkflow tool https://github.com/thtrieu/darkflow, here an example after clone the repository:
```
./flow --model ../data/yolov2-tiny.cfg --load ../data/yolov2-tiny.weights --savepb
```


For more detail about Yolo look at offical page https://pjreddie.com/darknet/yolo/
