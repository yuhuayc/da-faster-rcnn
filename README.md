#  Domain Adaptive Faster R-CNN for Object Detection in the Wild 

This is the implementation of our CVPR 2018 work 'Domain Adaptive Faster R-CNN for Object Detection in the Wild'. The aim is to improve the cross-domain robustness of object detection, in the screnario where training and test data are drawn from different distributions. The original paper can be found [here](https://arxiv.org/pdf/1803.03243.pdf). 

If you find it helpful for your research, please consider citing:

    @inproceedings{chen2018domain,
      title={Domain Adaptive Faster R-CNN for Object Detection in the Wild},
      author={Chen, Yuhua and Li, Wen and Sakaridis, Christos and Dai, Dengxin and Van Gool, Luc},
      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
      year={2018}
    }

If you encounter any problems with the code, please contact me at yuhua[dot]chen[at]vision[dot]ee[dot]ethz[dot]ch

### Acknowledgment

The implementation is built on the python implementation of Faster RCNN [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

### Usage
1. Build Caffe and pycaffe (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

2. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    
3. Follow the instrutions of [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) to download related data.
    
4. Prepare the dataset, source domain data should start with the filename 'source_', and target domain data with 'target_'.

5. To train the Domain Adaptive Faster R-CNN:
    ```Shell
    cd $FRCN_ROOT
    ./tools/train_net.py --gpu {GPU_ID} --solver models/da_faster_rcnn/solver.prototxt --weights data/imagenet_models/VGG16.v2.caffemodel --imdb voc_2007_trainval --iters  {NUM_ITER}  --cfg  {CONFIGURATION_FILE}
    
### Example
An example of adapting from **Cityscapes** dataset to **Foggy Cityscapes** dataset is provided:
1. Download the datasets from [here](https://www.cityscapes-dataset.com/downloads/). Specifically, we will use **gtFine_trainvaltest.zip**, **leftImg8bit_trainvaltest.zip** and **leftImg8bit_trainvaltest_foggy.zip**.

2. Prepare the data using the scripts in 'prepare_data/prepare_data.m'.

3. Train the Domain Adaptive Faster R-CNN:
    ```Shell
    cd $FRCN_ROOT
    ./tools/train_net.py --gpu {GPU_ID} --solver models/da_faster_rcnn/solver.prototxt --weights data/imagenet_models/VGG16.v2.caffemodel --imdb voc_2007_trainval --iters  70000  --cfg  models/da_faster_rcnn/faster_rcnn_end2end.yml
    
3. Test the trained model:
    ```Shell
    cd $FRCN_ROOT
    ./tools/test_net.py --gpu {GPU_ID} --def models/da_faster_rcnn/test.prototxt --net output/faster_rcnn_end2end/voc_2007_trainval/vgg16_da_faster_rcnn_iter_70000.caffemodel --imdb voc_2007_test --cfg models/da_faster_rcnn/faster_rcnn_end2end.yml

### Other Implementation
[Detectron-DA-Faster-RCNN](https://github.com/krumo/Detectron-DA-Faster-RCNN) in Caffe2(Detectron)
