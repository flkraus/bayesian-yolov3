# bayesian-yolov3
YOLOv3 object detection architecture with uncertainty estimation in TensorFlow.

Accompanying code for: https://arxiv.org/abs/1905.10296 (IEEE Xplore: https://ieeexplore.ieee.org/document/8917494/)

### Citation
If you find this work useful in your research, please consider citing:

    @inproceedings{kraus_uncertainty_2019,
      address = {Auckland, New Zealand},
      title = {Uncertainty {Estimation} in {One}-{Stage} {Object} {Detection}},
      url = {https://ieeexplore.ieee.org/document/8917494/},
      doi = {10.1109/ITSC.2019.8917494},
      booktitle = {2019 {IEEE} {Intelligent} {Transportation} {Systems} {Conference} ({ITSC})},
      publisher = {IEEE},
      author = {Kraus, Florian and Dietmayer, Klaus},
      month = oct,
      year = {2019},
      pages = {53--60}
    }

### Notes
- Training examples with documentation:
  - pretraining.py - pretraining for models with uncertainty estimation
  - uncertainty_training.py - training for models with uncertainty estimation,
    use checkpoints produced by pretraining.py as a starting point.
  - yolov3_training.py  - standard yolov3 without any uncertainty estimation
  - look for "edit" comments
- Forward passes:
  - detect.py (processes a list of images)
  - inference_*.py scripts. They process tfrecord files and produce ECP (euro city persons) formated json files.
  - Note NMS yields up to 1000 boxes, might be slow. Change the "nms" functions if you want better performance.
  - Current NMS implementation ignores classes.
     Example code used in the paper is given as comments (only works for two classes).
  - look for "edit" comments
- tfrecords format:
  - same as for the tensorflow object detection API,
     however we also support tfrecords files where the label ids start at 0 instead of 1.
     This is controlled by setting "implicit_background_class" to True (start at 1) or False (start at 0).
  - example script to create tfrecordsfile is provided (create_tf_records_citypersons.py)
- Pretrained yolov3 weights:
  - you need to download the "darknet53.conv.74" from the original yolov3 site (pjreddie).
- Most things you can change should be marked with an "edit" comment.
