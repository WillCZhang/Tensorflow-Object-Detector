# Object Detector

The detector is built based on tutorials from [Training Custom Object Detector](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) and [How to train your own Object Detector with TensorFlow’s Object Detector API](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9). Both tutorials are awesome, but configuring TensorFlow Object Detector is just too painful. So I wrote this object detector for everyone who enjoys seeing results rather than fighting with tools. It's still recommanded to follow the tutorials to learn the process, but all you really need to run this detector is a bunch of images and [docker](https://www.docker.com/).

Note: the detector is mainly designed for people who are starting from only having images. If you already have a dataset created, you may need to change some scripts (but that process should be pretty easy).

## Workflow

This is the workflow for you to start with nothing and end with a trained model. Please refer to tutorials if any terminalogy looks wired to you.

1. __Install Detector__ To install the detector, you will need docker installed in your system. Once you have done that, simply run `./build.sh`, it will create a docker image tagged `object-detector`.
2. __Label Images__ You will need a bunch of images for training, and for each image, you need to add labels to it to indicate an object (i.e. having an XML file associated with the JPG file). I used [labelImg](https://github.com/tzutalin/labelImg) for image labeling, and it's good enough for me, just remember to save it!!!
3. __Create Dataset__ As mentioned in the tutorials, you will need TFRecord format for you data.
   1. After you labeled the images, open the [configuration](./configuration) file, change the `LABELED_IMAGE_PATH` variable to the path of labeled images.
   2. The `CLASSES` variable in the configuration file specifies the classes (labels) you want to be detected, split classes by comma.
   3. You will also need a path to hold the dataset (consider creating a new folder for it). You can specify that path by modifying `DATA_PATH` variable in the configuration file.
   4. Now you can run `./create-dataset.sh`, it will create three folders under the `<DATA_PATH>` you specified.
      * `<DATA_PATH>/images` for your labeled images
      * `<DATA_PATH>/training` for TF_Record, label_map, and training images
      * `<DATA_PATH>/testing`for TF_Record, label_map, and testing images
4. __Prepare Pre-trained Model__ As mentioned in the tutorials, you will need a pre-trained model to start training.
   1. Before you start, you need to change `MODEL_PATH` variable in the configuration file to a folder that will be holding your models. Consider creating a new folder for it.
   2. Download a pre-trained model from [TensorFlow’s detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models), then unzip it. For this project, you __must__ rename the model folder and save it as `<MODEL_PATH>/model`.
   3. Because the pre-trained model contains checkpoints that exceeded max_step (i.e. the model will not run), simply remove `checkpoint` file inside the model folder.
   4. Configure a `pipeline.config` file corresponding to the model path and data path. You __must__ also save the config file to `<MODEL_PATH>/pipeline.config`. For this detector, `DATA_PATH` is mounted as `/data/` and `MODEL_PATH` is mounted as `/model/`, so anytime you want to refer to something inside these paths, you should use to their mounted paths. For example, you should configure it as the following:

       ```
           fine_tune_checkpoint`: "/model/model/model.ckpt"
           ...
           tf_record_input_reader { # for all occasion
               input_path: "/data/<training or testing>/tf.record"
           }
           label_map_path: "/data/<training or testing>/label_map.pbtxt"
       ```

       Note that `tf.record` and `label_map.pbtxt` are defined in `detector/prepare_data.py`. Please refer to [configuring-a-training-pipeline](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configuring-a-training-pipeline) for more details.
5. __Start__ Now we are ready! Run `./start.sh`, you will enter the detector shell, type in `train`, press enter, enjoy training! You can also open browser `http://localhost:6006/` to observe the training status!
