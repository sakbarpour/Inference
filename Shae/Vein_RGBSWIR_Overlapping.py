import argparse
import sys
import os
import json
import glob
import pandas as pd
import traceback
import numpy as np
import cv2
import tensorflow as tf
from os.path import exists
from util.unetvein import binary_unet
from util.SWIROverlap import Vein_SWIR_Threshold
from MyUtils.MpiHelper import BaseRunner

# TensorFlow uses five levels of logging: DEBUG, INFO, WARN, ERROR, and FATAL.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # show only fatal messages
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This line disables the GPU
tf.get_logger().setLevel('ERROR')

class Runner(BaseRunner):
    def __init__(self, args, comRank, hostName, **kwargs):
        BaseRunner.__init__(self, comRank, hostName)
        BaseRunner.copy_args(self, args)
        assert (type(args) == argparse.Namespace)


        # Check the parameters
        assert exists(self.processedDir), "Main directory not found, check processedDir"
        assert exists(self.processedDir + os.sep+ self.core_mask_overlays), "Core masks directory not found, check core_mask_overlays"
        assert exists(self.processedDir + os.sep+ "rgb-tubes" ), "rgb-tubes directory not found check drillhole folder"
        assert exists(self.vein_weights), "Unet file not found, check vein_weights"
        assert exists(self.processedDir+ os.sep + "wavelengthfeatures"+ os.sep + self.SWIR_dir), "SWIR directory"
        
        self.SWIR_dir =self.processedDir+ os.sep + "wavelengthfeatures"+ os.sep + self.SWIR_dir 
        self.core_mask_dir = self.processedDir + os.sep + self.core_mask_overlays
        self.rgb_dir = self.processedDir + os.sep + "rgb-tubes"
        #self.SWIR_dir=self.processedDir+ os.sep + "wavelengthfeatures"+ os.sep + SWIR_dir


        if self.comRank==0 and not os.path.exists(self.processedDir +os.sep+ self.out_name):
            os.mkdir(self.processedDir +os.sep+ self.out_name)

        # some parameters passed from training
        self.num_classes = 1
        self.input_shape = (256, 256,3)


        # load model
        self.graph1, self.model, self.batch_size = self.load_model(self.input_shape, self.num_classes, self.vein_weights,
                                                         self.batch_size)


    def vein_inference(self, model, graph1, tube , batch_size):
        """ Get a tube as a task, apply U-Net to generate veins overlay"""
        image_path= glob.glob(self.processedDir + os.sep + "rgb-tubes" +os.sep +f"*{tube}*.png")[0]
        image = cv2.imread(image_path)
        img_pix_fn = image_path.replace(".png", ".txt")

        if not os.path.exists(img_pix_fn):
            print(f"{tube} image pixel file not found", flush=True)
        else:
            with open(img_pix_fn, 'r') as reader:
                pix = int(reader.read().split(",")[2])
            image = image[:pix, :, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        # Make overlay
        vein_mask_RGB = np.zeros((image_height, image_width))
        px1, px2 = int(self.border_crop * image_width), int((1. - self.border_crop) * image_width)
        patch_size = np.abs(px2 - px1)
        stride = patch_size // 2  # Define the overlap/stride
        patches = []
        for j in range(0, image_height - patch_size, stride):
            # Calculate patch boundaries with overlap
            py1, py2 = j, j + patch_size
            for i in range(0, image_width - patch_size, stride):
                px1, px2 = i, i + patch_size

                patch = image[py1:py2, px1:px2, :]
                patch = cv2.resize(patch, self.input_shape[0:2])
                patch = np.array(patch, dtype=np.float32) / 255.
                patches.append(patch)

        # Batch processing
        with graph1.as_default():
            pred_batch = model.predict(np.array(patches), batch_size=batch_size)

        # Binarize patches
        threshold_mask = pred_batch > self.min_confidence
        pred_batch[threshold_mask] = 1
        pred_batch[~threshold_mask] = 0

        # Assign predictions to the vein_mask
        patch_index = 0
        for j in range(0, image_height - patch_size, stride):
            # Calculate patch boundaries with overlap
            py1, py2 = j, j + patch_size
            for i in range(0, image_width - patch_size, stride):
                px1, px2 = i, i + patch_size

                pred_square = pred_batch[patch_index].squeeze()
                vein_mask_RGB[py1:py2, px1:px2] += cv2.resize(pred_square, (patch_size, patch_size))
                patch_index += 1

        # Threshold the final vein_mask (if needed)
        vein_mask_RGB = (vein_mask_RGB > 0.4).astype(np.uint8) * 255
        vein_mask_RGB = cv2.cvtColor(vein_mask_RGB, cv2.COLOR_GRAY2RGB)

        # Apply morphological operations to remove small components
        kernel = np.ones((1,1), np.uint8)
        vein_mask_RGB = cv2.morphologyEx(vein_mask_RGB, cv2.MORPH_OPEN, kernel)
        
        # Label connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vein_mask_RGB[:, :, 0], connectivity=8)
        # Filter out small components
        min_component_size = 10  # Define your minimum component size threshold
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < min_component_size:
                vein_mask_RGB[labels == label] = 0

        vein_mask_RGB = (vein_mask_RGB >= 128).astype(np.uint8) * 255

        # Initialize vein_mask_filteredby_SWIR as vein_mask_RGB
        vein_mask_filteredby_SWIR = vein_mask_RGB.copy()

        # Apply SWIR Thresholding on the binary mask generated by UNet mode using RGB images
        tubeID = tube.split('_')[-1]  # tube is `DH_tubeID`
        if self.SWIRThresholding_enabled:
            vein_mask_filteredby_SWIR = Vein_SWIR_Threshold(self.SWIR_dir, vein_mask_RGB, tubeID, min_area=100)
            vein_mask_filteredby_SWIR = vein_mask_filteredby_SWIR.astype('uint8')
            vein_mask_filteredby_SWIR = cv2.resize(vein_mask_filteredby_SWIR, (image_width, image_height))
            

        # Apply core masker
        if self.coremasker_enabled:
            core_mask_path = glob.glob(self.core_mask_dir + os.sep + f"*{tube}*.png")
            if len(core_mask_path) > 0:  # Apply core mask
                core_mask = cv2.imread(core_mask_path[0])
                core_mask = cv2.resize(core_mask, (image_width, image_height))
                core_mask = core_mask[:, :, 0] != 0
                core_mask = cv2.dilate(core_mask.astype(np.uint8), np.ones((5, 5)))
                core_mask = core_mask == 1
                # Modify only where the core_mask is True
                vein_mask_filteredby_SWIR[core_mask] = [89, 89, 89]  # Gray for 'not-core' pixels

        # Ensure that vein_mask_filteredby_SWIR is not None even if SWIR or core masking is not enabled
        if vein_mask_filteredby_SWIR is None:
            vein_mask_filteredby_SWIR = vein_mask_RGB
            
        # Save the results with a fixed file name
        vein_mask_RGBandSWIR = cv2.cvtColor(vein_mask_filteredby_SWIR, cv2.COLOR_RGB2BGR)

        # Save the result, even if it contains only zero values
        cv2.imwrite(self.processedDir + os.sep + self.out_name + os.sep + f"{tube}s1_{self.out_name}.png", vein_mask_RGBandSWIR)
        print(50 * "--", "\n", tube, "\n", flush=True)
        return True



    def load_model(self, input_shape, num_classes, vein_weights, batch_size):
        # Load and initialize model
        tf.keras.backend.clear_session()
        graph1 = tf.compat.v1.Graph()

        with graph1.as_default():
            if tf.test.is_gpu_available() and False:
                with tf.device('/CPU:0'):  # Specify the GPU device
                    model = binary_unet(input_shape, num_classes, activation='sigmoid')
                    model.load_weights(vein_weights)
                    batch_size = 1
            else:
                with tf.device('/CPU:0'):
                    
                    print("WARNING: GPU NOT FOUND, running on CPU will be slower.", flush=True)
                    model = binary_unet(input_shape, num_classes, activation='sigmoid')
                    model.load_weights(vein_weights)
                    print(30*"---", flush=True)
                    print("---* Unet is loaded on CPU *----", flush = True)
                    print(30*"---", flush=True)

                    batch_size = 1
        return graph1, model, batch_size
   
    
    def find_basename(self, file_path):
        return os.path.basename(file_path).split(".")[0]


    def create_output_folder(self):
        out_dir = self.processedDir + os.sep + self.out_name
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            print(f"Created output folder: {out_dir}")
        else:
            print(f"Output folder already exists: {out_dir}")

    
    def getTasks(self):
        # To DO: USE BOX UTILS
        tasks = [tube.split('.png')[0] for tube in os.listdir(os.path.join(self.processedDir, "rgb-tubes")) if tube.endswith(".png")]
        # Filter tasks based on the box if it's provided
        if isinstance(self.box, list):
            tasks = [task for task in tasks if any(task_id in task for task_id in self.box)]

        # Remove existing tasks if skip_existing is true
        if self.skip_existing:
            existing_files = [fn.split('s1_')[0] for fn in os.listdir(os.path.join(self.processedDir, self.out_name)) if fn.endswith('.png')]
            tasks = [task for task in tasks if task not in existing_files]
        print("NUMBER TASKS:", len(tasks), flush=True)
        return tasks


    def run(self, task):
        the_tube = task
        self.create_output_folder()


        out_dir = self.processedDir + os.sep + self.out_name
        self.flagFilename = f".veins_{the_tube}"

        self.removeErrorFlag(out_dir)
        self.flag(out_dir)
        try:
            print(30*"---", flush=True)
            print(f"Processing {the_tube}", flush=True)
            print("----------*  Done *-----------", flush=True)
            self.vein_inference(self.model, self.graph1, the_tube, self.batch_size)
        except Exception as e:
            self.processException(e, traceback.format_exc(), sys.exc_info()[2], out_dir)
        finally:
            self.unflag()
                

        return True



    @staticmethod
    def getParser():
            parser = BaseRunner.getParser()
            parser.add_argument('--skip_existing', help='skip_existing if already processed', type=bool, default=False)
            parser.add_argument('--SWIRThresholding_enabled', help='Allow SWIR Thresholding', type=bool, default=False)
            parser.add_argument('--SWIR_dir', help='directory with SWIR pickles', type=str, default="numeric")
            parser.add_argument('--processedDir', help='directory with processed folders', type=str)
            parser.add_argument('--out_name', help='Name for veins folder and overlays', type=str, default='VeinQ')
            parser.add_argument('--box',help='tubes to run, e.g. c000b028c or all to check all. Split with , if multiple boxes', type=str,default= "all")
            parser.add_argument('--vein_weights', help='Path to .h5 file unetWeights of the veins U-Net model', type=str,default= r"\\192.168.213.101\mpi\Data\Vein\unet\w_vein_v1_256_Batch8_Eboosted.h5")
            parser.add_argument('--batch_size ', help='inference batch size', type=int,default=5)
            parser.add_argument('--batch_size', help='batch size', type=int, default=8)
            parser.add_argument('--min_confidence', help='minimum confidence probability to be considered as a vein', type=float, default=0.5)
            parser.add_argument('--core_mask_overlays', help='directory with core mask overlays', type=str,default='coremaskoverlays')
            parser.add_argument('--coremasker_enabled', help='Alllow coremasker overlays', type=bool, default=True)
            parser.add_argument('--border_crop', help='crop from top and bottom of the core', type=float, default=0.01)



            return parser
