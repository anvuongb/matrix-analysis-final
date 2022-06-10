# python basic libs
import time, traceback, logging, logging.config
import numpy as np
import cv2

# trt libs
from tritongrpcclient import InferenceServerClient, InferInput

# this libs
from .ServingTRTISgRPCBaseClass import ServingTRTISgRPCBaseClass

class ArcFace(ServingTRTISgRPCBaseClass):
    '''
    selfie image embedding extractor for face server
    Wrap around ServingTFgRPCBaseClass and BaseFaceDetector
    '''

    def __init__(self, trt_serving_host, trt_serving_port, model_name='arc_face',
                 inputs_name='data',
                 outputs_name='fc1', child_class_name="ARCFACE-PAIR",
                 timeout=60.0, ssl_enable=False, model_version=""):
        # if none provided, pull from config

        self.match_level_thresholds = [0,
                                       1.20,  # 99% confidently matched
                                       1.40,  # matched with low confidence
                                       1.50  # unmatched with low confidence
                                       ]

        self.image_size = (112,112)

        super().__init__(trt_serving_host, trt_serving_port, model_name,
                         inputs_name, outputs_name, child_class_name, timeout, ssl_enable, model_version)

    def predict(self, input_image_list):
        '''
        extract embedding from selfie image
        '''

        preprocessed_images = self._preprocessing(np.stack(input_image_list))
        predictions = self._predict_infer_context(preprocessed_images)
        norms = np.linalg.norm(predictions, axis=1).reshape((-1,1))
        predictions = predictions/norms
        return predictions

    def _preprocessing(self, input_imgs, batch=False):
        if batch is True:
            raise NotImplementedError('_preprocessing batch function not implemented')
        else:
            mean = 0.0
            std = 1.0
            images_blob = cv2.dnn.blobFromImages(images=input_imgs,scalefactor=1.0/std,size=self.image_size,
                                                 mean=(mean,mean,mean),swapRB=True)
            return images_blob

    def _predict_infer_context(self, images, batch=False):

        infer_input_images = self._make_infer_input(batch_size=len(images))

        images_reshape = np.array(images)

        infer_input_images.set_data_from_numpy(images_reshape)

        predictions = self.client.infer(self.model_name, [infer_input_images])
        predictions_np = predictions.as_numpy(self.outputs_name)
        
        return predictions_np

    @staticmethod
    def get_confidence_score(sim_score, min_sim=0.4, max_sim=2.83, min_confidence=0.01):
        sim_score = min(max(sim_score, min_sim), max_sim)
        confidence_score = (-sim_score + max_sim) * (1 - min_confidence) / (max_sim - min_sim) + min_confidence
        return confidence_score
