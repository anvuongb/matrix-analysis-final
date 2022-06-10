from tritonclient.grpc import InferenceServerClient, InferInput
from google.protobuf.json_format import MessageToDict

import logging, logging.config
import grpc
import traceback

import numpy as np

class ServingTRTISgRPCBaseClass(object):
    def _get_logger(self, class_name=None):
        if class_name is None:
            class_name = self.__class__.__name__
        self.logger = logging.getLogger(class_name)
        
    def __init__(self, trtis_serving_host, trtis_serving_port, model_name, inputs_name, outputs_name, child_class_name="AGE", timeout=10.0, ssl_enable=False, model_version=""):
        # init logger with correct class name
        self._get_logger()

        # init vars
        self.trtis_service = "{}:{}".format(trtis_serving_host, trtis_serving_port)
        self.model_name = model_name
        self.inputs_name = inputs_name
        self.outputs_name = outputs_name
        self.timeout = timeout
        self.ssl_enable = ssl_enable
        self.child_class_name = child_class_name
        self.model_version = model_version

        self.logger.info("[INIT-{}] Initializing connection to {}, model {} - version {}, SSL {}, timeout {}s".format(self.child_class_name, self.trtis_service, self.model_name, self.model_version, self.ssl_enable, self.timeout))
        self.client = self.__init_client(ssl_enable=self.ssl_enable)
        
        self.logger.info("[INIT-{}] Checking if model is available to take requests".format(self.child_class_name))
        self.model_state, self.__model_state_exc = self.__check_connection()
        
        if self.model_state.lower() == 'available':
            self.logger.info('Model {} from {} is AVAILABLE for serving'.format(self.model_name, self.trtis_service))
        else:
            self.logger.error('Model {} from {} is NOT AVAILABLE. Exception {}'.format(self.model_name, self.trtis_service, self.__model_state_exc))
            raise Exception('Model {} from {} is NOT AVAILABLE'.format(self.model_name, self.trtis_service))
            
        self.logger.info("[INIT-{}] Checking if provided inputs_name, outputs_name match model's metadata".format(self.child_class_name))
        self.server_model_dict, self.metadata_state, self.__metadata_exc = self.__get_model_metadata()
        if self.metadata_state.lower() == 'matched':
            self.logger.info('Provided inputs_name, outputs_name MATCH model\'s metadata')
        else:
            self.logger.error('Provided inputs_name, outputs_name DO NOT MATCH model\'s metadata. Exception {}'.format(self.__metadata_exc))
            raise Exception('Model {} from {} is available, but its provided inputs/outputs {}:{} DO NOT MATCH what is on server:\n {}'.format(self.model_name, self.trtis_service,
                                                                                                                                                self.inputs_name, self.outputs_name,
                                                                                                                                                self.server_model_dict))
        self.logger.info('[INIT-{}] Acquiring desired input tensor dimensions'.format(self.child_class_name))
        try: 
            self.expected_input_size, self.expected_output_size = self.__acquire_dims()
            self.logger.info('Expected input dims = {}'.format(self.expected_input_size))
            self.logger.info('Expected output dims = {}'.format(self.expected_output_size))
        except Exception as e:
            self.expected_input_size = None
            self.logger.warning('Failed to acquired tensor dims, may cause unexpected problem. Exception {}'.format(e))
        self.logger.info('[INIT-{}] Init and check run successfully, READY TO SERVE'.format(self.child_class_name))

    def __init_client(self, ssl_enable=False):
        client = InferenceServerClient(self.trtis_service, ssl=ssl_enable)
        return client

    def _predict_infer_context(self, image, batch=False):
        if len(self.expected_output_size) > 1:
            raise NotImplementedError('there are multiple outputs, this function needs to be overwrite in that case')
        if batch is True:
            raise NotImplementedError('_postprocessing batch function not implemented')

        infer_input = self._make_infer_input(datatype='FP32')
        image = image[np.newaxis,:]
        infer_input.set_data_from_numpy(image.astype('float32'))

        predictions = self.client.infer(self.model_name, [infer_input])
        predictions_np = predictions.as_numpy(self.outputs_name)
        
        return predictions_np
    
    def _make_infer_input(self, datatype='FP32', batch_size=1):
        '''
        make input that comply with TRT server, only works with single input
        multiple inputs model will need to re-implement this function
        '''
        if len(self.expected_input_size) > 1:
            raise NotImplementedError('there are multiple inputs, this function needs to be overwrite in that case')
        
        inputs = InferInput(self.inputs_name, [batch_size] + list(self.expected_input_size[0][1:]), datatype=datatype)
        return inputs
    
    def __check_connection(self):
        '''
        get state_ctx
        return 'AVAILABLE' if model exists
        '''
        state = 'AVAILABLE'
        e_ret = ''
        
        # Check if server ready
        try:
            if not self.client.is_server_live():
                state = 'FAILED'
                e_ret = 'TRT server is not live'
                return state, e_ret

            if not self.client.is_server_ready():
                state = 'FAILED'
                e_ret = 'TRT server is not ready'
                return state, e_ret
        except Exception as e:
            state = 'FAILED'
            e_ret = e
            return state, e_ret
        
        # Check if model available
        try: 
            model_status = self.client.get_model_metadata(self.model_name, self.model_version)
        except Exception as e:
            state = 'FAILED'
            e_ret = 'model {} - version {} not found in TRT server'.format(self.model_name, self.model_version)
            return state, e_ret
        
        # Check if model ready
        try:
            model_ready_state = self.client.is_model_ready(self.model_name, self.model_version)
        except Exception as e:
            state = 'FAILED'
            e_ret = e
            return state, e_ret
        
        if model_ready_state is False:
            state = 'FAILED'
            e_ret = 'model {} - version {} not ready in TRT server'.format(self.model_name, self.model_version)
        
        return state, e_ret

    def __get_model_metadata(self):
        '''
        generate a model_metadata_request from init config
        return 'MATCH' if signature matches
        '''
        
        e_ret = ''
        state = 'MATCHED'
        server_model_dict = {}
        
        try: 
            server_model_metadata = self.client.get_model_metadata(self.model_name, self.model_version)
            
            server_model_inputs = [c.name for c in server_model_metadata.inputs]
            server_model_outputs = [c.name for c in server_model_metadata.outputs]
            
            server_model_dict = {
                'inputs_keys': server_model_inputs,
                'outputs_keys': server_model_outputs
            }
                                                                                                                                               
        except Exception as e:
            state = 'NOT MATCHED'
            e_ret = e
            return server_model_dict, state, e_ret
        
        if type(self.inputs_name) is list:
            for iname in self.inputs_name:
                if iname not in server_model_inputs:
                    e_ret = 'inputs_name {} not in server_model_inputs, model returns:\n{}'.format(iname, server_model_inputs)
                    state = 'NOT MATCHED'
                    return server_model_dict, state, e_ret
        else:
            if self.inputs_name not in server_model_inputs:
                e_ret = 'inputs_name {} not in server_model_inputs, model returns:\n{}'.format(self.inputs_name, server_model_inputs)
                state = 'NOT MATCHED'
                return server_model_dict, state, e_ret

        if type(self.outputs_name) is list:
            for oname in self.outputs_name:
                if oname not in server_model_outputs:
                    e_ret = 'outputs_name {} not in server_model_outputs, model returns:\n{}'.format(oname, server_model_outputs)
                    state = 'NOT MATCHED'
                    return server_model_dict, state, e_ret
        else:
            if self.outputs_name not in server_model_outputs:
                e_ret = 'outputs_name {} not in server_model_outputs, model returns:\n{}'.format(self.outputs_name, server_model_outputs)
                state = 'NOT MATCHED'
                return server_model_dict, state, e_ret
        
        return server_model_dict, state, e_ret
    
    def __acquire_dims(self):
        '''get input dimension from serving model of dim (-1, W, H, C)'''
        server_model_metadata = self.client.get_model_metadata(self.model_name, self.model_version)
        
        ret_tup_in = [tuple(z.shape) for z in server_model_metadata.inputs]
        
        ret_tup_out = [tuple(z.shape) for z in server_model_metadata.outputs]
            
        return ret_tup_in, ret_tup_out

    def __init_grpc_connection(self, ssl_enable=False):
        raise Exception('TRTIS 20.08 no longer needs this function')

    @staticmethod
    def _make_grpc_request(input_image, model_name, signature_name, inputs_name):
        raise Exception('TRTIS does not support _make_grpc_request()')

    def _preprocessing(self, input_images, batch=False):
        '''
        Function to perform preprocessing, must be implemented by children class
        '''
        raise NotImplementedError('_preprocessing function not implemented')
    
    def _postprocessing(self, predictions, batch=False):
        '''
        Function to perform postprocessing, must be implemented by children class
        '''
        raise NotImplementedError('_postprocessing function not implemented')

    def predict(self, input_image):
        '''
        Make single call prediction with preprocessing and post processing
        '''
        raise NotImplementedError('predict function not implemented')

    def predict_batch(self, input_images):
        '''
        Make batch prediction with preprocessing and post processing
        '''
        raise NotImplementedError('predict_batch function not implemented')