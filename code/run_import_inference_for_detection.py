import os
import sys
import argparse
import cv2
import yaml
import shutil
import PIL
import numpy as np
import onnx
import glob
import shutil
import edgeai_benchmark

def main(target_device, run_type):

    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] in ('scripts', 'tutorials'):
        os.chdir('../')
    #

    #########################################################################
    assert ('TIDL_TOOLS_PATH' in os.environ and 'LD_LIBRARY_PATH' in os.environ), \
        "Check the environment variables, TIDL_TOOLS_PATH, LD_LIBRARY_PATH"
    print("TIDL_TOOLS_PATH=", os.environ['TIDL_TOOLS_PATH'])
    print("LD_LIBRARY_PATH=", os.environ['LD_LIBRARY_PATH'])
    print("TARGET_SOC=", target_device)
    print(f"INFO: current dir is: {os.getcwd()}")

    if not os.path.exists(os.environ['TIDL_TOOLS_PATH']):
        print(f"ERROR: TIDL_TOOLS_PATH: {os.environ['TIDL_TOOLS_PATH']} not found")
    else:
        print(f'INFO: TIDL_TOOLS_PATH: {os.listdir(os.environ["TIDL_TOOLS_PATH"])}')
    #

    #########################################################################
    num_frames = 1
    calibration_frames = 5 #12
    calibration_iterations = 5 #12

    #########################################################################
    modelartifacts_tempdir_name = os.path.abspath('./work_dirs_custom')
    modelartifacts_custom = os.path.join(modelartifacts_tempdir_name, 'modelartifacts')
    print(f'INFO: clearing modelartifacts folder: {modelartifacts_custom}')
    if run_type=='IMPORT' and os.path.exists(modelartifacts_custom):
        shutil.rmtree(modelartifacts_custom, ignore_errors=True)
    #

    #########################################################################
    settings = edgeai_benchmark.core.ConfigRuntimeOptions('./settings_import_on_pc.yaml',
                    target_device=target_device,
                    target_machine='pc',
                    modelartifacts_path=modelartifacts_custom,
                    model_selection=None, model_shortlist=None,
                    calibration_frames=calibration_frames,
                    calibration_iterations=calibration_iterations,
                    num_frames=1)

    #########################################################################
    # download dataset if it doesn't exist
    dataset_name='own_dataset_for_detection'
    if not os.path.exists(f'{settings.datasets_path}/{dataset_name}'):
        print(f'INFO: downloading the dataset - {dataset_name}')
        edgeai_benchmark.interfaces.run_download_dataset(settings, dataset_name=dataset_name)
    else:
        print(f'INFO: dataset exists, will reuse - {dataset_name}')
    

    #########################################################################
    # give the path to your model here
    # model_path=f'{settings.models_path}/vision/classification/imagenet1k/yolov8m-cls.onnx'
    # model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx'
    # model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_mobilenetp5_lite_320x320_20230404_model.onnx'),
    model_path = os.path.abspath('./tutorials/merged_model_3.onnx')
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    run_dir = os.path.join(work_dir, os.path.splitext(os.path.basename(model_path))[0])
    shutil.rmtree(run_dir, ignore_errors=True)

    model_file = os.path.join(run_dir, 'model', os.path.basename(model_path))
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    shutil.copy2(model_path, model_file)
    # onnx.shape_inference.infer_shapes_path(model_file, model_file)
    print(f'INFO: model_file - {model_file}')

    artifacts_folder = os.path.join(run_dir, 'artifacts')
    os.makedirs(artifacts_folder, exist_ok=True)
    print(f'INFO: artifacts_folder - {artifacts_folder}')


    #########################################################################
    runtime_options = settings.get_runtime_options()
    print(f'INFO: runtime_options - {runtime_options}')


    #########################################################################
    def preprocess_input(input_img_file):
        width = 640
        height = 640
        input_mean=[123.675, 116.28, 103.53]
        input_scale=[0.017125, 0.017507, 0.017429]
        input_img = PIL.Image.open(input_img_file).convert("RGB").resize((width, height), PIL.Image.BILINEAR)
        input_data = np.expand_dims(input_img, axis=0)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        normalized_data = np.zeros(input_data.shape, dtype=np.float32)
        for mean, scale, ch in zip(input_mean, input_scale, range(input_data.shape[1])):
            normalized_data[:, ch, :, :] = (input_data[:, ch, :, :] - mean) * scale
        return normalized_data


    #########################################################################
    def run_import():
        dataset_path = f'{settings.datasets_path}/{dataset_name}'        
        calib_dataset = glob.glob(f'{dataset_path}/datasets/*.*') 
        onnxruntime_wrapper = edgeai_benchmark.core.ONNXRuntimeWrapper(
                runtime_options=runtime_options,
                model_file=model_file,
                artifacts_folder=artifacts_folder,
                tidl_tools_path=os.environ['TIDL_TOOLS_PATH'],
                tidl_offload=True)

        for input_index in range(calibration_frames):
            input_data = preprocess_input(calib_dataset[input_index])
            onnxruntime_wrapper.run_import(input_data)
        print(f'INFO: model import done')
    def run_inference():
        class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            29: 'frisbee',
            30: 'skis',
            31: 'snowboard',
            32: 'sports ball',
            33: 'kite',
            34: 'baseball bat',
            35: 'baseball glove',
            36: 'skateboard',
            37: 'surfboard',
            38: 'tennis racket',
            39: 'bottle',
            40: 'wine glass',
            41: 'cup',
            42: 'fork',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            47: 'apple',
            48: 'sandwich',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',
            53: 'pizza',
            54: 'donut',
            55: 'cake',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }
        dataset_path = f'{settings.datasets_path}/{dataset_name}'        
        val_dataset = glob.glob(f'{dataset_path}/datasets/*.*') 
        input_dir = os.path.join(dataset_path, 'input')
        output_dir = os.path.join(dataset_path, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        onnxruntime_wrapper = edgeai_benchmark.core.ONNXRuntimeWrapper(
            runtime_options=runtime_options,
            model_file=model_file,
            artifacts_folder=artifacts_folder,
            tidl_tools_path=os.environ['TIDL_TOOLS_PATH'],
            tidl_offload=True
        )
        conf_threshold = 0.5
        nms_threshold = 0.45
        answer = set()
        for input_index in range(num_frames):
            input_image_path = val_dataset[input_index]
            print(input_image_path)
            shutil.copy(input_image_path, os.path.join(input_dir, f'input_{input_index}.jpg'))
            image = cv2.imread(input_image_path)
            image_resized = cv2.resize(image, (640, 640))
            input_image = image_resized[:, :, ::-1].transpose(2, 0, 1)
            input_image = np.ascontiguousarray(input_image).astype(np.float32) / 255.0
            input_data = input_image[np.newaxis, :]
            outputs = onnxruntime_wrapper.run_inference(input_data)[0]
            outputs = np.squeeze(outputs).transpose(1, 0)
            boxes, confidences, class_ids = [], [], []
            for output in outputs:
                scores = output[4:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    x, y, w, h = output[0:4]
                    x1 = (x - w / 2) * image.shape[1] / 640
                    y1 = (y - h / 2) * image.shape[0] / 640
                    x2 = (x + w / 2) * image.shape[1] / 640
                    y2 = (y + h / 2) * image.shape[0] / 640
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cls = class_ids[i]
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(image, f"{class_names.get(cls, 'Unknown')}: {confidences[i]:.2f}", (int(x), max(int(y)-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                answer.add(class_names.get(cls))
            output_path = os.path.join(output_dir, f'output_{input_index}.jpg')
            saved = cv2.imwrite(output_path, image)
            print(f'Saved: {output_path} - Success: {saved}')
        print('INFO: detection model inference done')
        print(answer)

    #########################################################################
    # import and inference can be run in single call if separat3 process is used for them
    # otehrwise one would have to choose between either import or inference in one call of this script.,
    if run_type == "IMPORT":
        run_import()
    elif run_type == "INFERENCE":
        run_inference()
    else:
        assert False, f"ERROR: please set parallel_processes>1 or set run_type to IMPORT or INFERENCE - got {run_type}"


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, choices=('IMPORT', 'INFERENCE'))
    parser.add_argument('--target_device', type=str, default='AM68A')
    return parser

if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = get_arg_parser()
    args = parser.parse_args()

    main(args.target_device, args.run_type)