# common dependencies
import os
import warnings
import logging
from typing import Any, Dict, List, Union, Optional
# No need to refer to readline functions. importing it, then it will take care of the rest.
import readline
import signal
import cv2
import re

# this has to be set before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# pylint: disable=wrong-import-position

# 3rd party dependencies
import numpy as np
import pandas as pd
import tensorflow as tf

# package dependencies
from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.modules import (
    modeling,
    representation,
    verification,
    recognition,
    demography,
    detection,
    streaming,
    preprocessing,
)
from deepface import __version__

logger = Logger()

# -----------------------------------
# configurations for dependencies

# users should install tf_keras package if they are using tf 2.16 or later versions
package_utils.validate_for_keras3()

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
# -----------------------------------

# create required folders if necessary to store model weights
folder_utils.initialize_folder()


def build_model(model_name: str, task: str = "facial_recognition") -> Any:
    """
    This function builds a pre-trained model
    Args:
        model_name (str): model identifier
            - VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace, GhostFaceNet for face recognition
            - Age, Gender, Emotion, Race for facial attributes
            - opencv, mtcnn, ssd, dlib, retinaface, mediapipe, yolov8, yunet,
                fastmtcnn or centerface for face detectors
            - Fasnet for spoofing
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
            default is facial_recognition
    Returns:
        built_model
    """
    return modeling.build_model(task=task, model_name=model_name)


def verify(
    img1_path: Union[str, np.ndarray, List[float]],
    img2_path: Union[str, np.ndarray, List[float]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
) -> Dict[str, Any]:
    """
    Verify if an image pair represents the same person or different persons.
    Args:
        img1_path (str or np.ndarray or List[float]): Path to the first image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images
            or pre-calculated embeddings.

        img2_path (str or np.ndarray or List[float]): Path to the second image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images
            or pre-calculated embeddings.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        result (dict): A dictionary containing verification results with following keys.

        - 'verified' (bool): Indicates whether the images represent the same person (True)
            or different persons (False).

        - 'distance' (float): The distance measure between the face vectors.
            A lower distance indicates higher similarity.

        - 'threshold' (float): The maximum threshold used for verification.
            If the distance is below this threshold, the images are considered a match.

        - 'model' (str): The chosen face recognition model.

        - 'distance_metric' (str): The chosen similarity metric for measuring distances.

        - 'facial_areas' (dict): Rectangular regions of interest for faces in both images.
            - 'img1': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the first image.
            - 'img2': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the second image.

        - 'time' (float): Time taken for the verification process in seconds.
    """

    return verification.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
        silent=silent,
        threshold=threshold,
        anti_spoofing=anti_spoofing,
    )


def verify2(
    img1_path: str,        
    img2_path: str,
    model_name: str = "VGG-Face",
    detector_backend: str = "retinaface",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
):
    """
    Verify if an image pair represents the same person or different persons.
    Args:
        img1_path (str or np.ndarray or List[float]): Path to the first image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images
            or pre-calculated embeddings.

        img2_path (str or np.ndarray or List[float]): Path to the second image.
            Accepts exact image path as a string, numpy array (BGR), base64 encoded images
            or pre-calculated embeddings.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        result (dict): A dictionary containing verification results with following keys.

        - 'verified' (bool): Indicates whether the images represent the same person (True)
            or different persons (False).

        - 'distance' (float): The distance measure between the face vectors.
            A lower distance indicates higher similarity.

        - 'threshold' (float): The maximum threshold used for verification.
            If the distance is below this threshold, the images are considered a match.

        - 'model' (str): The chosen face recognition model.

        - 'distance_metric' (str): The chosen similarity metric for measuring distances.

        - 'facial_areas' (dict): Rectangular regions of interest for faces in both images.
            - 'img1': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the first image.
            - 'img2': {'x': int, 'y': int, 'w': int, 'h': int}
                    Region of interest for the second image.

        - 'time' (float): Time taken for the verification process in seconds.
    """

    img2_paths = img2_path.split(",")
    distances = []
    for i in range(len(img2_paths)):
        img2_path = img2_paths[i].strip()
        # Extend "~" symbol to the home directory in the path
        img2_path = os.path.expanduser(img2_path)

        try:
            result = verification.verify(
                    img1_path=img1_path,
                    img2_path=img2_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    enforce_detection=enforce_detection,
                    align=align,
                    expand_percentage=expand_percentage,
                    normalization=normalization,
                    silent=silent,
                    threshold=threshold,
                    anti_spoofing=anti_spoofing,
                )
            print(f"{img2_path}: {result['distance']:.3f}")
            distances.append(result['distance'])
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    if len(distances) == 0:
        return None
    
    avg_distance = sum(distances) / len(distances)
    distances.append(avg_distance)
    print(f"Average distance: {avg_distance:.3f}")
    return distances

def show(history_records, prompt_pat, exclude_pats, indices, last_n=6):
    if history_records is None and prompt_pat is not None:
        with open("manual-eval.log") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if prompt_pat in line]
            if len(exclude_pats) > 0:
                lines = [line for line in lines if all([exclude_pat not in line for exclude_pat in exclude_pats])]
            history_records = lines
    
    if len(history_records) == 0:
        print("No history records yet. Do a face verify first to pull some records.")
        return

    record_sigs = [ " ".join(record.split()[:2]) for record in history_records]
    # Deduplicate lines
    history_records = list(dict(zip(record_sigs, history_records)).values())

    indices = indices.split(",")    
    sel_records = []
    # Select records based on indices. Support -3:, -4:-2, :2, etc.
    for index in indices:
        if index == ':':
            sel_records = history_records
            break
        # 2:5, -4:, etc.
        elif ':' in index:
            id1, id2 = index.split(':')
            if id1 == "":
                id1 = 0
            else:
                id1 = int(id1)
                # Positive indices are 1-based, map them to 0-based
                if id1 > 0:
                    id1 -= 1
            if id2 == "":
                id2 = len(history_records)
            else:
                id2 = int(id2)
                # Positive indices are 1-based, map them to 0-based
                if id2 > 0:
                    id2 -= 1
            sel_records.extend(history_records[id1:id2])
        # < 0.4, < 0.5, etc, as a filter condition.
        elif index.startswith('<'):
            thres = float(index[1:])
            # The last element is the average distance. So use it for filtering.
            sel_records.extend([record for record in history_records if float(record.split()[-1]) < thres])
        else:
            index = int(index)
            # Positive indices are 1-based, map them to 0-based
            if index > 0:
                index -= 1
            sel_records.append(history_records[index])

    sel_records = sel_records[-last_n:]
    rows_paths = []
    for i, record in enumerate(sel_records):
        subj_prompt, ckpt_sig, *distances = record.split()
        subj, prompt_sig = subj_prompt.split("-")
        row_paths = [ f"~/test/{subj}-adaface{ckpt_sig}-{prompt_sig}-{i}.png" for i in range(1, 5) ]
        rows_paths.append(row_paths)
        print(i+1, record)

    if prompt_pat is None:
        prompt_pat = subj_prompt
        
    # Stitch images together, each list in rows_paths as a row
    imgs = []
    for row_paths in rows_paths:
        row_imgs = [ cv2.imread(os.path.expanduser(img_path)) for img_path in row_paths ]
        imgs.append(row_imgs)
    imgs = [ np.concatenate(row_imgs, axis=1) for row_imgs in imgs ]
    img = np.concatenate(imgs, axis=0)
    grid_image_path = os.path.expanduser(f"~/test/{prompt_pat}.png")
    cv2.imwrite(grid_image_path, img)
    print(f"gpicview {grid_image_path}") 
    # call gpicview to show the image
    os.system(f"gpicview {grid_image_path}")

def console(image_root="~/test", last_n=10, model_name="VGG-Face",
            detector_backend="retinaface"):
    """Continuously prompt the user for input and evaluate it."""
    print("Interactive Evaluator: Type arguments and press Enter (type 'q' to quit)")
    print("Example: 121105-9000 wiki guitar")
    signal.signal(signal.SIGINT, lambda signum, frame: print("\nCtrl+C disabled. Type 'q' to quit."))    

    history_records = []

    while True:
        try:
            # Prompt for input
            user_input = input(">> ")
            user_input = user_input.strip()

            # Exit condition
            if user_input == "q":
                print("Exiting interactive evaluator.")
                break

            # Split input into arguments and call the function
            if user_input.strip():
                # Show relevant images as a grid.
                if user_input.startswith('show '):
                    args = user_input[5:].split()       
                    exclude_pats = []
                    while args[-1].startswith('~'):
                        exclude_pats.append(args.pop(-1)[1:])

                    if len(args) == 2:
                        subj_prompt, indices = args
                        show(None, subj_prompt, exclude_pats, indices, last_n=20)
                    elif len(args) == 1:
                        if re.match(r'^[a-z0-9-]+$', args[0]):
                            subj_prompt = args[0]
                            indices = ':'
                            show(None, subj_prompt, exclude_pats, indices, last_n=20)
                        else:
                            indices = args[0]
                            show(history_records, None, exclude_pats, indices, last_n=20)
                    else:
                        print("Invalid input. Format: <subj-prompt_signature> <index1,index2,...>")
                    continue

                # Do face validation.
                args = user_input.split()
                exclude_pats = []
                while args[-1].startswith('~'):
                    exclude_pats.append(args.pop(-1)[1:])

                if len(args) == 3:
                    ckpt_iter, subject, prompt_sig = args
                    # Paths are hard-coded for now
                    img1_path = os.path.expanduser(f"{image_root}/{subject}-1.jpg")
                    img2_paths = [ os.path.expanduser(f"{image_root}/{subject}-adaface{ckpt_iter}-{prompt_sig}-{i}.png") for i in range(1, 5) ]
                    img2_path = ",".join(img2_paths)
                    do_adaface_eval = True
                elif len(args) == 2:
                    img1_path, img2_path = args
                    do_adaface_eval = False
                else:
                    print("Invalid input. Please provide 3 arguments.")
                    continue

                print(f"verify2: {img1_path} {img2_path}")
                distances = None
                try:
                    distances = verify2(img1_path, img2_path, model_name=model_name, detector_backend=detector_backend)
                except Exception as e:
                    print(f"Error: {e}")
                if distances is not None and do_adaface_eval:
                    with open("manual-eval.log", "a") as f:
                        f.write(f"{subject}-{prompt_sig} {ckpt_iter} ")
                        f.write(" ".join([f"{distance:.3f}" for distance in distances]))
                        f.write("\n")

                    # grep manual-eval.log for lines starting with {subject}-{prompt_sig}
                    with open("manual-eval.log") as f:
                        lines = f.readlines()
                        lines = [line.strip() for line in lines if line.startswith(f"{subject}-{prompt_sig}")]
                        if len(exclude_pats) > 0:
                            lines = [line for line in lines if all([exclude_pat not in line for exclude_pat in exclude_pats])]
                        # Deduplicate lines
                        lines = list(dict.fromkeys(lines))                        
                        lines = lines[-last_n:]
                        for no, line in enumerate(lines):
                            print(no+1, line)
                        history_records = lines

            else:
                print("No arguments provided. Try again.")
        except EOFError:
            # Handle Ctrl+D gracefully
            print("\nExiting interactive evaluator.")
            break
        except Exception as e:
            print(f"Error: {e}")


def analyze(
    img_path: Union[str, np.ndarray],
    actions: Union[tuple, list] = ("emotion", "age", "gender", "race"),
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    silent: bool = False,
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:
    """
    Analyze facial attributes such as age, gender, emotion, and race in the provided image.
    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        actions (tuple): Attributes to analyze. The default is ('age', 'gender', 'emotion', 'race').
            You can exclude some of these attributes from the analysis if needed.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents
           the analysis results for a detected face. Each dictionary in the list contains the
           following keys:

        - 'region' (dict): Represents the rectangular region of the detected face in the image.
            - 'x': x-coordinate of the top-left corner of the face.
            - 'y': y-coordinate of the top-left corner of the face.
            - 'w': Width of the detected face region.
            - 'h': Height of the detected face region.

        - 'age' (float): Estimated age of the detected face.

        - 'face_confidence' (float): Confidence score for the detected face.
            Indicates the reliability of the face detection.

        - 'dominant_gender' (str): The dominant gender in the detected face.
            Either "Man" or "Woman".

        - 'gender' (dict): Confidence scores for each gender category.
            - 'Man': Confidence score for the male gender.
            - 'Woman': Confidence score for the female gender.

        - 'dominant_emotion' (str): The dominant emotion in the detected face.
            Possible values include "sad," "angry," "surprise," "fear," "happy,"
            "disgust," and "neutral"

        - 'emotion' (dict): Confidence scores for each emotion category.
            - 'sad': Confidence score for sadness.
            - 'angry': Confidence score for anger.
            - 'surprise': Confidence score for surprise.
            - 'fear': Confidence score for fear.
            - 'happy': Confidence score for happiness.
            - 'disgust': Confidence score for disgust.
            - 'neutral': Confidence score for neutrality.

        - 'dominant_race' (str): The dominant race in the detected face.
            Possible values include "indian," "asian," "latino hispanic,"
            "black," "middle eastern," and "white."

        - 'race' (dict): Confidence scores for each race category.
            - 'indian': Confidence score for Indian ethnicity.
            - 'asian': Confidence score for Asian ethnicity.
            - 'latino hispanic': Confidence score for Latino/Hispanic ethnicity.
            - 'black': Confidence score for Black ethnicity.
            - 'middle eastern': Confidence score for Middle Eastern ethnicity.
            - 'white': Confidence score for White ethnicity.
    """
    return demography.analyze(
        img_path=img_path,
        actions=actions,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        silent=silent,
        anti_spoofing=anti_spoofing,
    )


def find(
    img_path: Union[str, np.ndarray],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
    silent: bool = False,
    refresh_database: bool = True,
    anti_spoofing: bool = False,
    batched: bool = False,
) -> Union[List[pd.DataFrame], List[List[Dict[str, Any]]]]:
    """
    Identify individuals in a database
    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        refresh_database (boolean): Synchronizes the images representation (pkl) file with the
            directory/db files, if set to false, it will ignore any file changes inside the db_path
            (default is True).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        results (List[pd.DataFrame] or List[List[Dict[str, Any]]]):
            A list of pandas dataframes (if `batched=False`) or
            a list of dicts (if `batched=True`).
            Each dataframe or dict corresponds to the identity information for
            an individual detected in the source image.

            Note: If you have a large database and/or a source photo with many faces,
            use `batched=True`, as it is optimized for large batch processing.
            Please pay attention that when using `batched=True`, the function returns
            a list of dicts (not a list of DataFrames),
            but with the same keys as the columns in the DataFrame.

            The DataFrame columns or dict keys include:

            - 'identity': Identity label of the detected individual.

            - 'target_x', 'target_y', 'target_w', 'target_h': Bounding box coordinates of the
                    target face in the database.

            - 'source_x', 'source_y', 'source_w', 'source_h': Bounding box coordinates of the
                    detected face in the source image.

            - 'threshold': threshold to determine a pair whether same person or different persons

            - 'distance': Similarity score between the faces based on the
                    specified model and distance metric
    """
    return recognition.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        threshold=threshold,
        normalization=normalization,
        silent=silent,
        refresh_database=refresh_database,
        anti_spoofing=anti_spoofing,
        batched=batched,
    )


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet
            (default is VGG-Face.).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images
            (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace
            (default is base).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        max_faces (int): Set a limit on the number of faces to be processed (default is None).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, each containing the
            following fields:

        - embedding (List[float]): Multidimensional vector representing facial features.
            The number of dimensions varies based on the reference model
            (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).

        - facial_area (dict): Detected facial area by face detection in dictionary format.
            Contains 'x' and 'y' as the left-corner point, and 'w' and 'h'
            as the width and height. If `detector_backend` is set to 'skip', it represents
            the full image area and is nonsensical.

        - face_confidence (float): Confidence score of face detection. If `detector_backend` is set
            to 'skip', the confidence will be 0 and is nonsensical.
    """
    return representation.represent(
        img_path=img_path,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces,
    )


def stream(
    db_path: str = "",
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enable_face_analysis: bool = True,
    source: Any = 0,
    time_threshold: int = 5,
    frame_threshold: int = 5,
    anti_spoofing: bool = False,
) -> None:
    """
    Run real time face recognition and facial attribute analysis

    Args:
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enable_face_analysis (bool): Flag to enable face analysis (default is True).

        source (Any): The source for the video stream (default is 0, which represents the
            default camera).

        time_threshold (int): The time threshold (in seconds) for face recognition (default is 5).

        frame_threshold (int): The frame threshold for face recognition (default is 5).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).
    Returns:
        None
    """

    time_threshold = max(time_threshold, 1)
    frame_threshold = max(frame_threshold, 1)

    streaming.analysis(
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enable_face_analysis=enable_face_analysis,
        source=source,
        time_threshold=time_threshold,
        frame_threshold=frame_threshold,
        anti_spoofing=anti_spoofing,
    )


def extract_faces(
    img_path: Union[str, np.ndarray],
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    grayscale: bool = False,
    color_face: str = "rgb",
    normalize_face: bool = True,
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract faces from a given image

    Args:
        img_path (str or np.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        grayscale (boolean): (Deprecated) Flag to convert the output face image to grayscale
            (default is False).

        color_face (string): Color to return face image output. Options: 'rgb', 'bgr' or 'gray'
            (default is 'rgb').

        normalize_face (boolean): Flag to enable normalization (divide by 255) of the output
            face image output face image normalization (default is True).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary contains:

        - "face" (np.ndarray): The detected face as a NumPy array.

        - "facial_area" (Dict[str, Any]): The detected face's regions as a dictionary containing:
            - keys 'x', 'y', 'w', 'h' with int values
            - keys 'left_eye', 'right_eye' with a tuple of 2 ints as values. left and right eyes
                are eyes on the left and right respectively with respect to the person itself
                instead of observer.

        - "confidence" (float): The confidence score associated with the detected face.

        - "is_real" (boolean): antispoofing analyze result. this key is just available in the
            result only if anti_spoofing is set to True in input arguments.

        - "antispoof_score" (float): score of antispoofing analyze result. this key is
            just available in the result only if anti_spoofing is set to True in input arguments.
    """

    return detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        grayscale=grayscale,
        color_face=color_face,
        normalize_face=normalize_face,
        anti_spoofing=anti_spoofing,
    )


def cli() -> None:
    """
    command line interface function will be offered in this block
    """
    import fire

    fire.Fire()


# deprecated function(s)


def detectFace(
    img_path: Union[str, np.ndarray],
    target_size: tuple = (224, 224),
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
) -> Union[np.ndarray, None]:
    """
    Deprecated face detection function. Use extract_faces for same functionality.

    Args:
        img_path (str or np.ndarray): Path to the first image. Accepts exact image path
            as a string, numpy array (BGR), or base64 encoded images.

        target_size (tuple): final shape of facial image. black pixels will be
            added to resize the image (default is (224, 224)).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

    Returns:
        img (np.ndarray): detected (and aligned) facial area image as numpy array
    """
    logger.warn("Function detectFace is deprecated. Use extract_faces instead.")
    face_objs = extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )
    extracted_face = None
    if len(face_objs) > 0:
        extracted_face = face_objs[0]["face"]
        extracted_face = preprocessing.resize_image(img=extracted_face, target_size=target_size)
    return extracted_face
