import dlib

from dfd import assets

def test_face_landmarks_model_loads():

    dlib.shape_predictor(str(assets.FACE_LANDMARKS_MODEL_PATH))