import pathlib

_ASSETS_DIRECTORY_PATH = pathlib.Path(__file__).parent.joinpath("assets").resolve()

FACE_LANDMARKS_MODEL_PATH = _ASSETS_DIRECTORY_PATH / "shape_predictor_68_face_landmarks.dat"
