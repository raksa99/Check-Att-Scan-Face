import io
import cv2
import numpy as np
import pandas as pd

def bytes_to_cv2_image(image_bytes):
    """
    Converts raw image bytes (e.g. from Streamlit file_uploader or camera_input)
    into a BGR OpenCV image.
    """
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image_bgr

def convert_df_to_excel(df):
    """
    Converts a pandas DataFrame into Excel bytes using openpyxl.
    """
    output = io.BytesIO()
    # Use pandas to write to the memory stream
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance Reports')
    return output.getvalue()
