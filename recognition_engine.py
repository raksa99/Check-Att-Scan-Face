import cv2
import face_recognition
import numpy as np
import database

class FaceRecognitionEngine:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_metadata = []  # List of dicts with 'id' and 'name'
        self.load_known_faces()

    def load_known_faces(self):
        """Loads all registered users and their face encodings from the database into memory cache."""
        self.known_face_encodings = []
        self.known_face_metadata = []
        
        users = database.get_all_users()
        for user in users:
            # Reconstruct the 128D numpy array from list
            encoding_np = np.array(user['face_encoding'])
            self.known_face_encodings.append(encoding_np)
            self.known_face_metadata.append({
                'id': user['id'],
                'name': user['name']
            })

    def extract_encoding(self, image_np):
        """
        Extracts face encoding from a single image numpy array (RGB).
        
        :param image_np: RGB image numpy array
        :return: 128-dimensional numpy array representing the face encoding
        :raises ValueError: If no face or multiple faces are detected
        """
        # Find all the faces and face encodings in the image
        face_locations = face_recognition.face_locations(image_np)
        
        if len(face_locations) == 0:
            raise ValueError("No face detected in the image. Please make sure your face is visible and well-lit.")
        elif len(face_locations) > 1:
            raise ValueError(f"Multiple faces ({len(face_locations)}) detected. Please capture/upload an image with exactly one face.")
            
        # Get encoding for the single detected face
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        return face_encodings[0]

    def recognize_faces_in_frame(self, frame_bgr, tolerance=0.6):
        """
        Detects and recognizes faces in a single video frame.
        
        :param frame_bgr: Webcam frame in BGR format (standard OpenCV format)
        :param tolerance: Distance threshold for face matching (lower is stricter, default 0.6)
        :return: List of dicts, each containing:
                 - 'location': (top, right, bottom, left) coordinates scaled to the original frame
                 - 'id': Registered user ID (None if unknown)
                 - 'name': Registered user name ('Unknown' if unrecognized)
                 - 'is_known': Boolean indicating whether the face is registered
                 - 'distance': Match distance float (None if no known faces exist)
        """
        # Resize frame to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        results = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Default values
            user_id = None
            name = "Unknown"
            is_known = False
            distance = None
            
            # Only compare if there are registered faces in the database
            if len(self.known_face_encodings) > 0:
                # Filter out placeholder encodings (all zeros) to prevent false positives
                valid_indices = [
                    i for i, enc in enumerate(self.known_face_encodings)
                    if not np.all(enc == 0)
                ]
                
                if valid_indices:
                    valid_encodings = [self.known_face_encodings[i] for i in valid_indices]
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(valid_encodings, face_encoding, tolerance=tolerance)
                    face_distances = face_recognition.face_distance(valid_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        distance = float(face_distances[best_match_index])
                        
                        if matches[best_match_index]:
                            orig_index = valid_indices[best_match_index]
                            metadata = self.known_face_metadata[orig_index]
                            user_id = metadata['id']
                            name = metadata['name']
                            is_known = True
            
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            results.append({
                'location': (top, right, bottom, left),
                'id': user_id,
                'name': name,
                'is_known': is_known,
                'distance': distance
            })
            
        return results
