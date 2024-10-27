import cv2
import mediapipe as mp
import numpy as np

class BellsPalsyEffect:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Enhanced landmarks - removed nose area landmarks
        self.MOUTH_LANDMARKS = [61, 62, 63, 64, 65, 66, 67]   # Left mouth landmarks
        self.LEFT_JAW_LANDMARKS = [58, 172, 136, 150, 149, 176, 148, 152]
        self.LEFT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173]  # Enhanced eye landmarks
        self.LEFT_EYE_LOWER = [247, 30, 29, 27, 28, 56, 190]  # Lower eye region for drool
        
        # Fixed parameters
        self.strength = -0.2  # Fixed strength as requested
        self.drool_intensity = 0.5  # Fixed drool intensity

    def create_drool_effect(self, img, start_x, start_y, strength=1.0):
        """Create a drooling effect with fixed array indexing"""
        try:
            if img is None or not isinstance(strength, (int, float)):
                return img
            
            height, width = img.shape[:2]
            
            if not self.validate_coordinates(start_x, start_y, width, height):
                return img
            
            # Create separate mask for drool effect
            mask = np.zeros((height, width), dtype=np.float32)
            
            drool_length = int(40 * abs(strength))
            drool_width = max(1, int(5 * abs(strength)))
            
            points = []
            for i in range(drool_length):
                x_offset = int(np.sin(i / drool_length * np.pi) * 5)
                x = start_x + x_offset
                y = start_y + i
                
                if self.validate_coordinates(x, y, width, height):
                    points.append((x, y))
                else:
                    break

            if not points:
                return img

            # Draw drool points
            for i, (x, y) in enumerate(points):
                opacity = 1.0 - (i / len(points))
                cv2.circle(mask, (x, y), drool_width, opacity, -1)

            # Apply Gaussian blur to smooth the effect
            mask = cv2.GaussianBlur(mask, (7, 7), 2)
            
            # Create result image
            result = img.copy()
            
            # Apply the drool effect using proper broadcasting
            for c in range(3):  # Process each color channel separately
                result[:, :, c] = (
                    result[:, :, c] * (1 - mask) + 
                    255 * mask
                ).astype(np.uint8)
            
            return result
        except Exception as e:
            print(f"Error in drool effect: {str(e)}")
            return img

    def validate_coordinates(self, x, y, width, height):
        """Validate if coordinates are within image bounds"""
        return (0 <= x < width) and (0 <= y < height)

    def create_drooping_displacement(self, height, width, center_x, center_y, 
                                   radius, strength):
        """Create a downward and slightly outward drooping effect"""
        try:
            y, x = np.mgrid[0:height, 0:width]
            dx = x - center_x
            dy = y - center_y
            distances = np.sqrt(dx*dx + dy*dy)
            
            falloff = np.exp(-distances**2 / (2 * radius**2))
            displacement_y = strength * falloff
            displacement_x = -0.4 * strength * falloff * (dy / (distances + 1e-6))
            
            return displacement_x, displacement_y
        except Exception as e:
            print(f"Error in displacement map creation: {str(e)}")
            return np.zeros((height, width)), np.zeros((height, width))

    def apply_displacement(self, img, disp_x, disp_y):
        """Apply displacement map to image"""
        try:
            height, width = img.shape[:2]
            y, x = np.mgrid[0:height, 0:width]
            
            map_x = (x + disp_x).astype(np.float32)
            map_y = (y + disp_y).astype(np.float32)
            
            return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Error in displacement application: {str(e)}")
            return img

    def apply_eye_effect(self, img, landmarks, strength=30):
        """Apply enhanced eye drooping effect"""
        try:
            if img is None or landmarks is None:
                return img

            height, width = img.shape[:2]
            
            eye_corner = landmarks.landmark[self.LEFT_EYE_LANDMARKS[0]]
            if not (0 <= eye_corner.x <= 1 and 0 <= eye_corner.y <= 1):
                return img
            
            corner_x = int(eye_corner.x * width)
            corner_y = int(eye_corner.y * height)
            
            disp_x, disp_y = self.create_drooping_displacement(
                height, width, corner_x, corner_y,
                radius=30, strength=strength
            )
            result = self.apply_displacement(img, disp_x, disp_y)
            
            lower_eye_point = landmarks.landmark[self.LEFT_EYE_LOWER[3]]
            if not (0 <= lower_eye_point.x <= 1 and 0 <= lower_eye_point.y <= 1):
                return result
            
            drool_start_x = int(lower_eye_point.x * width)
            drool_start_y = int(lower_eye_point.y * height)
            
            if self.validate_coordinates(drool_start_x, drool_start_y, width, height):
                result = self.create_drool_effect(
                    result, 
                    drool_start_x, 
                    drool_start_y, 
                    self.drool_intensity * abs(self.strength)
                )
            
            return result
        except Exception as e:
            print(f"Error in eye effect: {str(e)}")
            return img

    def process_frame(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return frame

            landmarks = results.multi_face_landmarks[0]
            result_frame = frame.copy()

            # Apply all effects with fixed strength
            result_frame = self.apply_mouth_droop(result_frame, landmarks, 30 * self.strength)
            result_frame = self.apply_jaw_droop(result_frame, landmarks, 25 * self.strength)
            result_frame = self.apply_eye_effect(result_frame, landmarks, 20 * self.strength)

            return result_frame
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            return frame

    def apply_mouth_droop(self, img, landmarks, strength=30):
        """Apply mouth drooping effect on left side"""
        try:
            height, width = img.shape[:2]
            corner_point = landmarks.landmark[61]
            
            if not (0 <= corner_point.x <= 1 and 0 <= corner_point.y <= 1):
                return img
                
            corner_x = int(corner_point.x * width)
            corner_y = int(corner_point.y * height)
            
            disp_x, disp_y = self.create_drooping_displacement(
                height, width, corner_x, corner_y,
                radius=40, strength=strength
            )
            
            return self.apply_displacement(img, disp_x, disp_y)
        except Exception as e:
            print(f"Error in mouth droop: {str(e)}")
            return img

    def apply_jaw_droop(self, img, landmarks, strength=25):
        """Apply jaw drooping effect on left side"""
        try:
            height, width = img.shape[:2]
            jaw_points = [(int(landmarks.landmark[idx].x * width), 
                          int(landmarks.landmark[idx].y * height)) 
                         for idx in self.LEFT_JAW_LANDMARKS]
            
            if not jaw_points:
                return img
                
            center_x = int(np.mean([p[0] for p in jaw_points]))
            center_y = int(np.mean([p[1] for p in jaw_points]))
            
            disp_x, disp_y = self.create_drooping_displacement(
                height, width, center_x, center_y,
                radius=45, strength=strength
            )
            
            return self.apply_displacement(img, disp_x, disp_y)
        except Exception as e:
            print(f"Error in jaw droop: {str(e)}")
            return img

def main():
    try:
        processor = BellsPalsyEffect()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        window_name = "Bell's Palsy Simulation"
        cv2.namedWindow(window_name)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            processed_frame = processor.process_frame(frame)
            cv2.imshow(window_name, processed_frame)

            # Only check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()