from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from bells_palsy_effect import BellsPalsyEffect
import threading
import queue
import time
from typing import Optional, Generator

app = Flask(__name__)

class VideoStream:
    def __init__(self, resolution: tuple = (640, 480), queue_size: int = 4):
        self.processor = BellsPalsyEffect()
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.resolution = resolution
        self.fps_limiter = FPSLimiter(target_fps=30)
        self.camera: Optional[cv2.VideoCapture] = None
        
    def initialize_camera(self) -> bool:
        """Initialize the camera with specified settings."""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            return False
            
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay
        return True
        
    def generate_frames(self) -> None:
        """Capture and process frames in a separate thread."""
        if not self.initialize_camera():
            print("Error: Could not initialize camera")
            return
            
        try:
            while not self.stop_event.is_set():
                self.fps_limiter.limit()  # Control frame rate
                
                ret, frame = self.camera.read()
                if not ret:
                    continue
                    
                # Process frame with Bell's Palsy effect
                try:
                    processed_frame = self.processor.process_frame(frame)
                    ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        self._update_queue(frame_bytes)
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
                    
        finally:
            if self.camera is not None:
                self.camera.release()
                
    def _update_queue(self, frame_bytes: bytes) -> None:
        """Update the frame queue, dropping old frames if necessary."""
        try:
            self.frame_queue.put_nowait(frame_bytes)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()  # Remove oldest frame
                self.frame_queue.put_nowait(frame_bytes)
            except queue.Empty:
                pass
                
    def stream_frames(self) -> Generator[bytes, None, None]:
        """Generate frames for streaming."""
        while not self.stop_event.is_set():
            try:
                frame_bytes = self.frame_queue.get(timeout=0.5)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                continue  # Skip if no frame is available
                
    def start(self) -> None:
        """Start the video streaming thread."""
        threading.Thread(target=self.generate_frames, daemon=True).start()
        
    def stop(self) -> None:
        """Stop the video streaming."""
        self.stop_event.set()


class FPSLimiter:
    def __init__(self, target_fps: int):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.last_frame_time = time.time()
        
    def limit(self) -> None:
        """Limit the frame rate to target FPS."""
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        sleep_time = max(0, self.frame_time - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_frame_time = time.time()


# Create global video stream instance
video_stream = VideoStream()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream.stream_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        video_stream.start()
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        video_stream.stop()