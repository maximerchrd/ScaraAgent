import cv2

# ==========================================
# CONFIGURATION
# ==========================================
USE_ESP32 = True  # Set to True for ESP32, False for MacBook webcam
ESP32_URL = "http://10.86.51.100:81/stream"  # Update with your ESP32 IP

# The dictionary MUST match the marker you printed.
# Common options: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_ARUCO_ORIGINAL
ARUCO_DICT = cv2.aruco.DICT_4X4_50 
# ==========================================

def get_aruco_detector():
    """Handles API changes between OpenCV 4.6- and 4.7+"""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    
    # OpenCV 4.7+ API
    if hasattr(cv2.aruco, 'ArucoDetector'):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector, None, None
    # OpenCV 4.6 and older API
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        return None, dictionary, parameters

def detect_markers(image, detector, dictionary, parameters):
    """Converts image to grayscale and detects markers"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids

def main():
    detector, dictionary, parameters = get_aruco_detector()
    
    # Select the video source based on the USE_ESP32 flag
    source = ESP32_URL if USE_ESP32 else 0
    print(f"Connecting to video source: {source}")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}.")
        return

    print("Stream running. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        #frame = cv2.flip(frame, 1)
        if not ret:
            print("Failed to grab frame. Reconnecting...")
            cv2.waitKey(1000)
            cap = cv2.VideoCapture(source)
            continue
            
        # Detect and draw ArUco markers
        corners, ids = detect_markers(frame, detector, dictionary, parameters)
        
        if ids is not None and len(ids) > 0:
            # Overlays the green square and the ID number
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
        # Display the live feed
        cv2.imshow('ArUco Tracker', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()