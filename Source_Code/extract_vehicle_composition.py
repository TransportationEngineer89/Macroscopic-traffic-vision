import cv2
from ultralytics import YOLO
import csv 

# --- Helper Functions for Line Intersection ---
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
# ----------------------------------------------

# 1. Load model and video
model = YOLO('yolo11n.pt') 
video_path = r"D:\Group 2.mp4" # Replace with your video path
cap = cv2.VideoCapture(video_path)

# 2. Define your slanted counting lines
# IMPORTANT: Make sure these coordinates match your new 1920x1080 layout!
lane1_line = ((602, 514), (981, 676)) 
lane2_line = ((1002, 598), (1327, 732)) 

# 3. Dictionaries for tracking
track_history = {} 
vehicle_data = {} 
counted_ids = set() 
class_names = {2: 'Car', 3: 'Two-Wheeler', 5: 'Bus', 7: 'Truck'}

# 4. Area Thresholds for LCV Logic
# NOTE: Tune these values based on your specific 1920x1080 video!
LCV_MAX_AREA_PIXELS = 105743    # If a Truck is smaller than this, it's an LCV
LCV_CAR_MIN_AREA_PIXELS = 88850 # If a Car is larger than this, it's an LCV

# 5. Open the CSV file in write mode ('w')
csv_filename = 'traffic_data.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp (s)', 'Track_ID', 'Vehicle_Class', 'Lane'])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Resize the frame to 1920x1080 immediately after reading
        #frame = cv2.resize(frame, (1920, 1080))
        
        # Get the current timestamp of the video in seconds
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = round(current_time_ms / 1000.0, 2)
        
        results = model.track(frame, classes=[2, 3, 5, 7], persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # Draw the lines
        cv2.line(frame, lane1_line[0], lane1_line[1], (0, 255, 0), 2) 
        cv2.line(frame, lane2_line[0], lane2_line[1], (255, 0, 0), 2) 
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy() 
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) 
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int) 
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                curr_pt = (cx, cy)
                
                box_area = (x2 - x1) * (y2 - y1)
                
                if track_id not in vehicle_data:
                    vehicle_data[track_id] = {
                        'class': class_names.get(class_id, "Unknown"),
                        'box_area_at_entry': box_area
                    }
                
                cv2.circle(frame, curr_pt, 4, (0, 0, 255), -1)
                
                if track_id in track_history:
                    prev_pt = track_history[track_id]
                    
                    crossed_lane = None
                    if intersect(prev_pt, curr_pt, lane1_line[0], lane1_line[1]) and track_id not in counted_ids:
                        crossed_lane = "Lane 1"
                    elif intersect(prev_pt, curr_pt, lane2_line[0], lane2_line[1]) and track_id not in counted_ids:
                        crossed_lane = "Lane 2"
                    
                    if crossed_lane:
                        counted_ids.add(track_id)
                        
                        # --- Hybrid Classification Logic for LCV ---
                        v_class = vehicle_data[track_id]['class']
                        v_area = vehicle_data[track_id]['box_area_at_entry']
                        
                        if v_class == 'Truck' and v_area < LCV_MAX_AREA_PIXELS:
                            v_class = 'LCV'
                        elif v_class == 'Car' and v_area > LCV_CAR_MIN_AREA_PIXELS:
                            v_class = 'LCV'
                        # -------------------------------------------
                        
                        print(f"Time: {timestamp_sec}s | ID: {track_id} | Class: {v_class} | Lane: {crossed_lane}")
                        writer.writerow([timestamp_sec, track_id, v_class, crossed_lane])
                
                track_history[track_id] = curr_pt

        cv2.imshow("Traffic Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Processing complete. Data saved to {csv_filename}")