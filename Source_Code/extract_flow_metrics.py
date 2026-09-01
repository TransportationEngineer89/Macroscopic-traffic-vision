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
video_path = r"D:\IITG\Sem 2\TE\Lab\Group 4.mp4" # Replace with your video path
cap = cv2.VideoCapture(video_path)

# 2. Speed Trap lines (Distance MUST be 50m)
lane1_entry = ((1703, 517), (1788, 560)) 
lane1_exit  = ((600, 500), (1300, 620)) 
lane2_entry = ((1797, 544), (1876, 569)) 
lane2_exit  = ((1310, 575), (1697, 654)) 
TRAP_DISTANCE_METERS = 50.0 

# 3. Dictionaries for tracking
track_history = {} 
vehicle_data = {}  
counted_ids = set() 
last_entry_time = {'Lane 1': None, 'Lane 2': None} 

# 4. Open the CSV file in write mode ('w')
csv_filename = 'traffic_data_flow_metrics.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # The header no longer contains vehicle class
    writer.writerow(['Timestamp (s)', 'Track_ID', 'Lane', 'Speed_kmh', 'Time_Headway_s', 'Space_Headway_m', 'Flow_State'])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Enforce the 1920x1080 resolution for coordinate accuracy
        frame = cv2.resize(frame, (1920, 1080))
        
        timestamp_sec = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
        
        # Tracking: We filter for vehicle COCO classes so we don't track pedestrians, 
        # but we no longer extract or care about what the specific class is.
        results = model.track(frame, classes=[1, 2, 3, 5, 7], persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # Draw the lines
        cv2.line(frame, lane1_entry[0], lane1_entry[1], (0, 255, 0), 2) 
        cv2.line(frame, lane1_exit[0], lane1_exit[1], (0, 0, 255), 2) 
        cv2.line(frame, lane2_entry[0], lane2_entry[1], (0, 255, 0), 2) 
        cv2.line(frame, lane2_exit[0], lane2_exit[1], (0, 0, 255), 2) 
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy() 
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) 
            
            # We no longer extract class_ids
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                curr_pt = (cx, cy)
                
                cv2.circle(frame, curr_pt, 4, (0, 255, 255), -1) 
                
                if track_id in track_history:
                    prev_pt = track_history[track_id]
                    
                    # --- Check ENTRY Lines ---
                    if track_id not in vehicle_data:
                        entry_lane = None
                        if intersect(prev_pt, curr_pt, lane1_entry[0], lane1_entry[1]):
                            entry_lane = 'Lane 1'
                        elif intersect(prev_pt, curr_pt, lane2_entry[0], lane2_entry[1]):
                            entry_lane = 'Lane 2'
                            
                        if entry_lane:
                            time_headway = None
                            if last_entry_time[entry_lane] is not None:
                                time_headway = round(timestamp_sec - last_entry_time[entry_lane], 2)
                            
                            last_entry_time[entry_lane] = timestamp_sec
                            
                            vehicle_data[track_id] = {
                                'entry_time': timestamp_sec,
                                'lane': entry_lane,
                                'time_headway': time_headway
                            }
                    
                    # --- Check EXIT Lines ---
                    elif track_id in vehicle_data and track_id not in counted_ids:
                        exit_lane = None
                        if intersect(prev_pt, curr_pt, lane1_exit[0], lane1_exit[1]):
                            exit_lane = 'Lane 1'
                        elif intersect(prev_pt, curr_pt, lane2_exit[0], lane2_exit[1]):
                            exit_lane = 'Lane 2'
                            
                        if exit_lane and exit_lane == vehicle_data[track_id]['lane']:
                            entry_time = vehicle_data[track_id]['entry_time']
                            time_taken = timestamp_sec - entry_time
                            
                            if time_taken > 0:
                                # Speed Calculations
                                speed_ms = TRAP_DISTANCE_METERS / time_taken
                                speed_kmh = round(speed_ms * 3.6, 2)
                                
                                # Space Headway and Flow State Logic
                                time_hw_val = vehicle_data[track_id]['time_headway']
                                space_hw_m = ""
                                flow_state = "Lead Vehicle" # First vehicle has no headway
                                
                                if time_hw_val is not None:
                                    space_hw_val = time_hw_val * speed_ms
                                    space_hw_m = round(space_hw_val, 2)
                                    if space_hw_val > 50.0:
                                        flow_state = "Free Flow"
                                    else:
                                        flow_state = "Stream Flow"
                                
                                time_hw_str = time_hw_val if time_hw_val is not None else ""
                                
                                print(f"Logged -> ID:{track_id} | Spd:{speed_kmh}km/h | Spc_HW:{space_hw_m}m | State:{flow_state}")
                                
                                writer.writerow([timestamp_sec, track_id, exit_lane, speed_kmh, time_hw_str, space_hw_m, flow_state])
                                counted_ids.add(track_id)

                track_history[track_id] = curr_pt

        cv2.imshow("50m Speed Trap & Flow Extraction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Data extraction complete. Saved to {csv_filename}")