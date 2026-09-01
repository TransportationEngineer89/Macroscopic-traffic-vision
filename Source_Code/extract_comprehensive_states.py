import cv2
from ultralytics import YOLO
import csv

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

model = YOLO('yolo11n.pt') 

video_path = r"D:\Group 2.mp4" # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Speed Trap lines (Distance MUST be 50m)
lane1_entry = ((1127, 519), (1316, 634)) 
lane1_exit  = ((41, 394), (649, 700)) 
lane2_entry = ((1310, 579), (1454, 660)) 
lane2_exit  = ((671, 497), (1170, 797)) 
TRAP_DISTANCE_METERS = 50.0 

# Threshold for LCV classification
LCV_TRUCK_MAX_AREA_PIXELS = 6000 
LCV_CAR_MIN_AREA_PIXELS = 222000

track_history = {} 
vehicle_data = {}  
counted_ids = set() 
last_entry_time = {'Lane 1': None, 'Lane 2': None} 

class_names = {
    1: 'Two-Wheeler', 
    2: 'Car', 
    3: 'Two-Wheeler', 
    5: 'Bus', 
    7: 'Truck'
}

csv_filename = 'traffic_data_with_flow_states.csv'

# Open the file and KEEP it open by placing the video loop inside this block
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Added columns for Space Headway and Flow State
    writer.writerow(['Timestamp (s)', 'Track_ID', 'Vehicle_Class', 'Lane', 'Speed_kmh', 'Time_Headway_s', 'Space_Headway_m', 'Flow_State'])

    # Create a resizable window before starting the loop
    cv2.namedWindow("50m Speed Trap Extraction", cv2.WINDOW_NORMAL)

    # Force the window to a size that fits on a standard laptop screen (e.g., 720p)
    cv2.resizeWindow("50m Speed Trap Extraction", 1280, 720)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        timestamp_sec = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
        
        results = model.track(
            frame, 
            classes=[1, 2, 3, 5, 7], 
            conf=0.30,      
            imgsz=640,      
            persist=True, 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        
        cv2.line(frame, lane1_entry[0], lane1_entry[1], (0, 255, 0), 2) 
        cv2.line(frame, lane1_exit[0], lane1_exit[1], (0, 0, 255), 2) 
        cv2.line(frame, lane2_entry[0], lane2_entry[1], (0, 255, 0), 2) 
        cv2.line(frame, lane2_exit[0], lane2_exit[1], (0, 0, 255), 2) 
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy() 
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) 
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int) 
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                curr_pt = (cx, cy)
                
                # Calculate bounding box area for LCV logic
                box_area = (x2 - x1) * (y2 - y1)
                
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
                                'time_headway': time_headway,
                                'class': class_names.get(class_id, "Unknown"),
                                'box_area_at_entry': box_area # Store area to evaluate later
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
                                
                                # Hybrid Classification Logic for LCV
                                v_class = vehicle_data[track_id]['class']
                                v_area = vehicle_data[track_id]['box_area_at_entry']
                                if v_class == 'Truck' and v_area < LCV_TRUCK_MAX_AREA_PIXELS:
                                    v_class = 'LCV'
                                # Classification Logic for LCV from Car
                                v_class = vehicle_data[track_id]['class']
                                v_area = vehicle_data[track_id]['box_area_at_entry']
                                if v_class == 'Car' and v_area > LCV_CAR_MIN_AREA_PIXELS:
                                    v_class = 'LCV'
                                
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
                                
                                print(f"Logged -> ID:{track_id} | Class:{v_class} | Spd:{speed_kmh}km/h | Spc_HW:{space_hw_m}m | State:{flow_state}")
                                
                                writer.writerow([timestamp_sec, track_id, v_class, exit_lane, speed_kmh, time_hw_str, space_hw_m, flow_state])
                                counted_ids.add(track_id)

                track_history[track_id] = curr_pt
        
        cv2.imshow("50m Speed Trap Extraction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Data extraction complete. Saved to {csv_filename}")