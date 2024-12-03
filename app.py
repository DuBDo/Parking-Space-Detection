import os
from flask import Flask, request, jsonify, send_from_directory, abort, render_template
import cv2
from werkzeug.utils import secure_filename

import numpy as np
from ultralytics import YOLO
import ast

app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath('uploads')  # Absolute path
FRAMES_FOLDER = os.path.abspath('frames')   # Absolute path

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER

ALLOWED_EXTENSIONS = {'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return "No file part in the request.", 400

    video = request.files['video']
    if video.filename == '':
        return "No file selected.", 400

    if allowed_file(video.filename):
        filename = secure_filename(video.filename)
        global filepath 
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)

        # Extract first frame
        video_capture = cv2.VideoCapture(filepath)
        success, frame = video_capture.read()
        if success:
            frame_filename = f"{filename}_frame.jpg"
            frame_path = os.path.join(app.config['FRAMES_FOLDER'], frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"Frame saved at: {frame_path}")
            return jsonify({"frame_url": f"/frames/{frame_filename}"})
        else:
            return "Failed to extract frame.", 500

    return "Invalid file type.", 400

@app.route('/frames/<filename>')
def serve_frame(filename):
    try:
        return send_from_directory(app.config['FRAMES_FOLDER'], filename)
    except FileNotFoundError:
        print(f"Frame not found: {filename}")
        abort(404)


def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Use ast.literal_eval to safely parse single-quoted dictionary strings
    coordinates = [ast.literal_eval(line.strip()) for line in lines]

    # Convert to tuples of (x, y)
    points = [(int(coord['x']), int(coord['y'])) for coord in coordinates]

    # Group points into sets of four consecutive points
    grouped_polygons = [points[i:i + 4] for i in range(0, len(points), 4)]

    # Ensure all groups have exactly four points (skip incomplete groups)
    grouped_polygons = [group for group in grouped_polygons if len(group) == 4]

    return grouped_polygons

@app.route('/save-coordinates', methods=['POST'])
def save_coordinates():
    data = request.json
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'coordinates.txt')
    with open(filepath, 'w') as file:
        for coord in data.get('coordinates', []):
            file.write(f"{coord}\n")
    return jsonify({"message": "Coordinates saved successfully."})

@app.route('/detect', methods=['POST'])
def run_function():
        # Example logic to execute
        # Load parking areas from the coordinate file
    coordinate_file =  os.path.join(app.config['UPLOAD_FOLDER'], 'coordinates.txt') # Replace with your file name
    areas = read_coordinates(coordinate_file)

    # Load YOLO model and class labels
    model = YOLO('yolov8s.pt')
    coco_path = os.path.join(os.path.dirname(__file__), 'coco.txt')
    with open(coco_path, "r") as f:
        class_list = f.read().splitlines()

    # Function to check which area contains the car
    def check_parking_areas(cx, cy, areas):
        for i, area in enumerate(areas):
            if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                return i
        return -1

    # Process video feed
    cap = cv2.VideoCapture(filepath)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020,573))

        # Run YOLO inference
        results = model.predict(frame, stream=True)
        space_count = [0] * len(areas)

        for detection in results:
            for box in detection.boxes:
                x1, y1, x2, y2, conf, cls = map(int, box.data[0])
                class_name = class_list[cls]
                if "car" in class_name:
                    # Calculate center of bounding box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Check if the car is in a parking area
                    area_index = check_parking_areas(cx, cy, areas)
                    if area_index != -1:
                        space_count[area_index] += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw parking areas
        for i, area in enumerate(areas):
            color = (0, 255, 0) if space_count[i] == 0 else (0, 0, 255)
            cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
            cv2.putText(frame, f"{i + 1}", tuple(area[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display available spaces
        total_spaces = len(areas)
        available_spaces = total_spaces - sum(space_count)
        
        cv2.putText(frame, f"Available Spaces: {available_spaces}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("Parking Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    return jsonify({
        "available_spaces": available_spaces,
        "total_spaces": total_spaces
    })


@app.route('/delete-coordinates', methods=['POST'])
def delete_coordinates():
    coordinate_file = os.path.join(app.config['UPLOAD_FOLDER'], 'coordinates.txt')
    if os.path.exists(coordinate_file):
        os.remove(coordinate_file)
        return jsonify({"message": "Coordinates file deleted successfully!", "success": True})
    else:
        return jsonify({"message": "Coordinates file does not exist.", "success": False})


if __name__ == "__main__":
    #print(f"Uploads folder: {UPLOAD_FOLDER}")
    #print(f"Frames folder: {FRAMES_FOLDER}")
    app.run(debug=True)
