import os
import uuid
from flask import Flask, render_template, request, Response, send_from_directory
import cv2
from ultralytics import YOLO

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create upload folder if not exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)  # Create results folder if not exists


model_person = YOLO("trash.pt")  

def draw_boxes(frame, results, model, box_color):
    """
    Draw bounding boxes and labels on the frame for detected objects.
    """
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        conf = float(box.conf)                   # Confidence score
        cls_id = int(box.cls)                    # Class ID
        label = model.names[cls_id]              # Class name (e.g., 'person')
        # Draw rectangle around detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        # Put label text above rectangle
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

def gen_frames():
    """
   it  Capture's video from webcam, detect persons, and yield frames with boxes.
    """
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()  
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))  

        
        results_person = model_person(frame_resized, conf=0.5)[0]

        
        draw_boxes(frame_resized, results_person, model_person, (0, 255, 0))

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_resized)
        frame_bytes = buffer.tobytes()

        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()  # Release webcam when done

@app.route('/')
def index():
    """
    Render the main HTML page.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route. Streams frames from webcam with detections.
    """
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle image or video uploads, run person detection, and return results.
    """
    file = request.files.get('file')
    if not file:
        return render_template('index.html', error="No file uploaded")

    # Generate unique filename to avoid conflicts
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)  # Save uploaded file

    ext = filename.rsplit('.', 1)[1].lower()  # Get file extension

    # Process image files
    if ext in ['jpg', 'jpeg', 'png', 'bmp']:
        img = cv2.imread(filepath)
        if img is None:
            return render_template('index.html', error="Invalid image file")

        # Run person detection on image
        results_person = model_person(img, conf=0.5)[0]

        # Draw bounding boxes on image
        draw_boxes(img, results_person, model_person, (0, 255, 0))

        # Save the processed image with detections
        out_filename = filename.rsplit('.', 1)[0] + '_det.' + ext
        out_path = os.path.join(RESULTS_FOLDER, out_filename)
        cv2.imwrite(out_path, img)

        # Render page with detected image
        return render_template('index.html', uploaded_image=out_filename)

    # Process video files
    elif ext in ['mp4', 'avi', 'mov', 'mkv']:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return render_template('index.html', error="Invalid video file")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_filename = filename.rsplit('.', 1)[0] + '_det.mp4'
        out_path = os.path.join(RESULTS_FOLDER, out_filename)
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run person detection on each frame
            results_person = model_person(frame, conf=0.5)[0]

            # Draw bounding boxes on frame
            draw_boxes(frame, results_person, model_person, (0, 255, 0))

            out.write(frame)  # Write frame to output video

        cap.release()
        out.release()

        # Render page with detected video
        return render_template('index.html', uploaded_video=out_filename)

    else:
        return render_template('index.html', error="Unsupported file type")

@app.route('/static/results/<filename>')
def send_result_file(filename):
    """
    Serve processed result files.
    """
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/static/uploads/<filename>')
def send_upload_file(filename):
    """
    Serve uploaded files.
    """
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
