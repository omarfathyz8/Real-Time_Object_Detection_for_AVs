import gradio as gr
import cv2
from ultralytics import YOLO
import tempfile

# Load your 3 YOLO models
models = [
    YOLO("yolo_trained_model.pt"),
    YOLO("car_person_best.pt"),
    YOLO("license best.pt")
]

# Function to detect objects in an image
def detect_on_image(image):
    result_frame = image.copy()
    for model in models:
        results = model(result_frame)
        result_frame = results[0].plot()
    return result_frame

# Function to detect objects in a video
def detect_on_video(video):
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(temp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = frame.copy()
        for model in models:
            results = model(result_frame)
            result_frame = results[0].plot()
        out.write(result_frame)

    cap.release()
    out.release()
    return temp_out.name

# Gradio interfaces
image_interface = gr.Interface(
    fn=detect_on_image,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Image(type="numpy", label="Annotated Image"),
    title="YOLO Image Detection"
)

video_interface = gr.Interface(
    fn=detect_on_video,
    inputs=gr.Video(label="Upload a Video"),
    outputs=gr.Video(label="Processed Video"),
    title="YOLO Video Detection"
)

# Combine both interfaces into tabs
demo = gr.TabbedInterface(
    interface_list=[image_interface, video_interface],
    tab_names=["Image Detection", "Video Detection"]
)

if __name__ == "__main__":
    demo.launch()
