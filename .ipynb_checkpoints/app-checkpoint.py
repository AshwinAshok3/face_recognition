import streamlit as st
import cv2
import numpy as np
import tempfile
import insightface
from numpy.linalg import norm
import time

# ==========================================
# 1️⃣ Initialize Model (Auto GPU/CPU fallback)
# ==========================================
@st.cache_resource
def load_model():
    try:
        app = insightface.app.FaceAnalysis(name='buffalo_l',
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        st.success("✅ Model loaded with GPU support.")
    except Exception as e:
        st.warning(f"⚠️ GPU unavailable, switching to CPU mode.\n{e}")
        app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        st.info("✅ Running on CPU.")
    return app

app = load_model()

# ==========================================
# 2️⃣ Utility Functions
# ==========================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def find_best_match(face_embedding, target_embedding, threshold=0.35):
    score = cosine_similarity(face_embedding, target_embedding)
    if score > threshold:
        return True, score
    return False, score

def process_frame(frame, target_embedding, threshold=0.35):
    faces = app.get(frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        is_match, score = find_best_match(embedding, target_embedding, threshold)

        color = (0, 255, 0) if is_match else (0, 0, 255)
        label = f"Match ({score:.2f})" if is_match else f"Unknown ({score:.2f})"

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ==========================================
# 3️⃣ Streamlit UI
# ==========================================
st.title("🧠 Real-Time Face Recognition System (InsightFace)")
st.markdown("Detect whether a target person appears in a video or webcam feed.")

upload_img = st.file_uploader("📷 Upload target image (JPG/PNG):", type=["jpg", "jpeg", "png"])
upload_vid = st.file_uploader("🎥 Upload video (.mp4) [Optional if using webcam]:", type=["mp4"])
use_webcam = st.checkbox("Use Webcam Instead of Video", value=False)

if upload_img is not None:
    # Save and load target face
    tmp_img = tempfile.NamedTemporaryFile(delete=False)
    tmp_img.write(upload_img.read())
    image = cv2.imread(tmp_img.name)
    faces = app.get(image)
    if len(faces) == 0:
        st.error("❌ No face detected in uploaded image.")
    else:
        target_face = faces[0]
        target_embedding = target_face.embedding
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Target Face Detected ✅")

        threshold = st.slider("Adjust Matching Threshold", 0.2, 0.6, 0.35, 0.01)
        start_btn = st.button("🚀 Start Detection")

        if start_btn:
            if use_webcam:
                st.info("Starting webcam feed... Press Stop to end.")
                cap = cv2.VideoCapture(0)
            elif upload_vid is not None:
                tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_vid.write(upload_vid.read())
                cap = cv2.VideoCapture(tmp_vid.name)
            else:
                st.error("❌ Please upload a video or enable webcam.")
                st.stop()

            frame_window = st.empty()
            match_detected = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = process_frame(frame, target_embedding, threshold)
                frame_window.image(processed, channels="RGB")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            st.success("✅ Detection completed successfully.")
