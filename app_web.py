import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import torch
import numpy as np
import os
import mediapipe as mp
from model import SignLSTM

# 1. Page Configuration for crisp mobile viewing
st.set_page_config(page_title="Sign2Sound Mobile", layout="centered")
st.title("🤟 Sign2Sound Multimodal Mobile")
st.write("Hold your signs steady in front of the camera to translate them to text.")

# 2. Cached Model Loader (Prevents the app from reloading the model on every frame)
@st.cache_resource
def load_model():
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    model = SignLSTM(num_classes=len(labels))
    
    # Expects 'isl_model.pth' to be right next to this script in your GitHub repo
    if os.path.exists("isl_model.pth"):
        model.load_state_dict(torch.load("isl_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model, labels

try:
    model, labels = load_model()
    st.success("🤖 SignLSTM Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. Global Configuration for WebRTC Connection (Uses Google's free public STUN server)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 4. Video Processing Class (The Core Brain)
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize MediaPipe Tasks Hand Landmarker
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Download asset locally on the cloud server if it doesn't exist
        self.model_path = "hand_landmarker.task"
        if not os.path.exists(self.model_path):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2
        )
        self.detector = HandLandmarker.create_from_options(options)
        self.sequence = []

    def transform(self, frame):
        # Convert incoming WebRTC browser frame to standard OpenCV BGR array
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip for intuitive mirror view
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        hands_result = self.detector.detect(mp_image)
        features = np.zeros(126)
        
        if hands_result.hand_landmarks and hands_result.handedness:
            for hand_landmarks, hand_info in zip(hands_result.hand_landmarks, hands_result.handedness):
                hand_type = hand_info[0].category_name
                
                # Strict alignment matching our verified Holistic dataset structure
                start_offset = 0 if hand_type == "Left" else 63
                
                temp_hand = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks):
                    temp_hand[j, 0] = lm.x
                    temp_hand[j, 1] = lm.y
                    temp_hand[j, 2] = lm.z
                    
                    # Draw visual skeleton dots overlay on the web stream preview
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
                
                # Wrist Normalization Calculation
                wrist = temp_hand[0, :].copy()
                temp_hand -= wrist
                features[start_offset : start_offset + 63] = temp_hand.flatten()
                
            # Manage running 30-frame window queue for the LSTM input layer
            self.sequence.append(features)
            self.sequence = self.sequence[-30:]
            
            if len(self.sequence) == 30:
                with torch.no_grad():
                    input_tensor = torch.from_numpy(np.array([self.sequence])).float()
                    outputs = model(input_tensor)
                    prob = torch.nn.functional.softmax(outputs, dim=1)
                    max_prob, idx = torch.max(prob, dim=1)
                    
                    if max_prob.item() >= 0.75:
                        predicted_letter = labels[idx.item()]
                        # Set a session state variable to display text outside the video thread
                        st.session_state["live_prediction"] = f"{predicted_letter} ({max_prob.item()*100:.0f}%)"
        else:
            self.sequence = []
            
        return img

# 5. Render Web Camera Element
ctx = webrtc_streamer(
    key="sign-stream",
    video_transformer_factory=SignLanguageTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# 6. Responsive UI Output Text Elements Below Video Box
st.write("---")
st.subheader("Live Output:")
prediction_placeholder = st.empty()

if "live_prediction" in st.session_state:
    prediction_placeholder.metric(label="Predicted Sign Language Letter", value=st.session_state["live_prediction"])
else:
    prediction_placeholder.info("Start the camera stream above and place your hands in view.")
