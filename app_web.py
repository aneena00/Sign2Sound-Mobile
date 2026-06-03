import streamlit as st
import os

# Force headless environment variables before anything imports
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import torch
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Direct absolute imports to completely bypass the mp.solutions attribute bug
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
from model import SignLSTM

st.set_page_config(page_title="Sign2Sound Mobile", layout="centered")
st.title("🤟 Sign2Sound Multimodal Mobile")
st.write("Hold your hand signs steady in front of the camera stream.")

# Stable Model Loader
@st.cache_resource
def load_model():
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    model = SignLSTM(num_classes=len(labels))
    if os.path.exists("isl_model.pth"):
        model.load_state_dict(torch.load("isl_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model, labels

try:
    model, labels = load_model()
    st.success("🤖 SignLSTM Model Weights Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model layout: {e}")

# Universal ICE Server setup so it connects properly over mobile data/ambient Wi-Fi
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize the tracker directly from the explicit class path
        self.detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sequence = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Natural mirror flip
        
        # Privacy Mode: Create a pure black canvas matching the webcam frame size
        black_canvas = np.zeros_like(img)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img_rgb)
        
        features = np.zeros(126)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label
                start_offset = 0 if hand_type == "Left" else 63
                
                temp_hand = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark):
                    temp_hand[j, 0] = lm.x
                    temp_hand[j, 1] = lm.y
                    temp_hand[j, 2] = lm.z
                
                # Draw the skeleton dots onto the privacy black canvas ONLY
                mp_drawing.draw_landmarks(black_canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Wrist tracking normalization calculation logic
                wrist = temp_hand[0, :].copy()
                temp_hand -= wrist
                features[start_offset : start_offset + 63] = temp_hand.flatten()
                
            self.sequence.append(features)
            self.sequence = self.sequence[-30:] # Target sequence frame cap
            
            if len(self.sequence) == 30:
                with torch.no_grad():
                    input_tensor = torch.from_numpy(np.array([self.sequence])).float()
                    outputs = model(input_tensor)
                    prob = torch.nn.functional.softmax(outputs, dim=1)
                    max_prob, idx = torch.max(prob, dim=1)
                    
                    if max_prob.item() >= 0.75:
                        st.session_state["live_prediction"] = f"{labels[idx.item()]} ({max_prob.item()*100:.0f}%)"
        else:
            self.sequence = []
            
        return black_canvas

# Live Stream view element setup
ctx = webrtc_streamer(
    key="sign-stream",
    video_transformer_factory=SignLanguageTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.write("---")
st.subheader("Predicted Output:")
prediction_placeholder = st.empty()

if "live_prediction" in st.session_state:
    prediction_placeholder.metric(label="Predicted Sign Language Letter", value=st.session_state["live_prediction"])
else:
    prediction_placeholder.info("Start the WebRTC camera stream above to display text.")
