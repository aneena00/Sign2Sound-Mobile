import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import torch
import numpy as np
import os
import mediapipe as mp
from model import SignLSTM

# 1. Page Configuration for Crisp Mobile Layout
st.set_page_config(page_title="Sign2Sound Mobile", layout="centered")
st.title("🤟 Sign2Sound Multimodal Mobile")
st.write("Hold your signs steady in front of the camera to translate them to text.")

# 2. Stable Model Loader
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
    st.success("🤖 SignLSTM Model Loaded Successfully on Python 3.12!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. WebRTC ICE Configuration (Uses Google's Free Stun Network for Mobile Connections)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 4. Video Processing Class (Standard Python 3.12 Solutions API)
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.sequence = []

    def transform(self, frame):
        # Convert incoming browser stream to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror reflection for intuitive usage
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img_rgb)
        
        features = np.zeros(126)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label
                
                # Strict 126-point spatial array mapping (Left=0-62, Right=63-125)
                start_offset = 0 if hand_type == "Left" else 63
                
                temp_hand = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark):
                    temp_hand[j, 0] = lm.x
                    temp_hand[j, 1] = lm.y
                    temp_hand[j, 2] = lm.z
                
                # Draw skeleton landmarks over the video stream
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Apply Your Dataset's Wrist Coordinate-Normalization
                wrist = temp_hand[0, :].copy()
                temp_hand -= wrist
                features[start_offset : start_offset + 63] = temp_hand.flatten()
                
            # Manage running 30-frame calculation window queue
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
                        st.session_state["live_prediction"] = f"{predicted_letter} ({max_prob.item()*100:.0f}%)"
        else:
            self.sequence = []
            
        return img

# 5. Live Stream Visual Window Elements
ctx = webrtc_streamer(
    key="sign-stream",
    video_transformer_factory=SignLanguageTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# 6. Responsive Output HUD Interface Elements
st.write("---")
st.subheader("Live Output:")
prediction_placeholder = st.empty()

if "live_prediction" in st.session_state:
    prediction_placeholder.metric(label="Predicted Sign Language Letter", value=st.session_state["live_prediction"])
else:
    prediction_placeholder.info("Start the camera stream above and place your hands in view.")
