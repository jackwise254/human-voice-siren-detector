# Import necessary libraries
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
import time
import pyaudio
import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa
import json
from audio_utils import SpecViewer
spec_viewer = SpecViewer()
from kivy.graphics import Rectangle
from huggingface_hub import snapshot_download
snapshot_download('nccratliri/vad-human-ava-speech', local_dir = "human-ava-speech", repo_type="dataset" )
from model import WhisperSegmenterFast
segmenter = WhisperSegmenterFast( "nccratliri/whisperseg-large-ms-ct2", device="cuda" )

class VoiceSirenDetectorApp(App):
    def clear_input(self, instance):
        self.root.ids.additional_input.text = ''

    def submit_input(self, instance):
        input_text = self.root.ids.additional_input.text
        print(f"Submitted Input: {input_text}")

    def build(self):
        self.detector = VoiceSirenDetector()

        # Main layout
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Background image
        with layout.canvas.before:
            Rectangle(pos=layout.pos, size=layout.size, source='assets/background_image2.jpg')

        # Title label
        title_label = Label(text='Human Voice and Siren Detector', font_size=24, size_hint_y=None, height=5, bold=True, color=(1, 1, 1, 1))
        layout.add_widget(title_label)

        # Buttons for start and stop
        button_layout = BoxLayout(orientation='horizontal', spacing=20)
        start_button = Button(text='Start', size_hint_y=None, height=40, font_size=18, background_color=(0.5, 0.9, 0.5, 0.8))
        start_button.bind(on_press=self.detector.start_detection)
        start_button_image = Image(source='assets/start.png', size_hint=(None, None), size=(40, 40), pos_hint={'center_y': 0.5})
        button_layout.add_widget(start_button)
        button_layout.add_widget(start_button_image)

        stop_button = Button(text='Stop', size_hint_y=None, height=40, font_size=18, background_color=(0.9, 0.5, 0.5, 0.8))
        stop_button.bind(on_press=self.detector.stop_detection)
        stop_button_image = Image(source='assets/stop.png', size_hint=(None, None), size=(40, 40), pos_hint={'center_y': 0.5})
        button_layout.add_widget(stop_button)
        button_layout.add_widget(stop_button_image)

        layout.add_widget(button_layout)

        # Analysis Signals label
        analysis_label = Label(text='Analysis Signals', font_size=18, size_hint_y=None, height=40, color=(1, 1, 1, 1))
        layout.add_widget(analysis_label)

        # Visualization area (a widget needed here)

        # TextInput for additional input
        additional_input = TextInput(text='Enter text here', font_size=16, multiline=False,

                                     size_hint_y=None, height=120, background_color=(0.7, 0.7, 0.7, 0.5))
        layout.add_widget(additional_input)

        # Buttons for clear and submit
        button_layout = BoxLayout(orientation='horizontal', spacing=20)
        clear_button = Button(text='Clear', size_hint_y=None, height=60, font_size=18, background_color=(0.9, 0.5, 0.5, 0.8))
        clear_button.bind(on_press=self.clear_input)
        submit_button = Button(text='Submit', size_hint_y=None, height=60, font_size=18, background_color=(0.5, 0.9, 0.5, 0.8))
        submit_button.bind(on_press=self.submit_input)
        button_layout.add_widget(clear_button)
        button_layout.add_widget(submit_button)
        layout.add_widget(button_layout)

        return layout

class VoiceSirenDetector:
    def __init__(self):
        self.is_detecting = False
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.voice_threshold = 0.02
        self.siren_threshold = 0.1
        self.model = LogisticRegression()  # Replace with your chosen model

    def start_detection(self, instance):
        self.is_detecting = True
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024,
                                  stream_callback=self.detect_callback)
        self.stream.start_stream()

    def stop_detection(self, instance):
        self.is_detecting = False
        self.stream.stop_stream()
        self.stream.close()

    def detect_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)

        # Compute the root mean square (RMS) energy of the audio signal
        rms_energy = librosa.feature.rms(y=audio_data)

        # Assuming you have a pre-trained model for voice and siren detection
        # You may replace this part with actual model inference logic
        if self.is_detecting:
            self.process_audio_data(rms_energy)

        return in_data, pyaudio.paContinue

    def process_audio_data(self, rms_energy):
        # Check if the RMS energy exceeds the voice threshold
        if np.mean(rms_energy) > self.voice_threshold:
            print("Voice detected!")
            # You can add logic to handle voice detection here
            self.handle_voice_detection(rms_energy)

        # Check if the RMS energy exceeds the siren threshold
        elif np.mean(rms_energy) > self.siren_threshold:
            print("Siren detected!")
            # You can add logic to handle siren detection here
            self.handle_siren_detection(rms_energy)

    def handle_voice_detection(self, rms_energy):
        # Placeholder logic for handling voice detection
        # You can use this data to collect labeled training samples
        label = 1  # Assuming 1 represents voice
        self.collect_labeled_data(rms_energy, label)

    # Inside handle_voice_detection method
    def handle_voice_detection(self, rms_energy):
        label = 1  # Assuming 1 represents voice
        self.collect_labeled_data(rms_energy, label)


    def human_voice(self):
        sr = 16000  
        min_frequency = 0
        spec_time_step = 0.01
        min_segment_length = 0.1
        eps = 0.2
        num_trials = 1

        audio_file = "data/example_subset/Human_AVA_Speech/test/human_xO4ABy2iOQA_clip.wav"
        label_file = audio_file[:-4] + ".json"
        audio, _ = librosa.load( audio_file, sr = sr )
        label = json.load( open(label_file) )

        prediction = segmenter.segment(  audio, sr = sr, min_frequency = min_frequency, spec_time_step = spec_time_step,
                            min_segment_length = min_segment_length, eps = eps,num_trials = num_trials )
        spec_viewer.visualize( audio = audio, sr = sr, min_frequency= min_frequency, prediction = prediction, label=label, 
                            window_size=20, precision_bits=0, xticks_step_size = 2 )


    def collect_labeled_data(self, rms_energy, label):
        # Placeholder for collecting labeled training data
        # You may want to store the features and labels in a data structure
        # Here, we'll just print the label and RMS energy for demonstration
        print(f"Label: {label}, RMS Energy: {np.mean(rms_energy)}")
        # Further data storage and preprocessing logic can be added here

    def train_model(self, features, labels):
        # Placeholder for model training logic
        # You need to replace this with your actual model training code
        # Here, we use a simple logistic regression model as an example
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy}")
        # You would typically save the trained model for later use

if __name__ == '__main__':
    VoiceSirenDetectorApp().run()
    detector = VoiceSirenDetector()
    detector.start_detection(None)  # You can pass any instance or None
    time.sleep(5)  # Let it run for 5 seconds
    detector.stop_detection(None)

    # Placeholder data for training (replace with actual data collection logic)
    features = np.random.rand(100, 1)  # Replace with actual feature vectors
    labels = np.random.randint(2, size=100)  # Replace with actual labelsi

    detector.train_model(features, labels)

    # detector.close_stream() 