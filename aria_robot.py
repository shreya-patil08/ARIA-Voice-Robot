"""
ARIA Service Robot — Unified Control System
==========================================
Raspberry Pi 5 | L298N + PCA9685 | HC-SR04 x2 | USB Camera

Threads:
    1. Flask web server       (port 5000)
    2. Motor control loop     (100ms cycle, obstacle-aware)
    3. Voice loop             (STT → motor/arm/LLM → TTS)
    4. Camera/vision loop     (OpenCV color detection → MJPEG)
"""

# —— IMPORTS ————————————————————————————
import os
import signal
import time
import threading
import ctypes
import cv2
import numpy as np
import RPi.GPIO as GPIO
import speech_recognition as sr
from google import genai
from gtts import gTTS
from flask import Flask, render_template, Response, request, jsonify
from adafruit_servokit import ServoKit
from dotenv import load_dotenv

load_dotenv()

# —— HARDWARE CONSTANTS — GPIO PINS ————————————
# L298N Motor Pins (BCM numbering)
L_IN1, L_IN2, L_ENA = 17, 27, 18    # Left motor pair
R_IN3, R_IN4, R_ENB = 22, 23, 19    # Right motor pair

# HC-SR04 Ultrasonic Sensor Pins
FRONT_TRIG, FRONT_ECHO = 24, 25      # Front sensor
BACK_TRIG,  BACK_ECHO  =  8,  7     # Back sensor

STOP_DISTANCE = 15.0   # cm — obstacle threshold
MOTOR_SPEED   = 75     # PWM duty cycle (%)

# PCA9685 Servo (I2C: SDA=GPIO2, SCL=GPIO3)
NUM_JOINTS = 6          # 6 DOF arm
ARM_STEP   = 10         # degrees per command

# —— GPIO SETUP ————————————————————————————
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

motor_pins = [L_IN1, L_IN2, L_ENA, R_IN3, R_IN4, R_ENB]
sensor_pins_out = [FRONT_TRIG, BACK_TRIG]
sensor_pins_in  = [FRONT_ECHO, BACK_ECHO]

for pin in motor_pins + sensor_pins_out:
    GPIO.setup(pin, GPIO.OUT)

for pin in sensor_pins_in:
    GPIO.setup(pin, GPIO.IN)

pwm_l = GPIO.PWM(L_ENA, 1000)
pwm_r = GPIO.PWM(R_ENB, 1000)
pwm_l.start(0)
pwm_r.start(0)

# Servo kit
kit = ServoKit(channels=16)
arm_angles = [90] * NUM_JOINTS
_lock = threading.Lock()

# Flask app
app = Flask(__name__)

# Shared state
current_cmd   = "stop"
camera_frame  = None
frame_lock    = threading.Lock()

# Gemini client
SYSTEM_PROMPT = (
    "You are ARIA, a friendly AI service robot assistant. "
    "Keep responses short and clear — you are speaking aloud."
)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# —— DISTANCE CALCULATION — HC-SR04 ————————————
def get_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)          # 10µs trigger pulse
    GPIO.output(trig, False)

    start = end = time.time()
    timeout = time.time() + 0.05

    while GPIO.input(echo) == 0:
        start = time.time()
        if start > timeout:
            return 400.0

    while GPIO.input(echo) == 1:
        end = time.time()
        if end > timeout:
            return 400.0

    # Distance = echo_time × 17150  (speed of sound / 2)
    return round((end - start) * 17150, 2)


# —— MOTOR CONTROL — L298N ————————————————————
def apply_motor(cmd):
    if cmd == "stop":
        pwm_l.ChangeDutyCycle(0); pwm_r.ChangeDutyCycle(0)
        GPIO.output(L_IN1, False); GPIO.output(L_IN2, False)
        GPIO.output(R_IN3, False); GPIO.output(R_IN4, False)
        return

    if cmd == "forward":
        GPIO.output(L_IN1, GPIO.HIGH); GPIO.output(L_IN2, GPIO.LOW)
        GPIO.output(R_IN3, GPIO.HIGH); GPIO.output(R_IN4, GPIO.LOW)

    elif cmd == "backward":
        GPIO.output(L_IN1, GPIO.LOW);  GPIO.output(L_IN2, GPIO.HIGH)
        GPIO.output(R_IN3, GPIO.LOW);  GPIO.output(R_IN4, GPIO.HIGH)

    elif cmd == "left":
        GPIO.output(L_IN1, GPIO.LOW);  GPIO.output(L_IN2, GPIO.HIGH)
        GPIO.output(R_IN3, GPIO.HIGH); GPIO.output(R_IN4, GPIO.LOW)

    elif cmd == "right":
        GPIO.output(L_IN1, GPIO.HIGH); GPIO.output(L_IN2, GPIO.LOW)
        GPIO.output(R_IN3, GPIO.LOW);  GPIO.output(R_IN4, GPIO.HIGH)

    pwm_l.ChangeDutyCycle(MOTOR_SPEED)
    pwm_r.ChangeDutyCycle(MOTOR_SPEED)


def control_loop():
    """Thread 2 — Motor control with obstacle avoidance (100ms cycle)."""
    global current_cmd
    while True:
        front_dist = get_distance(FRONT_TRIG, FRONT_ECHO)
        back_dist  = get_distance(BACK_TRIG,  BACK_ECHO)

        cmd = current_cmd
        if cmd == "forward"  and front_dist < STOP_DISTANCE:
            cmd = "stop"
        if cmd == "backward" and back_dist  < STOP_DISTANCE:
            cmd = "stop"

        apply_motor(cmd)
        time.sleep(0.1)


# —— SERVO ARM CONTROL — PCA9685 ——————————————
def move_arm_joint(joint, value, absolute=False):
    """Move a single arm joint by relative or absolute degrees."""
    with _lock:
        if absolute:
            new_angle = max(0, min(180, value))
        else:
            new_angle = max(0, min(180, arm_angles[joint] + value))
        arm_angles[joint] = new_angle
        kit.servo[joint].angle = new_angle


# —— CAMERA — COLOUR DETECTION (OpenCV HSV) ————
COLOR_RANGES = {
    "green":  (np.array([35, 60, 60]),   np.array([85, 255, 255]),  (0, 200, 0)),
    "red":    (np.array([0, 120, 70]),    np.array([10, 255, 255]),  (0, 0, 220)),
    "yellow": (np.array([20, 100, 100]),  np.array([30, 255, 255]),  (0, 210, 210)),
    "blue":   (np.array([100, 150, 50]),  np.array([140, 255, 255]), (220, 80, 0)),
}

# Distance estimation — Pinhole Camera Model:
# Distance (cm) = (Real_Width × Focal_Length) / Pixel_Width
REAL_BOX_WIDTH_CM = 30.0
FOCAL_LENGTH_PX   = 600.0


def camera_loop():
    """Thread 4 — OpenCV colour detection + MJPEG stream."""
    global camera_frame
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color, (lower, upper, bgr) in COLOR_RANGES.items():
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 500:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                dist = (REAL_BOX_WIDTH_CM * FOCAL_LENGTH_PX) / w
                label = f"{color} {dist:.1f}cm"
                cv2.rectangle(frame, (x, y), (x+w, y+h), bgr, 2)
                cv2.putText(frame, label, (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)

        with frame_lock:
            _, buf = cv2.imencode(".jpg", frame)
            camera_frame = buf.tobytes()

        time.sleep(0.03)


# —— VOICE PIPELINE — STT → LLM → TTS ————————
def listen():
    """Capture one utterance from microphone and return text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
    return recognizer.recognize_google(audio)


def speak(text):
    """Convert text to speech and play via mpg123."""
    tts = gTTS(text=text, lang="en")
    tts.save("/tmp/aria_voice.mp3")
    os.system("mpg123 -q /tmp/aria_voice.mp3")


def ask_llm(user_input):
    """Send user input to Gemini and return response text."""
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=user_input,
        config={"system_instruction": SYSTEM_PROMPT}
    )
    return response.text.strip()


def voice_loop():
    """Thread 3 — Continuous voice command loop (STT → parse → act → TTS)."""
    global current_cmd
    motion_cmds = {"forward", "backward", "left", "right", "stop"}

    while True:
        try:
            text = listen().lower().strip()
            print(f"[VOICE] Heard: {text}")

            # Motion commands
            for mc in motion_cmds:
                if mc in text:
                    current_cmd = mc
                    speak(f"Moving {mc}")
                    break
            else:
                # Arm commands  e.g. "joint 2 up"
                if "joint" in text:
                    parts = text.split()
                    idx = parts.index("joint")
                    joint  = int(parts[idx + 1]) - 1
                    delta  = ARM_STEP if "up" in text else -ARM_STEP
                    move_arm_joint(joint, delta)
                    speak(f"Joint {joint+1} moved")
                else:
                    # General LLM response
                    reply = ask_llm(text)
                    print(f"[ARIA] {reply}")
                    speak(reply)

        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            print(f"[VOICE ERROR] {e}")


# —— FLASK ROUTES ——————————————————————————
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/control", methods=["POST"])
def control():
    global current_cmd
    data = request.get_json()
    current_cmd = data.get("cmd", "stop")
    return jsonify({"status": "ok", "cmd": current_cmd})


@app.route("/arm", methods=["POST"])
def arm():
    data   = request.get_json()
    joint  = int(data.get("joint", 0))
    value  = int(data.get("value", 0))
    absolute = data.get("absolute", False)
    move_arm_joint(joint, value, absolute)
    return jsonify({"status": "ok", "angles": arm_angles})


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                frame = camera_frame
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    front = get_distance(FRONT_TRIG, FRONT_ECHO)
    back  = get_distance(BACK_TRIG,  BACK_ECHO)
    return jsonify({
        "cmd":         current_cmd,
        "arm_angles":  arm_angles,
        "front_dist":  front,
        "back_dist":   back,
    })


# —— MAIN ENTRY — THREAD LAUNCH ———————————————
if __name__ == "__main__":
    threading.Thread(target=control_loop, daemon=True).start()
    threading.Thread(target=camera_loop,  daemon=True).start()
    threading.Thread(target=voice_loop,   daemon=True).start()
    threading.Thread(target=run_flask,    daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        current_cmd = "stop"
        pwm_l.stop(); pwm_r.stop()
        GPIO.cleanup()
