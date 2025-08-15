from flask import Flask, jsonify
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import base64
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # السماح لكل origins

# تحميل النموذج
interpreter = tf.lite.Interpreter(model_path="sign_language_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# تحميل Scaler و LabelEncoder
scaler = joblib.load("scaler.save")
label_encoder = joblib.load("label_encoder.pkl")

id_to_word = {
    1: "اتنين", 2: "احبك", 3: "اربعه", 4: "الاب", 5: "الام", 6: "بعيد", 7: "هاتف", 8: "تلاتة",
    9: "جدة", 10: "حب", 11: "خارج", 12: "جمسة", 13: "داخل", 14: "سبعة", 15: "ستة", 16: "شكرا",
    17: "صديق", 18: "صفر", 19: "صم", 20: "طفل", 21: "قريب", 22: "مدرسة", 23: "مرحبا", 24: "مش موفق",
    25: "من فضلك", 26: "موافق", 27: "هنا", 28: "هناك", 29: "واحد", 30: "يسار", 31: "يمين",32: "امطار",
    33: "رمال او صحراء",
    34: "كاس",
    35: "مخدة",
    36: "وادي",
    37: "شوكة",
    38: "صم",
    39: "طبق",
    40: "كاميرا",
    41: "كهرباء",
    42: "حفلة",
    43: "مطبخ",
    44: "نهر",
    45: "منشفة",
    46: "معلقة",
    47: "سلك كهرباء",
    48: "سكينة",
    49: "صفر",
    50: "وحد",
    51: "اتنين",
    52: "تلاتة",
    53: "اربعة",
    54: "خمسة",
    55: "ستة",
    56: "سبعة",
    57: "ثمانية",
    58: "تسعة",
    59: "مرتفع",
    60: "الشتاء",
    61: "الصيف",
    62: "تجميد",
    63: "كلية العلوم",
    64: "كل شي",
    65: "ربما",
    66: "امتحان",
    67: "ناجح",
    68: "راسب",
    69: "شعاع",
    70: "مثلث",
    71: "ضرب",
    72: "جمع",
    73: "اصغر من",
    74: "اكبر من",
    75: "الحمدالله",
    76: "الملاكمة",
    77: "عضلة",
    78: "جمجمة",
    79: "قفص صدري",
    80: "قصبة هوائية",
    81: "جهاز تنفسي",
    82: "كبد",
    83: "قلب",
    84: "جهز عصبي",
    85: "فحص نظر",
    86: "التهاب اسنان",
    87: "صداع",
    88: "حساسية",
    89: "مرهم",
    90: "اعاقة بصرية",
    91: "اهلا وسهلا",
    92: "تعا اشرب شاي",
    93: "كوخ يا حبيبي",
    94: "ابدا",
    95: "الم",
    96: "بيت",
    97: "اقف بروح امك",
    98: "هضمي",
    99: "قسمة",
    100: "كلية هندسة",
    101: "معهد تمريض",
    102: "زواج"

}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@socketio.on('predict_image')
def handle_prediction(data):
    try:
        img_b64 = data.get("image")
        frame = decode_base64_image(img_b64)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        data_aux = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks[:2]:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                z_ = [lm.z for lm in hand_landmarks.landmark]
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x / max(x_))
                    data_aux.append(lm.y / max(y_))
                    data_aux.append(lm.z / max(z_))

        while len(data_aux) < 126:
            data_aux.append(0.0)

        if len(data_aux) == 126:
            input_data = scaler.transform([np.asarray(data_aux)]).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            predicted_index = int(np.argmax(output_data[0]))
            confidence = float(np.max(output_data[0]))

            if confidence > 0.5:
                predicted_label = label_encoder.inverse_transform([predicted_index])[0]
                text = id_to_word.get(int(predicted_label), predicted_label)
                socketio.emit('prediction_result', {"prediction": text, "confidence": confidence})
                return

        socketio.emit('prediction_result', {"prediction": None, "confidence": 0.0})
    except Exception as e:
        socketio.emit('prediction_result', {"error": str(e)})

@app.route('/')
def index():
    return " WebSocket API جاهز! أرسل حدث 'predict_image' مع Base64."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)



