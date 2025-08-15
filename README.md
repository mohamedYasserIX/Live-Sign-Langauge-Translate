# Live-Sign-Langauge-Translate

# 🖐️ Real-Time Sign Language Recognition API (Flask + Socket.IO + Mediapipe + TFLite)

مشروع للتعرف على لغة الإشارة في الوقت الحقيقي باستخدام:
- **Flask** + **Flask-SocketIO** لاستقبال الصور عبر WebSocket.
- **Mediapipe Hands** لاستخراج نقاط اليد (Landmarks).
- **TensorFlow Lite** لتشغيل نموذج التعرف على الإشارة.
- **Scaler** و **LabelEncoder** لمعالجة البيانات وتحويل التوقعات إلى نصوص عربية.

---

## 📌 المميزات
- استقبال صور Base64 مباشرة عبر WebSocket.
- معالجة الإطار باستخدام **Mediapipe** لاستخراج 21 Landmark لليد.
- تطبيع البيانات (Normalization) باستخدام Scaler.
- تشغيل نموذج TFLite لعمل التنبؤات.
- إرسال النتيجة الفورية مع درجة الثقة (Confidence).

---

## 📷 شكل الـ Hand Landmarks من Mediapipe

Mediapipe بيرجع 21 Landmark لليد الواحدة بالشكل التالي:  

![Mediapipe Hand Landmarks](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

**ترقيم النقاط:**
- المعصم (Wrist) → 0
- الإبهام (Thumb) → 1–4
- السبابة (Index) → 5–8
- الوسطى (Middle) → 9–12
- البنصر (Ring) → 13–16
- الخنصر (Pinky) → 17–20

---

## 📦 المتطلبات

```bash
pip install flask flask-socketio opencv-python mediapipe numpy tensorflow joblib pillow
