<!DOCTYPE html>
<html>
<body>
<video id="video" width="320" height="240" autoplay></video>
<script>
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    document.getElementById('video').srcObject = stream;

    setInterval(() => {
      const canvas = document.createElement('canvas');
      const video = document.getElementById('video');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      const base64 = canvas.toDataURL('image/jpeg').split(',')[1];

      fetch("http://192.168.1.7:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ image: base64 })
      })
      .then(res => res.json())
      .then(data => {
        console.log("👋 Prediction:", data);
      });
    }, 3000); // كل 3 ثواني
  });
</script>
</body>
</html>
