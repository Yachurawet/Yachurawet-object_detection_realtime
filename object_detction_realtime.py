import torch
import cv2
model = torch.hub.load("ultralytics/yolov5","yolov5x")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        image_rhg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(image_rhg)
        predictions = results.xyxy[0]. cpu(). numpy()
        for pred in predictions:
            x1, y1, x2, y2, score, label = pred
            label = int(label)
            object_name = model.names[label]
            cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (255,0,255), 2)
            cv2.putText(frame, f"{object_name} {score*100:.2f}%", (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        cv2.imshow("object detection", frame)
        if cv2.waitKey(1) == ord('w'):
            break
cap.release()
cv2.destroyAllWindows()

