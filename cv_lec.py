import numpy as np
import cv2
import ultralytics

model_cls = ultralytics.YOLO('yolo26n-cls.pt')
model_det = ultralytics.YOLO('yolo26n.pt')

## original
rocky = cv2.imread('rocky.png')
(frame_height, frame_width) = rocky.shape[:2]
cv2.imshow('Rocky', rocky)

## grayscale
rocky_gray = cv2.cvtColor(rocky, cv2.COLOR_BGR2GRAY)
cv2.imshow('Rocky Gray', rocky_gray)
cv2.imwrite('rocky_gray.png', rocky_gray)

## Gaussian blur
rocky_blur = cv2.GaussianBlur(rocky, (5, 5), 0)
cv2.imshow('Rocky Blur', rocky_blur)
cv2.imwrite('rocky_blur.png', rocky_blur)

## extract middle of image
rocky_crop = rocky[frame_height//4:frame_height*3//4-50, frame_width//4+50:frame_width*3//4]
cv2.imshow('Rocky Crop', rocky_crop)
cv2.imwrite('rocky_crop.png', rocky_crop)

## Canny edge detection
rocky_edges = cv2.Canny(rocky_blur, 100, 200)
cv2.imshow('Rocky Edges', rocky_edges)
cv2.imwrite('rocky_edges.png', rocky_edges)

## contour detection
rocky_gray_blur = cv2.GaussianBlur(rocky_gray, (5, 5), 0)
rocky_thresh = cv2.threshold(rocky_gray_blur, 127, 255, cv2.THRESH_BINARY)[1]
contours, _ = cv2.findContours(rocky_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rocky_contours = rocky.copy()
cv2.drawContours(rocky_contours, contours, -1, (0, 255, 0), 2)
cv2.imshow('Rocky Contours', rocky_contours)
cv2.imwrite('rocky_contours.png', rocky_contours)

## color segmentation
rocky_hsv = cv2.cvtColor(rocky, cv2.COLOR_BGR2HSV)
lower_beige = np.array([10, 50, 50])
upper_beige = np.array([30, 255, 255])
mask_beige = cv2.inRange(rocky_hsv, lower_beige, upper_beige)
rocky_beige = cv2.bitwise_and(rocky, rocky, mask=mask_beige)
cv2.imshow('Rocky Beige', rocky_beige)
cv2.imwrite('rocky_beige.png', rocky_beige)

## image classification with YOLO
results = model_cls(rocky)
for r in results:
    top1_id = r.probs.top1
    confidence = r.probs.top1conf.numpy()
    label = r.names[top1_id]
    print(f"Predicted class: {label}, Confidence: {confidence:.2f}")

## object detection with YOLO
rocky_det = cv2.imread('rocky.png')
results = model_det(rocky_det)
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
        confidence = box.conf[0].numpy()
        class_id = int(box.cls[0].numpy())
        label = r.names[class_id]
        print(f"Detected object: {label}, Confidence: {confidence:.2f}")
        cv2.rectangle(rocky_det, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rocky_det, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imshow('Rocky Detection', rocky_det)
cv2.imwrite('rocky_detection.png', rocky_det)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

## detect orange object using color segmentation
simple_example = cv2.imread('simple_example.png')
simple_example_hsv = cv2.cvtColor(simple_example, cv2.COLOR_BGR2HSV)
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])
mask_orange = cv2.inRange(simple_example_hsv, lower_orange, upper_orange)
orange_objects = cv2.bitwise_and(simple_example, simple_example, mask=mask_orange)
cv2.imshow('Orange Objects', orange_objects)
cv2.imwrite('orange_objects.png', orange_objects)
contours, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(orange_objects, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(orange_objects, "Orange Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imshow('Orange Objects with Contours', orange_objects)
cv2.imwrite('orange_objects_with_contours.png', orange_objects)


## detect clownfish using YOLO
clownfish_example = cv2.imread('simple_example.png')
results = model_det(clownfish_example)
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
        confidence = box.conf[0].numpy()
        class_id = int(box.cls[0].numpy())
        label = r.names[class_id]
        print(f"Detected object: {label}, Confidence: {confidence:.2f}")
        cv2.rectangle(clownfish_example, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(clownfish_example, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.imshow('Clownfish Detection', clownfish_example)
cv2.imwrite('clownfish_detection.png', clownfish_example)

clownfish_example_low_conf = cv2.imread('simple_example.png')
results = model_det(clownfish_example_low_conf, conf=0.1)
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
        confidence = box.conf[0].numpy()
        class_id = int(box.cls[0].numpy())
        label = r.names[class_id]
        print(f"Detected object: {label}, Confidence: {confidence:.2f}")
        cv2.rectangle(clownfish_example_low_conf, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(clownfish_example_low_conf, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.imshow('Clownfish Detection Low Conf', clownfish_example_low_conf)
cv2.imwrite('clownfish_detection_lowconf.png', clownfish_example_low_conf)

cv2.waitKey(0)
cv2.destroyAllWindows()