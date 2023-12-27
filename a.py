import cv2

def detect_traffic_signs(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.001 * perimeter, True)
        cv2.drawContours(image, [approx], 0, (0,255,0), 2)
    return image

for i in ['bien-bao-cam-1.png','cam-queo-trai-va-quanh-dau.jpg','cam-xe-may.jpg','di-thang.jpg','qua-duong.png','queo-trai.jpg','stop.png']:
    image = cv2.imread('./data/'+i)
    image = detect_traffic_signs(image)
    cv2.imshow('Detected Signs', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
