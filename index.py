import cv2
import numpy as np

# Load the image
import cv2
import numpy as np

def detect_traffic_signs(image):
    """Detects traffic signs in an image and draws rectangles around them.

    Args:
        image: The input image as a NumPy array.

    Returns:
        A list of detected traffic signs, each represented as a NumPy array.
    """

    # Preprocess the image to enhance contrast and remove noise (optional)
    image = cv2.equalizeHist(image)  # Improve contrast
    image = cv2.bilateralFilter(image, 9, 75, 75)  # Reduce noise

    # Find regions of interest based on color and shape
    regions = find_regions_by_color_and_shape(image)

    # Extract and identify traffic signs from regions
    signs = []
    for region in regions:
        sign = extract_traffic_sign(region)
        signs.append(sign)

    # Draw rectangles around detected signs
    for i, sign in enumerate(signs):
        color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Alternate colors for clarity
        cv2.rectangle(image, (sign[0], sign[1]), (sign[0] + sign[2], sign[1] + sign[3]), color, 2)

    return signs

def find_regions_by_color_and_shape(image):
    """Finds regions of interest that are likely to contain traffic signs.

    Args:
        image: The input image as a NumPy array.

    Returns:
        A list of regions of interest, each represented as a NumPy array.
    """

    # Threshold based on colors commonly used in traffic signs
    red_mask = cv2.inRange(image, (0, 0, 150), (100, 100, 255))
    blue_mask = cv2.inRange(image, (150, 0, 0), (255, 100, 100))
    yellow_mask = cv2.inRange(image, (0, 150, 150), (100, 255, 255))
    mask = cv2.bitwise_or(red_mask, blue_mask)
    mask = cv2.bitwise_or(mask, yellow_mask)

    # Find contours of potential signs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on shape and size
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 0.5 < aspect_ratio < 2 and 50 < w * h < 5000:  # Adjust thresholds as needed
            regions.append((x, y, w, h))

    return regions

def extract_traffic_sign(region):
    """Extracts the traffic sign from a region of interest.

    Args:
        region: A tuple representing the region of interest (x, y, w, h).

    Returns:
        The extracted traffic sign as a NumPy array.
    """

    x, y, w, h = region
    sign = image[y:y+h, x:x+w]
    return sign

image = cv2.imread('./data/cam-queo-trai-va-quanh-dau.jpg', cv2.IMREAD_GRAYSCALE)

# Detect traffic signs
signs = detect_traffic_signs(image)

# Display the image with detected signs
cv2.imshow("Traffic Sign Detection", image)
cv2.waitKey(0)
