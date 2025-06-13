import cv2

def draw_box(image_rgb, box):
    x, y, w, h = box
    image_copy = image_rgb.copy()
    cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return image_copy
