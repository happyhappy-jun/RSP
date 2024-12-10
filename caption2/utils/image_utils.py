import cv2

def resize_image(image, max_size=512):
    """Resize image to fit within max_size while maintaining aspect ratio"""
    height, width = image.shape[:2]
    if height > width:
        if height > max_size:
            ratio = max_size / height
            new_height = max_size
            new_width = int(width * ratio)
        else:
            return image
    else:
        if width > max_size:
            ratio = max_size / width
            new_width = max_size
            new_height = int(height * ratio)
        else:
            return image
            
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
