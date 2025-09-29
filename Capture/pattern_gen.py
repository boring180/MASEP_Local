import cv2
from settings_loader import settings

MARGIN_PX = 100                       # Margins size (in pixels)
A4_dim = (0.210, 0.297)
SQUARE_LENGTH = settings.pattern_square_size  # Square size from settings
MARKER_LENGTH = settings.marker_size           # Marker size from settings

# Calculate image size in pixels (convert from meters to reasonable pixel size)
# Using a scale factor to get a good resolution image
SCALE_FACTOR = 5000  # pixels per meter
IMG_SIZE = (int(settings.pattern_size[0] * SQUARE_LENGTH * SCALE_FACTOR + 2 * MARGIN_PX), 
            int(settings.pattern_size[1] * SQUARE_LENGTH * SCALE_FACTOR + 2 * MARGIN_PX))

OUTPUT_NAME = f'ChArUco_Marker_{settings.pattern_size[0]}_{settings.pattern_size[1]}_DICT_5X5_100.png'

def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(settings.aruco_dict)
    board = cv2.aruco.CharucoBoard((settings.pattern_size[0], settings.pattern_size[1]), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    img = board.generateImage(IMG_SIZE, marginSize=MARGIN_PX)
    img = cv2.resize(img, (int(A4_dim[0] * SCALE_FACTOR), int(A4_dim[1] * SCALE_FACTOR)))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 0, 0)
    (text_width, text_height), baseline = cv2.getTextSize(OUTPUT_NAME, font, font_scale, thickness)
    cv2.putText(img, OUTPUT_NAME, (10, 75), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_NAME, img)
    print(f"ChArUco board saved as: {OUTPUT_NAME}")
    print(f"Board size: {settings.pattern_size[0]}x{settings.pattern_size[1]}")
    print(f"Square size: {SQUARE_LENGTH}m")
    print(f"Marker size: {MARKER_LENGTH}m") 
    print(f"Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
    print(f"Dictionary: DICT_5X5_100")

if __name__ == "__main__":
    create_and_save_new_board()