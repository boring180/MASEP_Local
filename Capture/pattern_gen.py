import cv2
from settings_loader import settings


A4_dim = (0.210, 0.297)
SQUARE_LENGTH = settings.pattern_square_size  # Square size from settings
MARKER_LENGTH = settings.marker_size           # Marker size from settings

SCALE_FACTOR = 5000  # pixels per meter
MARGIN_PX = int(SCALE_FACTOR * 0.02) # 2cm margin
IMG_SIZE = (int(settings.pattern_size[0] * SQUARE_LENGTH * SCALE_FACTOR + 2 * MARGIN_PX), 
            int(settings.pattern_size[1] * SQUARE_LENGTH * SCALE_FACTOR + 2 * MARGIN_PX))
DICT = getattr(cv2.aruco, settings.aruco_dict)

OUTPUT_NAME = f'ChArUco_Marker_{settings.pattern_size[0]}_{settings.pattern_size[1]}_{settings.aruco_dict}.png'

def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard((settings.pattern_size[0], settings.pattern_size[1]), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    img = board.generateImage(IMG_SIZE, marginSize=MARGIN_PX, )
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 0, 0)
    (text_width, text_height), baseline = cv2.getTextSize(OUTPUT_NAME, font, font_scale, thickness)
    cv2.putText(img, OUTPUT_NAME, (75, 75), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_NAME, img)

if __name__ == "__main__":
    create_and_save_new_board()