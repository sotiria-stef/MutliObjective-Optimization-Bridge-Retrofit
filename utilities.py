import os
import cv2
import sys

# GLOBALS
PROJECT_DIR = os.getcwd()


class FolderDoesNotExists(Exception):
    def __init__(self, message='PetalPics folder does not exist! Please run optimum.py first'):
        self.message = message
        super().__init__(self.message)
        sys.exit()


def create_population_video():
    images_folder = os.path.join(PROJECT_DIR, 'PetalPics')
    if not os.path.exists(images_folder):
        raise FolderDoesNotExists

    video_name = 'petal_video.avi'
    images = [img for img in os.listdir(images_folder) if img.endswith('.png')]
    frame = cv2.imread(os.path.join(images_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 10, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(images_folder, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    create_population_video()
