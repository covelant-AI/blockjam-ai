from ai_core.letr_court_line_detector.core import LETRCourtDetector
from ai_core.court_line_detector.core import CourtLineDetector
import os
import cv2


def court_line_detection():
    images_dir = "input_images/extracted_frames"
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    court_line_detector = CourtLineDetector("models/model_tennis_court_det.pt", "cuda")
    letr_court_line_detector = LETRCourtDetector("models/letr_best_checkpoint.pth", "cuda")

    # images = [
    #     "2cvzTSuDRy0_middle_frame.jpg",
    #     "6IEoC0d_uOY_middle_frame.jpg"
    # ]

    os.makedirs(f"output_images/court_keypoints", exist_ok=True)
    os.makedirs(f"output_images/court_keypoints/letr", exist_ok=True)
    os.makedirs(f"output_images/court_keypoints/failed", exist_ok=True)

    failed_images = []
    for image_name in images:
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (1280, 720))
        try:
            keypoints = court_line_detector.detect(image)
            image = court_line_detector.draw_keypoints(image, keypoints)
        except Exception as e:
            try:
                keypoints = letr_court_line_detector.detect(image)
                image = court_line_detector.draw_keypoints(image, keypoints)
                image_name = f"letr/{image_name}.jpg"
            except Exception as e:
                failed_images.append(image_name)
                print(f"LETR court line detection failed for image {image_name}: {e}")
                image_name = f"failed/{image_name}"
        finally:
            cv2.imwrite(f"output_images/court_keypoints/{image_name}", image)
    
    print(f"Failed images: {failed_images}")

if __name__ == "__main__":
    court_line_detection()