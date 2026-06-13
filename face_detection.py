import cv2
import os
from mtcnn import MTCNN

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_IMAGE_SIDE = 1024

dataset_path = r"E:\Frames"
output_path = r"E:\anotated_images"
detector = MTCNN()

os.makedirs(output_path, exist_ok=True)

def load_and_resize_image(img_path, max_side=MAX_IMAGE_SIDE):
    img = cv2.imread(img_path)
    if img is None:
        return None

    height, width = img.shape[:2]
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        img = cv2.resize(
            img,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return img

for category in os.listdir(dataset_path):
    if category != 'original':
        continue
    category_path = os.path.join(dataset_path, category)

    if not os.path.isdir(category_path):
        continue

    print(f"Processing Category: {category}")

    category_output = os.path.join(output_path, category)
    os.makedirs(category_output, exist_ok=True)

    for video_file in os.listdir(category_path):
        video_path = os.path.join(category_path, video_file)
        if not video_file or not os.path.isdir(video_path):
            continue

        video_category_output = os.path.join(category_output, video_file)
        if os.path.exists(video_category_output):
            print(f"Skipping existing output: {video_category_output}")
            continue
        os.makedirs(video_category_output, exist_ok=True)

        print(f"video_path -> {video_path}")
        cnt = 1

        for frame_file in os.listdir(video_path):
            if os.path.splitext(frame_file)[1].lower() not in SUPPORTED_EXTENSIONS:
                continue

            print(f"frame_path -> {frame_file}")
            img_path = os.path.join(video_path, frame_file)
            img = load_and_resize_image(img_path)
            if img is None:
                print(f"Unable to read image: {img_path}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                faces = detector.detect_faces(rgb_img)
            except MemoryError:
                print(f"MemoryError while detecting faces for {img_path}. Skipping.")
                continue
            except Exception as error:
                print(f"Skipping frame due to error: {img_path}. {error}")
                continue

            for face_data in faces:
                if face_data.get('confidence', 0) <= 0.99:
                    continue

                x, y, width, height = face_data['box']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x1 + width, img.shape[1])
                y2 = min(y1 + height, img.shape[0])
                if x2 <= x1 or y2 <= y1:
                    continue

                cropped_face = img[y1:y2, x1:x2]
                resized_img = cv2.resize(cropped_face, (224, 224))

                frame_filename = f"frame_{cnt}.jpg"
                output_frame_path = os.path.join(video_category_output, frame_filename)
                cnt += 1
                cv2.imwrite(output_frame_path, resized_img)

cv2.destroyAllWindows()




