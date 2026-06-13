import cv2
import os

dataset_path = r"E:\downloads folder\archive\FaceForensics++_C23"
output_path = r"E:\Frames"

os.makedirs(output_path, exist_ok=True)

for category in os.listdir(dataset_path):

    category_path = os.path.join(dataset_path, category)

    if not os.path.isdir(category_path):
        continue

    print(f"\nProcessing Category: {category}")

    category_output = os.path.join(output_path, category)
    os.makedirs(category_output, exist_ok=True)

    for video_file in os.listdir(category_path):

        video_path = os.path.join(category_path, video_file)
        if not video_file.endswith(('.mp4', '.avi', '.mov')):
            continue

        print(f"Extracting Frames from: {video_file}")

        video_name = os.path.splitext(video_file)[0]

        video_output_folder = os.path.join(category_output, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        # Open video
        vid = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_frame_cnt = 0

        while True:
            success, frame = vid.read()

            if not success:
                break

            if saved_frame_cnt&25 == 0:
                frame_filename = f"frame_{frame_count}.jpg"
                frame_path = os.path.join(video_output_folder, frame_filename)

                cv2.imwrite(frame_path, frame)
                frame_count += 1
            saved_frame_cnt += 1

        vid.release()
        print(f"Saved {frame_count} frames in {video_output_folder}")

cv2.destroyAllWindows()
print("\nAll videos processed successfully!")

