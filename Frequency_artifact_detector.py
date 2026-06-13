import cv2
import numpy as np
from tqdm import tqdm


class FrequencyArtifactDetector:
    def __init__(self, image_size=224):
        self.image_size = image_size

    def _compute_fft_features(self, gray):
        # Step 1: FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)

        # Step 2: magnitude spectrum
        magnitude = np.abs(fft_shift)
        magnitude_log = np.log1p(magnitude)

        return magnitude_log

    def _extract_features(self, magnitude):
        h, w = magnitude.shape

        # center mask (low frequency region)
        center_h, center_w = h // 4, w // 4
        center = magnitude[
            h//2 - center_h : h//2 + center_h,
            w//2 - center_w : w//2 + center_w
        ]

        # high frequency region (rest of image)
        total_energy = np.sum(magnitude)
        low_energy = np.sum(center)
        high_energy = total_energy - low_energy

        # ratios
        high_freq_ratio = high_energy / (total_energy + 1e-8)

        # entropy (distribution randomness)
        norm = magnitude / (np.sum(magnitude) + 1e-8)
        entropy = -np.sum(norm * np.log(norm + 1e-8))

        # edge sharpness proxy (variance in frequency space)
        freq_variance = np.var(magnitude)

        return {
            "high_freq_ratio": high_freq_ratio,
            "entropy": entropy,
            "freq_variance": freq_variance
        }

    def _score(self, features):
        """
        Heuristic scoring:
        Deepfakes tend to have:
        - higher high-frequency artifacts
        - abnormal entropy patterns
        - unstable frequency variance
        """

        score = 0.0

        # high frequency weight
        score += min(features["high_freq_ratio"] * 1.2, 0.5)

        # entropy contribution
        score += min(features["entropy"] / 10.0, 0.3)

        # variance contribution
        score += min(features["freq_variance"] / 1e6, 0.2)

        return float(np.clip(score, 0.0, 1.0))

    def analyze_frame(self, frame):
        # Resize
        frame = cv2.resize(frame, (self.image_size, self.image_size))

        # grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # FFT
        magnitude = self._compute_fft_features(gray)

        # features
        features = self._extract_features(magnitude)

        # score
        score = self._score(features)

        return score, features


    def analyze_video(self, video_path, sample_every=5):
        cap = cv2.VideoCapture(video_path)

        scores = []
        frame_count = 0

        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_every == 0:
                    score, _ = self.analyze_frame(frame)
                    scores.append(score)

                frame_count += 1
                pbar.update(1)

        cap.release()

        if len(scores) == 0:
            return 0.0

        # final video score
        return float(np.mean(scores))


# ----------------------------
# RUN EXAMPLE
# ----------------------------

if __name__ == "__main__":
    detector = FrequencyArtifactDetector(image_size=224)

    video_path = "test_video.mp4"

    score = detector.analyze_video(video_path, sample_every=5)

    print("\n==========================")
    print(f"Frequency Artifact Score: {score:.4f}")
    print("==========================\n")

    if score > 0.6:
        print("⚠️ Likely Deepfake (Frequency anomalies detected)")
    else:
        print("✅ Likely Real (No strong frequency artifacts)")


        