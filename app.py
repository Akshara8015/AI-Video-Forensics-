import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from torchvision import transforms

from Frequency_artifact_detector import FrequencyArtifactDetector
from models import LSTM


APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "deepfake_models"
MODEL_PATH = APP_DIR / "best_deepfake_model.pth"
MODEL_CANDIDATES = (
    MODEL_PATH,
    MODEL_DIR / "best_deepfake_model.pth",
    MODEL_DIR / "latest_checkpoint.pth",
    MODEL_DIR / "checkpoint_epoch_10.pth",
    MODEL_DIR / "checkpoints" / "best_deepfake_model.pth",
    MODEL_DIR / "checkpoints" / "latest_checkpoint.pth",
    MODEL_DIR / "checkpoints" / "checkpoint_epoch_10.pth",
)
SEQUENCE_LENGTH = 16
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]


st.set_page_config(
    page_title="AI Video Forensics",
    page_icon=":movie_camera:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>

    #MainMenu, footer, header {
        visibility: hidden;
    }

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        scroll-behavior: smooth;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    

    .stApp {
        background:
            linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(180deg, rgba(255, 255, 255, 0.032) 1px, transparent 1px),
            linear-gradient(135deg, #08111b 0%, #111827 38%, #182126 68%, #221a26 100%);
        background-size: 44px 44px, 44px 44px, auto;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #08121f;
    }

    h1, h2, h3, p {
        letter-spacing: 0;
    }

    .site-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 12px 0 18px 0;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        color: #ffffff;
        font-weight: 900;
        font-size: 1.04rem;
    }

    .brand-mark {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: #7dd3fc;
        color: #07111d;
        font-weight: 900;
        box-shadow: 0 14px 34px rgba(125, 211, 252, 0.24);
    }

    .nav-links {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
    }

    .nav-links a {
        color: #cbd5e1 !important;
        font-size: 0.9rem;
        font-weight: 800;
        text-decoration: none;
        transition: color 160ms ease, transform 160ms ease;
    }

    .nav-links a:hover {
        color: #7dd3fc !important;
        transform: translateY(-1px);
    }

    .hero {
        min-height: 72vh;
        display: grid;
        grid-template-columns: minmax(0, 1.08fr) minmax(330px, 0.92fr);
        align-items: center;
        gap: 36px;
        padding: 12px 0 34px;
    }

    .eyebrow {
        color: #67e8f9;
        font-size: 0.78rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        margin-bottom: 14px;
    }

    .hero-title {
        color: #ffffff;
        font-size: clamp(2.45rem, 5.6vw, 5.1rem);
        font-weight: 900;
        line-height: 0.96;
        margin: 0;
    }

    .hero-copy {
        color: #cbd5e1;
        font-size: 1.08rem;
        line-height: 1.75;
        margin-top: 22px;
        max-width: 670px;
    }

    .hero-actions {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-top: 28px;
        flex-wrap: wrap;
    }

    .primary-link, .secondary-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 46px;
        padding: 0 18px;
        border-radius: 8px;
        font-weight: 900;
        text-decoration: none;
        transition: transform 160ms ease, filter 160ms ease, border-color 160ms ease;
        
    }

    .primary-link:hover, .secondary-link:hover {
        transform: translateY(-1px);
    }

    .primary-link {
        color: #07111d !important;
        background: linear-gradient(135deg, #7dd3fc, #86efac);
        box-shadow: 0 18px 36px rgba(45, 212, 191, 0.22);
    }

    .secondary-link {
        color: #e2e8f0 !important;
        border: 1px solid rgba(226, 232, 240, 0.24);
        background: rgba(15, 23, 42, 0.56);
    }

    .visual-stage {
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 8px;
        min-height: 430px;
        padding: 22px;
        background:
            linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(31, 41, 55, 0.78)),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.05) 0 1px, transparent 1px 58px);
        box-shadow: 0 28px 90px rgba(0, 0, 0, 0.32);
        position: relative;
        overflow: hidden;
    }

    .scan-frame {
        border: 1px solid rgba(103, 232, 249, 0.32);
        border-radius: 8px;
        min-height: 235px;
        background:
            linear-gradient(180deg, rgba(103, 232, 249, 0.09), rgba(15, 23, 42, 0.2)),
            repeating-linear-gradient(0deg, rgba(103, 232, 249, 0.09) 0 2px, transparent 2px 18px);
        display: grid;
        place-items: center;
        margin-bottom: 18px;
    }

    .scan-face {
        width: 150px;
        height: 190px;
        border-radius: 48% 48% 42% 42%;
        border: 2px solid rgba(103, 232, 249, 0.78);
        box-shadow: inset 0 0 28px rgba(103, 232, 249, 0.22), 0 0 34px rgba(52, 211, 153, 0.2);
        position: relative;
    }

    .scan-face:before, .scan-face:after {
        content: "";
        position: absolute;
        top: 76px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #67e8f9;
        box-shadow: 0 0 16px rgba(103, 232, 249, 0.8);
    }

    .scan-face:before {
        left: 38px;
    }

    .scan-face:after {
        right: 38px;
    }

    .stage-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
    }

    .mini-stat, .service {
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.74);
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.22);
        backdrop-filter: blur(14px);
    }

    .mini-stat {
        padding: 16px;
    }

    .mini-label {
        color: #94a3b8;
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .mini-value {
        color: #ffffff;
        font-size: 1.24rem;
        font-weight: 900;
        margin-top: 4px;
    }

    .section {
        padding: 34px 0;
    }

    .section-title {
        color: #ffffff;
        font-size: clamp(1.65rem, 3vw, 2.6rem);
        font-weight: 900;
        margin: 0 0 10px 0;
    }

    .section-copy {
        color: #b8c4d6;
        line-height: 1.7;
        max-width: 780px;
        margin-bottom: 22px;
    }

    .service-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
    }

    .service {
        padding: 22px;
    }

    .service-kicker {
        color: #34d399;
        font-size: 0.78rem;
        font-weight: 900;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .service h3 {
        color: #ffffff;
        font-size: 1.08rem;
        margin: 0 0 10px 0;
    }

    .service p, .soft-text {
        color: #b8c4d6;
        line-height: 1.65;
    }

    .anchor-target {
        scroll-margin-top: 26px;
    }

    .status-good {
        color: #53f0b3;
        font-weight: 900;
    }

    .status-warn {
        color: #fbbf24;
        font-weight: 900;
    }

    .result-real, .result-fake {
        padding: 22px;
        border-radius: 8px;
        margin-bottom: 16px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    .result-real {
        border-left: 5px solid #34d399;
        background: rgba(6, 78, 59, 0.56);
    }

    .result-fake {
        border-left: 5px solid #fb7185;
        background: rgba(136, 19, 55, 0.56);
    }

    .verdict {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 900;
        margin: 0 0 6px 0;
    }

    .stButton > button {
        width: 100%;
        min-height: 3.25rem;
        border-radius: 8px;
        border: 0 !important;
        color: #07111d !important;
        background: linear-gradient(135deg, #7dd3fc 0%, #86efac 100%);
        font-weight: 900;
        box-shadow: 0 14px 30px rgba(45, 212, 191, 0.22);
    }

    .stButton > button * {
        color: #07111d !important;
    }

    .stButton > button:hover {
        color: #07111d !important;
        border: 0 !important;
        filter: brightness(1.04);
    }

    .stButton > button:disabled,
    .stButton > button:disabled:hover {
        background: linear-gradient(135deg, #7dd3fc 0%, #86efac 100%) !important;
        color: #07111d !important;
        border: 0 !important;
        box-shadow: 0 14px 30px rgba(45, 212, 191, 0.18);
        cursor: not-allowed;
    }

    .stButton > button:disabled * {
        color: #07111d !important;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(103, 232, 249, 0.46);
        border-radius: 8px;
        padding: 18px;
        background: rgba(8, 20, 35, 0.66);
    }

    [data-testid="stFileUploader"] * {
        color: #e2e8f0;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 14px;
        background: rgba(2, 6, 23, 0.34);
    }

    .model-status-card {
        border: 1px solid rgba(125, 211, 252, 0.24);
        border-radius: 8px;
        padding: 14px;
        background: rgba(8, 20, 35, 0.58);
        margin-bottom: 14px;
    }

    .model-path {
        color: #e2e8f0;
        font-family: "Cascadia Mono", Consolas, monospace;
        font-size: 0.82rem;
        word-break: break-word;
        margin-top: 6px;
    }

    .model-size {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    @media (max-width: 860px) {
        .hero {
            grid-template-columns: 1fr;
            min-height: auto;
        }
        .service-grid, .stage-grid {
            grid-template-columns: 1fr;
        }
        .nav-links {
            display: none;
        }
    }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


SMOOTH_SCROLL_SCRIPT = """
<script>
(function () {
    const parentWindow = window.parent;
    const parentDocument = parentWindow.document;

    if (parentWindow.__aiVideoSmoothScrollV2Ready) {
        return;
    }

    parentWindow.__aiVideoSmoothScrollV2Ready = true;

    const scrollOffset = 20;

    function easeInOutCubic(progress) {
        return progress < 0.5
            ? 4 * progress * progress * progress
            : 1 - Math.pow(-2 * progress + 2, 3) / 2;
    }

    function isPageScroller(element) {
        return (
            element === parentDocument.body ||
            element === parentDocument.documentElement ||
            element === parentDocument.scrollingElement
        );
    }

    function canScroll(element) {
        if (!element) {
            return false;
        }

        return element.scrollHeight > element.clientHeight + 4;
    }

    function getScrollableAncestor(target) {
        let current = target.parentElement;

        while (current && current !== parentDocument.body) {
            const style = parentWindow.getComputedStyle(current);
            const overflowY = style.overflowY || "";

            if (canScroll(current) && /(auto|scroll|overlay)/.test(overflowY)) {
                return current;
            }

            current = current.parentElement;
        }

        return null;
    }

    function getScroller(target) {
        const closestScroller = getScrollableAncestor(target);

        if (closestScroller) {
            return closestScroller;
        }

        const candidates = [
            parentDocument.querySelector("[data-testid='stMain']"),
            parentDocument.querySelector("section[data-testid='stMain']"),
            parentDocument.querySelector("[data-testid='stAppViewContainer']"),
            parentDocument.querySelector(".stApp"),
            parentDocument.scrollingElement,
            parentDocument.documentElement,
            parentDocument.body
        ];

        for (const candidate of candidates) {
            if (candidate && candidate.contains(target) && canScroll(candidate)) {
                return candidate;
            }
        }

        return null;
    }

    function getScrollTop(scroller) {
        if (isPageScroller(scroller)) {
            return (
                parentWindow.pageYOffset ||
                parentDocument.documentElement.scrollTop ||
                parentDocument.body.scrollTop ||
                0
            );
        }

        return scroller.scrollTop;
    }

    function setScrollTop(scroller, value) {
        if (isPageScroller(scroller)) {
            parentWindow.scrollTo(0, value);
            parentDocument.documentElement.scrollTop = value;
            parentDocument.body.scrollTop = value;
            return;
        }

        scroller.scrollTop = value;
    }

    function smoothScrollTo(target) {
        const scroller = getScroller(target);

        if (!scroller) {
            return false;
        }

        const start = getScrollTop(scroller);
        const scrollerTop = isPageScroller(scroller) ? 0 : scroller.getBoundingClientRect().top;
        const end = start + target.getBoundingClientRect().top - scrollerTop - scrollOffset;
        const distance = end - start;
        const duration = Math.min(950, Math.max(480, Math.abs(distance) * 0.5));
        const startTime = parentWindow.performance.now();

        function step(now) {
            const progress = Math.min((now - startTime) / duration, 1);
            setScrollTop(scroller, start + distance * easeInOutCubic(progress));

            if (progress < 1) {
                parentWindow.requestAnimationFrame(step);
            }
        }

        parentWindow.requestAnimationFrame(step);
        return true;
    }

    parentDocument.addEventListener(
        "click",
        function (event) {
            const anchor = event.target && event.target.closest
                ? event.target.closest("a[href^='#']")
                : null;

            if (!anchor) {
                return;
            }

            const targetId = decodeURIComponent(anchor.getAttribute("href").slice(1));
            const target = parentDocument.getElementById(targetId);

            if (!target) {
                return;
            }

            const handled = smoothScrollTo(target);

            if (!handled) {
                return;
            }

            event.preventDefault();
            event.stopPropagation();

            try {
                parentWindow.history.pushState(null, "", "#" + targetId);
            } catch (error) {
                parentWindow.location.hash = targetId;
            }
        },
        true
    );
})();
</script>
"""


components.html(SMOOTH_SCROLL_SCRIPT, height=0)


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return MODEL_PATH


def model_location_label(model_path: Path | None = None) -> str:
    active_path = model_path or resolve_model_path()
    try:
        return active_path.relative_to(APP_DIR).as_posix()
    except ValueError:
        return str(active_path)


def model_size_label(model_path: Path | None = None) -> str:
    active_path = model_path or resolve_model_path()
    if not active_path.exists():
        return "Model file not found"
    size_mb = active_path.stat().st_size / (1024 * 1024)
    return f"{size_mb:.1f} MB checkpoint"


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict

        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Unsupported checkpoint format. Expected a raw state dict or a checkpoint with "
        "`model_state_dict`."
    )


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def sample_video_frames(video_path: str, sequence_length: int) -> tuple[list[Image.Image], dict]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Unable to open the uploaded video.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        indices = list(range(sequence_length))
    else:
        indices = np.linspace(0, max(total_frames - 1, 0), sequence_length, dtype=int).tolist()

    frames = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb_frame).resize((224, 224)))

    capture.release()

    if not frames:
        raise ValueError("No readable frames were found in the uploaded video.")

    while len(frames) < sequence_length:
        frames.append(frames[-1].copy())

    metadata = {
        "frames": total_frames,
        "fps": fps,
        "resolution": f"{width} x {height}" if width and height else "Unknown",
        "duration": (total_frames / fps) if total_frames and fps else 0,
    }
    return frames[:sequence_length], metadata


def prepare_tensor(frames: list[Image.Image], device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    tensor = torch.stack([transform(frame) for frame in frames])
    return tensor.unsqueeze(0).to(device)


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def predict_video(video_path: str, frames: list[Image.Image]) -> float:
    model, device = load_model(str(resolve_model_path()))
    input_tensor = prepare_tensor(frames, device)

    with torch.no_grad():
        logits = model(input_tensor).squeeze()
        fake_probability = torch.sigmoid(logits).item()

    return fake_probability


def analyze_frequency(frames: list[Image.Image]) -> tuple[float, dict]:
    detector = FrequencyArtifactDetector(image_size=224)
    scores = []
    feature_rows = []

    for frame in frames:
        rgb = np.array(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        score, features = detector.analyze_frame(bgr)
        scores.append(score)
        feature_rows.append(features)

    summary = {
        "high_freq_ratio": float(np.mean([row["high_freq_ratio"] for row in feature_rows])),
        "entropy": float(np.mean([row["entropy"] for row in feature_rows])),
        "freq_variance": float(np.mean([row["freq_variance"] for row in feature_rows])),
    }
    return float(np.mean(scores)), summary


def create_frequency_spectrum(frame: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    color = cv2.applyColorMap(magnitude.astype(np.uint8), cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))


def create_region_heatmap(frames: list[Image.Image]) -> tuple[Image.Image, float]:
    arrays = [np.array(frame.resize((224, 224))) for frame in frames[: min(len(frames), 8)]]
    base = arrays[len(arrays) // 2]

    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in arrays]
    temporal_motion = np.mean(
        [cv2.absdiff(gray_frames[i], gray_frames[i - 1]) for i in range(1, len(gray_frames))],
        axis=0,
    )
    edges = cv2.Laplacian(gray_frames[len(gray_frames) // 2], cv2.CV_32F)
    edges = np.abs(edges)
    heat = cv2.GaussianBlur((temporal_motion * 0.62) + (edges * 0.38), (0, 0), 5)
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.58, heatmap, 0.42, 0)
    region_score = float(np.mean(heat) / 255.0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), region_score


def render_nav() -> None:
    st.markdown(
        """
        <div class="site-nav">
            <div class="brand"><div class="brand-mark">AI</div><span>AI Video Forensics</span></div>
            <nav class="nav-links" aria-label="Page sections">
                <a href="#services">Services</a>
                <a href="#upload">Upload</a>
                <a href="#analysis-lab">Analysis</a>
            </nav>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
            <div>
                <div class="eyebrow">Deepfake detection platform</div>
                <h1 class="hero-title">Verify videos before you trust them.</h1>
                <p class="hero-copy">
                    AI Video Forensics inspects uploaded clips for manipulation signals using your trained
                    CNN-LSTM checkpoint, frequency analysis, and a visual region heatmap. The result is
                    fast to read, easy to explain, and grounded in visible forensic evidence.
                </p>
                <div class="hero-actions">
                    <a class="primary-link" href="#analysis-lab">Start Analysis</a>
                    <a class="secondary-link" href="#services">View Services</a>
                </div>
            </div>
            <div class="visual-stage">
                <div class="scan-frame"><div class="scan-face"></div></div>
                <div class="stage-grid">
                    <div class="mini-stat">
                        <div class="mini-label">Classifier</div>
                        <div class="mini-value">Trained</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Signal</div>
                        <div class="mini-value">FFT</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Visual</div>
                        <div class="mini-value">Heatmap</div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_services() -> None:
    st.markdown(
        """
        <section class="section" id="services">
            <p class="eyebrow">What the website does</p>
            <h2 class="section-title">A complete video authenticity check</h2>
            <p class="section-copy">
                Upload a video and review a layered authenticity report: the model verdict,
                spectral artifact score, and highlighted regions are shown together so the decision
                is understandable at a glance.
            </p>
            <div class="service-grid">
                <div class="service">
                    <div class="service-kicker">01</div>
                    <h3>Real or fake prediction</h3>
                    <p>The model samples video frames and uses temporal CNN-LSTM features to produce a final verdict.</p>
                </div>
                <div class="service">
                    <div class="service-kicker">02</div>
                    <h3>Frequency artifact detection</h3>
                    <p>FFT-based analysis checks for unusual high-frequency traces and compression-like anomalies.</p>
                </div>
                <div class="service">
                    <div class="service-kicker">03</div>
                    <h3>Region manipulation heatmap</h3>
                    <p>A visual overlay highlights frame regions that show stronger artifact or motion inconsistency.</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_result(fake_probability: float) -> None:
    is_fake = fake_probability >= 0.5
    confidence = fake_probability if is_fake else 1 - fake_probability
    label = "FAKE VIDEO DETECTED" if is_fake else "REAL VIDEO"
    css_class = "result-fake" if is_fake else "result-real"
    description = (
        "The model found manipulation-like temporal and visual patterns."
        if is_fake
        else "The model found stronger evidence of natural frame consistency."
    )
    st.markdown(
        f"""
        <div class="{css_class}">
            <p class="verdict">{label}</p>
            <p class="soft-text">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(confidence), text=f"Confidence: {confidence * 100:.2f}%")
    st.caption(f"Fake probability score: {fake_probability:.4f}")


def render_model_status() -> bool:
    active_model_path = resolve_model_path()
    model_ready = active_model_path.exists()
    if model_ready:
        st.markdown(
            f"""
            <div class="model-status-card">
                <div class="status-good">Model checkpoint found</div>
                <div class="model-path">{model_location_label(active_model_path)}</div>
                <div class="model-size">{model_size_label(active_model_path)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="model-status-card">
                <div class="status-warn">Model checkpoint missing</div>
                <div class="model-path">{model_location_label(active_model_path)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Place `best_deepfake_model.pth` directly in the project folder to enable CNN-LSTM predictions.")
    return model_ready


render_nav()
render_hero()
render_services()

st.markdown('<div id="analysis-lab" class="anchor-target"></div>', unsafe_allow_html=True)
st.markdown('<p class="eyebrow">Upload and analyze</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Forensic analysis lab</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Upload a video, run the detector, and review the final verdict with supporting forensic views.</p>',
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([1.05, 0.95], gap="large")

with top_left:
    st.markdown('<div id="upload" class="anchor-target"></div>', unsafe_allow_html=True)
    st.markdown("### Video Upload")
    uploaded_video = st.file_uploader(
        "Upload video",
        type=SUPPORTED_VIDEO_TYPES,
        label_visibility="collapsed",
    )

    if uploaded_video:
        st.video(uploaded_video)
        file_size_mb = len(uploaded_video.getbuffer()) / (1024 * 1024)
        st.caption(f"{uploaded_video.name} - {file_size_mb:.2f} MB")
    else:
        st.markdown(
            '<p class="soft-text">Supported formats: MP4, AVI, MOV, and MKV.</p>',
            unsafe_allow_html=True,
        )

with top_right:
    st.markdown("### System Status")
    model_ready = render_model_status()
    st.markdown("**Analysis modules**")
    st.markdown("- Real/fake sequence classifier")
    st.markdown("- Frequency artifact detector")
    st.markdown("- Region manipulation heatmap")
    analyze_clicked = st.button("Process Video", disabled=uploaded_video is None, type="primary")

if uploaded_video and analyze_clicked:
    temp_path = save_uploaded_file(uploaded_video)
    try:
        with st.spinner("Processing video and generating forensic views..."):
            sampled_frames, metadata = sample_video_frames(temp_path, SEQUENCE_LENGTH)
            frequency_score, frequency_features = analyze_frequency(sampled_frames)
            spectrum_image = create_frequency_spectrum(sampled_frames[len(sampled_frames) // 2])
            heatmap_image, heatmap_score = create_region_heatmap(sampled_frames)
            fake_probability = predict_video(temp_path, sampled_frames) if model_ready else frequency_score

        st.write("")
        result_col, freq_col, heat_col = st.columns(3, gap="large")

        with result_col:
            st.markdown("### Final Verdict")
            render_result(fake_probability)
            st.metric("Frames sampled", len(sampled_frames))
            st.metric("Video duration", f"{metadata['duration']:.1f}s" if metadata["duration"] else "Unknown")

        with freq_col:
            st.markdown("### Frequency Artifact Detection")
            st.image(spectrum_image, caption="Frequency spectrum preview", use_container_width=True)
            st.progress(float(frequency_score), text=f"Artifact score: {frequency_score * 100:.2f}%")
            st.metric("High frequency ratio", f"{frequency_features['high_freq_ratio']:.3f}")
            st.metric("Entropy", f"{frequency_features['entropy']:.2f}")

        with heat_col:
            st.markdown("### Region Manipulation Heatmap")
            st.image(heatmap_image, caption="Highlighted suspicious regions", use_container_width=True)
            st.progress(float(heatmap_score), text=f"Regional anomaly score: {heatmap_score * 100:.2f}%")
            st.metric("Resolution", metadata["resolution"])
            st.metric("FPS", f"{metadata['fps']:.1f}" if metadata["fps"] else "Unknown")

        st.markdown("### Sampled Video Frames")
        st.image(sampled_frames[:6], width=135)
    except Exception as error:
        st.error(f"Analysis failed: {error}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
elif uploaded_video:
    st.info("Click `Process Video` to generate the prediction, frequency artifact view, and heatmap.")
