import os
import glob
import io
import zipfile
import base64

import numpy as np
import rasterio
import streamlit as st
import streamlit.components.v1 as components
import torch

from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from model import prompt_model
import st_yled


st_yled.init(theme="scandinavian")



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Prithvi HLS VLM Viewer",
    page_icon="🌍",
    layout="wide",
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌍 Prithvi HLS Multi-Temporal VLM Viewer")


# ============================================================
# HUGGING FACE MODEL
# ============================================================

HF_REPO_ID = "elamandeep/vlm-remote-sensing"
HF_MODEL_FILE = "model.ckpt"


@st.cache_resource(show_spinner=False)
def load_vlm_model():

    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILE,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = checkpoint["state_dict"]

    missing, unexpected = prompt_model.load_state_dict(
        state_dict,
        strict=False,
    )

    prompt_model.to(device)
    prompt_model.eval()

    return prompt_model, checkpoint_path, missing, unexpected


with st.spinner("Loading VLM weights from Hugging Face..."):

    prompt_model, checkpoint_path, missing_keys, unexpected_keys = (
        load_vlm_model()
    )

st.success(
    f"Model loaded from Hugging Face | Device: `{device}`"
)


# ============================================================
# DATA DIRECTORIES
# ============================================================

DIR = r"./assets"

# Automatically derive:
#
# ...\images
#
# ->
#
# ...\masks

IMAGE_DIR = os.path.join(DIR, "images")
MASK_DIR = os.path.join(DIR, "masks")

VALID_EXTENSIONS = (
    "*.tif",
    "*.tiff",
)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "natural vegetation",
    "forest",
    "corn",
    "soybeans",
    "wetlands",
    "developed",
    "open water",
    "barren",
    "winter wheat",
    "alfalfa",
    "fallow",
    "idle cropland",
    "cotton",
    "sorghum",
    "other",
]


# RGB values are in [0, 1]
CLASS_COLORS = [
    [0.00, 0.80, 0.00],   # natural vegetation
    [0.00, 0.40, 0.00],   # forest
    [1.00, 0.80, 0.00],   # corn
    [0.20, 0.80, 0.20],   # soybeans
    [0.00, 0.60, 0.80],   # wetlands
    [0.85, 0.10, 0.10],   # developed
    [0.00, 0.00, 1.00],   # open water
    [0.60, 0.60, 0.60],   # barren
    [0.80, 0.60, 0.20],   # winter wheat
    [0.40, 0.80, 0.40],   # alfalfa
    [0.60, 0.40, 0.20],   # fallow
    [0.85, 0.65, 0.40],   # idle cropland
    [1.00, 0.70, 0.70],   # cotton
    [0.80, 0.20, 0.80],   # sorghum
    [0.00, 0.00, 0.00],   # other
]

CLASS_COLORS = np.asarray(
    CLASS_COLORS,
    dtype=np.float32,
)


BACKGROUND_COLOR = np.array(
    [0.0, 0.0, 0.0],
    dtype=np.float32,
)


ABS_SIMILARITY_THRESHOLD = 0.20


# ============================================================
# PROMPT ALIASES
# ============================================================

PROMPT_ALIASES = {
    "developed area": "developed",
    "urban": "developed",
    "urban area": "developed",
    "water": "open water",
    "water body": "open water",
    "forest area": "forest",
    "vegetation": "natural vegetation",
    "natural vegetation area": "natural vegetation",
}


def normalize_prompt(text):

    clean = text.lower().strip()

    return PROMPT_ALIASES.get(
        clean,
        clean,
    )


# ============================================================
# FILE LOADING
# ============================================================

@st.cache_data
def load_tif_files(folder):

    files = []

    if not os.path.exists(folder):
        return files

    for extension in VALID_EXTENSIONS:

        files.extend(
            glob.glob(
                os.path.join(
                    folder,
                    extension,
                )
            )
        )

    return sorted(files)


tif_files = load_tif_files(
    IMAGE_DIR
)


# ============================================================
# IMAGE -> MASK MATCHING
# ============================================================

@st.cache_data
def find_mask_for_image(
    image_path,
):

    filename = os.path.basename(
        image_path
    )

    stem = os.path.splitext(
        filename
    )[0]

    candidates = [
        os.path.join(
            MASK_DIR,
            filename,
        ),

        os.path.join(
            MASK_DIR,
            stem + ".tif",
        ),

        os.path.join(
            MASK_DIR,
            stem + ".tiff",
        ),
    ]

    for candidate in candidates:

        if os.path.exists(candidate):

            return candidate

    return None


# ============================================================
# GEOTIFF -> RGB
# ============================================================

@st.cache_data
def process_geotiff_to_rgb(
    tif_path,
    time_step=0,
):

    with rasterio.open(tif_path) as src:

        data = src.read()

    bands_per_step = 6

    r_idx = (
        time_step * bands_per_step
    ) + 2

    g_idx = (
        time_step * bands_per_step
    ) + 1

    b_idx = (
        time_step * bands_per_step
    ) + 0

    if data.shape[0] < r_idx + 1:

        r_idx = 0
        g_idx = 0
        b_idx = 0

    rgb = np.stack(
        [
            data[r_idx],
            data[g_idx],
            data[b_idx],
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    rgb = np.where(
        rgb == -9999,
        0,
        rgb,
    )

    # Percentile stretch independently per RGB channel.
    stretched = np.zeros_like(
        rgb,
        dtype=np.float32,
    )

    for channel in range(3):

        channel_data = rgb[:, :, channel]

        valid = channel_data[
            np.isfinite(channel_data)
        ]

        if valid.size == 0:

            continue

        p2, p98 = np.percentile(
            valid,
            [2, 98],
        )

        if p98 > p2:

            stretched[:, :, channel] = np.clip(
                (
                    channel_data - p2
                ) / (
                    p98 - p2
                ),
                0,
                1,
            )

        else:

            stretched[:, :, channel] = np.clip(
                channel_data,
                0,
                1,
            )

    return Image.fromarray(
        (
            stretched * 255
        ).astype(
            np.uint8
        )
    )


# ============================================================
# GEOTIFF -> MODEL TENSOR
# ============================================================

@st.cache_data
def process_geotiff_to_tensor(
    tif_path,
):

    with rasterio.open(tif_path) as src:

        data = src.read().astype(
            np.float32
        )

    data = np.where(
        data == -9999,
        0,
        data,
    )

    eps = 1e-4

    mean = data.mean(
        axis=(1, 2),
        keepdims=True,
    )

    std = data.std(
        axis=(1, 2),
        keepdims=True,
    )

    data = (
        data - mean
    ) / (
        std + eps
    )

    return torch.from_numpy(
        data.astype(
            np.float32
        )
    )


# ============================================================
# LOAD GT MASK
# ============================================================

@st.cache_data
def load_ground_truth_label(
    image_path,
):

    mask_path = find_mask_for_image(
        image_path
    )

    if mask_path is None:

        raise FileNotFoundError(
            "No corresponding mask found.\n\n"
            f"Image:\n{image_path}\n\n"
            f"Expected mask directory:\n{MASK_DIR}"
        )

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

    return (
        mask.astype(np.int32),
        mask_path,
    )


# ============================================================
# GT MASK ID CONVERSION
# ============================================================

def convert_mask_ids(
    raw_mask,
    mode,
):

    mask = raw_mask.copy()

    if mode == "1-based":

        # 1 -> 0
        # 2 -> 1
        # ...
        #
        # 0 remains background / ignored.

        converted = np.full_like(
            mask,
            -1,
        )

        valid = (
            (mask >= 1)
            & (
                mask
                <= len(CLASS_NAMES)
            )
        )

        converted[valid] = (
            mask[valid] - 1
        )

        return converted

    if mode == "0-based":

        converted = mask.copy()

        valid = (
            (converted >= 0)
            & (
                converted
                < len(CLASS_NAMES)
            )
        )

        converted[
            ~valid
        ] = -1

        return converted

    return mask


# ============================================================
# COLORIZE MASK
# ============================================================

def colorize_labels(
    label_array,
    colors,
    background=BACKGROUND_COLOR,
):

    height, width = label_array.shape

    rgb = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.float32,
    )

    rgb[:] = background

    for idx, color in enumerate(colors):

        rgb[
            label_array == idx
        ] = color

    return rgb


# ============================================================
# PROMPT COLORS
# ============================================================

def get_phrase_colors(
    phrases,
):

    colors = []

    for phrase in phrases:

        clean = normalize_prompt(
            phrase
        )

        if clean in CLASS_NAMES:

            index = CLASS_NAMES.index(
                clean
            )

            colors.append(
                CLASS_COLORS[index]
            )

        else:

            seed = sum(
                ord(c)
                for c in clean
            )

            rng = np.random.default_rng(
                seed
            )

            colors.append(
                rng.random(
                    3,
                    dtype=np.float32,
                )
            )

    return np.asarray(
        colors,
        dtype=np.float32,
    )


# ============================================================
# RGB NUMPY -> PNG BYTES
# ============================================================

def rgb_array_to_png_bytes(
    rgb,
):

    rgb = np.clip(
        rgb,
        0,
        1,
    )

    image = Image.fromarray(
        (
            rgb * 255
        ).astype(
            np.uint8
        )
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# ============================================================
# PNG BYTES -> BASE64
# ============================================================

def png_bytes_to_base64(
    png_bytes,
):

    return base64.b64encode(
        png_bytes
    ).decode(
        "utf-8"
    )


# ============================================================
# ADD LABELS TO MASK IMAGE
# ============================================================

def add_mask_label_panel(
    rgb,
    label_array,
    class_names,
    colors,
    title="Ground Truth",
):

    image = Image.fromarray(
        (
            np.clip(
                rgb,
                0,
                1,
            ) * 255
        ).astype(
            np.uint8
        )
    ).convert("RGB")

    draw = ImageDraw.Draw(
        image,
        "RGBA",
    )

    present_labels = []

    for label in np.unique(
        label_array
    ):

        label = int(label)

        if (
            0 <= label
            < len(class_names)
        ):

            present_labels.append(
                label
            )

    if not present_labels:

        return image

    # Determine panel size.
    panel_width = 260
    row_height = 25
    panel_height = (
        45
        + len(present_labels)
        * row_height
        + 15
    )

    # Panel
    draw.rounded_rectangle(
        (
            10,
            10,
            panel_width,
            panel_height,
        ),
        radius=8,
        fill=(
            0,
            0,
            0,
            175,
        ),
        outline=(
            255,
            255,
            255,
            180,
        ),
        width=1,
    )

    # Title
    draw.text(
        (
            20,
            18,
        ),
        title,
        fill=(
            255,
            255,
            255,
            255,
        ),
    )

    y = 48

    for label in present_labels:

        color = (
            colors[label] * 255
        ).astype(
            np.uint8
        )

        color_tuple = tuple(
            int(v)
            for v in color
        )

        draw.rectangle(
            (
                20,
                y,
                36,
                y + 16,
            ),
            fill=(
                *color_tuple,
                255,
            ),
        )

        draw.text(
            (
                44,
                y - 2,
            ),
            f"{class_names[label]}",
            fill=(
                255,
                255,
                255,
                255,
            ),
        )

        y += row_height

    return image


# ============================================================
# INTERACTIVE QGIS-STYLE VIEWER
# ============================================================

def render_layer_viewer(
    rgb,
    gt_rgb=None,
    prediction_rgb=None,
    gt_legend=None,
    prediction_legend=None,
    height=650,
):

    rgb_bytes = rgb_array_to_png_bytes(
        rgb
    )

    rgb_b64 = png_bytes_to_base64(
        rgb_bytes
    )

    if gt_rgb is not None:

        gt_bytes = rgb_array_to_png_bytes(
            gt_rgb
        )

        gt_b64 = png_bytes_to_base64(
            gt_bytes
        )

    else:

        gt_b64 = ""

    if prediction_rgb is not None:

        pred_bytes = rgb_array_to_png_bytes(
            prediction_rgb
        )

        pred_b64 = png_bytes_to_base64(
            pred_bytes
        )

    else:

        pred_b64 = ""

    gt_legend_html = ""

    if gt_legend:

        gt_legend_html = "".join(
            [
                f"""
                <div class="legend-row">
                    <span
                        class="legend-color"
                        style="background:
                        rgb({int(c[0]*255)},
                            {int(c[1]*255)},
                            {int(c[2]*255)})">
                    </span>
                    <span>{name}</span>
                </div>
                """
                for name, c in gt_legend
            ]
        )

    pred_legend_html = ""

    if prediction_legend:

        pred_legend_html = "".join(
            [
                f"""
                <div class="legend-row">
                    <span
                        class="legend-color"
                        style="background:
                        rgb({int(c[0]*255)},
                            {int(c[1]*255)},
                            {int(c[2]*255)})">
                    </span>
                    <span>{name}</span>
                </div>
                """
                for name, c in prediction_legend
            ]
        )

    html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 0;
    background: transparent;
    font-family: Arial, sans-serif;
    color: white;
}}

.viewer {{
    position: relative;
    width: 100%;
    height: {height}px;
    background: #111;
    border-radius: 10px;
    overflow: hidden;
    user-select: none;
}}

.viewer img {{
    position: absolute;
    width: 100%;
    height: 100%;
    object-fit: contain;
    left: 0;
    top: 0;
    pointer-events: none;
}}


/* =========================================================
   LAYERS
   ========================================================= */

.layer {{
    position: absolute;
    inset: 0;
    pointer-events: none;
}}

.layer img {{
    opacity: 1;
}}


/* =========================================================
   TOP TOOLBAR
   ========================================================= */

.toolbar {{
    position: absolute;
    left: 12px;
    right: 12px;
    top: 12px;
    z-index: 50;

    display: flex;
    align-items: center;
    gap: 14px;

    padding: 10px 12px;

    background: rgba(0,0,0,0.72);

    border: 1px solid rgba(255,255,255,0.2);

    border-radius: 8px;

    backdrop-filter: blur(6px);

    font-size: 13px;
}}

.layer-control {{
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}}

.layer-control input[type="checkbox"] {{
    width: 15px;
    height: 15px;
}}

.opacity {{
    width: 80px;
}}

select {{
    background: #222;
    color: white;
    border: 1px solid #666;
    border-radius: 4px;
    padding: 4px;
}}

button {{
    background: #333;
    color: white;
    border: 1px solid #777;
    border-radius: 5px;
    padding: 5px 9px;
    cursor: pointer;
}}

button:hover {{
    background: #555;
}}


/* =========================================================
   LEGEND
   ========================================================= */

.legend {{
    position: absolute;
    left: 12px;
    bottom: 12px;

    z-index: 45;

    max-width: 280px;

    padding: 9px 11px;

    background: rgba(0,0,0,0.70);

    border: 1px solid rgba(255,255,255,0.2);

    border-radius: 7px;

    font-size: 12px;

    pointer-events: none;
}}

.legend-title {{
    font-weight: bold;
    margin-bottom: 6px;
}}

.legend-row {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 3px 0;
}}

.legend-color {{
    width: 13px;
    height: 13px;
    display: inline-block;
    border: 1px solid white;
}}


/* =========================================================
   SWIPE DIVIDER
   ========================================================= */

.swipe {{
    position: absolute;
    top: 0;
    bottom: 0;

    width: 3px;

    background: white;

    box-shadow:
        0 0 5px black;

    z-index: 40;

    cursor: ew-resize;

    display: none;
}}

.swipe-handle {{
    position: absolute;

    top: 50%;
    left: 50%;

    transform:
        translate(-50%, -50%);

    width: 42px;
    height: 42px;

    border-radius: 50%;

    background: white;

    color: black;

    display: flex;

    justify-content: center;

    align-items: center;

    font-size: 18px;

    box-shadow:
        0 1px 8px rgba(0,0,0,0.5);
}}


/* =========================================================
   INFO
   ========================================================= */

.info {{
    position: absolute;
    right: 12px;
    bottom: 12px;

    z-index: 45;

    padding: 6px 9px;

    background: rgba(0,0,0,0.65);

    border-radius: 5px;

    font-size: 11px;

    color: #ddd;
}}

</style>

</head>


<body>

<div
    class="viewer"
    id="viewer"
>


<!-- =====================================================
     RGB BASE
     ===================================================== -->

<div class="layer" id="rgbLayer">

    <img
        src="data:image/png;base64,{rgb_b64}"
    >

</div>


<!-- =====================================================
     GT LAYER
     ===================================================== -->

<div
    class="layer"
    id="gtLayer"
    style="display:none;"
>

    <img
        id="gtImage"
        src="data:image/png;base64,{gt_b64}"
    >

</div>


<!-- =====================================================
     PREDICTION LAYER
     ===================================================== -->

<div
    class="layer"
    id="predictionLayer"
    style="display:none;"
>

    <img
        id="predictionImage"
        src="data:image/png;base64,{pred_b64}"
    >

</div>


<!-- =====================================================
     TOOLBAR
     ===================================================== -->

<div class="toolbar">


    <div class="layer-control">

        <input
            type="checkbox"
            id="gtToggle"
        >

        <label for="gtToggle">
            Ground Truth
        </label>

        <input
            class="opacity"
            type="range"
            id="gtOpacity"
            min="0"
            max="100"
            value="55"
        >

        <span id="gtOpacityValue">
            55%
        </span>

    </div>


    <div class="layer-control">

        <input
            type="checkbox"
            id="predictionToggle"
        >

        <label for="predictionToggle">
            Prediction
        </label>

        <input
            class="opacity"
            type="range"
            id="predictionOpacity"
            min="0"
            max="100"
            value="55"
        >

        <span id="predictionOpacityValue">
            55%
        </span>

    </div>


    <div class="layer-control">

        <label>
            Swipe:
        </label>

        <select id="swipeLayer">

            <option value="none">
                Off
            </option>

            <option value="gt">
                Ground Truth
            </option>

            <option value="prediction">
                Prediction
            </option>

        </select>

    </div>


    <button
        onclick="resetViewer()"
    >
        Reset
    </button>


</div>


<!-- =====================================================
     SWIPE
     ===================================================== -->

<div
    class="swipe"
    id="swipe"
>

    <div
        class="swipe-handle"
    >
        ↔
    </div>

</div>


<!-- =====================================================
     GT LEGEND
     ===================================================== -->

<div
    class="legend"
    id="gtLegend"
    style="display:none;"
>

    <div class="legend-title">
        Ground Truth
    </div>

    {gt_legend_html}

</div>


<!-- =====================================================
     PREDICTION LEGEND
     ===================================================== -->

<div
    class="legend"
    id="predictionLegend"
    style="
        display:none;
        left:auto;
        right:12px;
    "
>

    <div class="legend-title">
        Prediction
    </div>

    {pred_legend_html}

</div>


<div class="info">
    RGB base layer · toggle GT / Prediction independently
</div>


</div>


<script>


const viewer =
    document.getElementById("viewer");


const gtLayer =
    document.getElementById("gtLayer");


const predictionLayer =
    document.getElementById("predictionLayer");


const gtImage =
    document.getElementById("gtImage");


const predictionImage =
    document.getElementById("predictionImage");


const gtToggle =
    document.getElementById("gtToggle");


const predictionToggle =
    document.getElementById("predictionToggle");


const gtOpacity =
    document.getElementById("gtOpacity");


const predictionOpacity =
    document.getElementById("predictionOpacity");


const gtOpacityValue =
    document.getElementById("gtOpacityValue");


const predictionOpacityValue =
    document.getElementById("predictionOpacityValue");


const gtLegend =
    document.getElementById("gtLegend");


const predictionLegend =
    document.getElementById("predictionLegend");


const swipe =
    document.getElementById("swipe");


const swipeLayer =
    document.getElementById("swipeLayer");


let swipePosition = 50;

let draggingSwipe = false;



// ========================================================
// GT
// ========================================================

gtToggle.addEventListener(
    "change",
    function() {{

        gtLayer.style.display =
            this.checked
            ? "block"
            : "none";

        gtLegend.style.display =
            this.checked
            ? "block"
            : "none";

    }}
);


gtOpacity.addEventListener(
    "input",
    function() {{

        gtImage.style.opacity =
            this.value / 100;

        gtOpacityValue.innerText =
            this.value + "%";

    }}
);



// ========================================================
// PREDICTION
// ========================================================

predictionToggle.addEventListener(
    "change",
    function() {{

        predictionLayer.style.display =
            this.checked
            ? "block"
            : "none";

        predictionLegend.style.display =
            this.checked
            ? "block"
            : "none";

    }}
);


predictionOpacity.addEventListener(
    "input",
    function() {{

        predictionImage.style.opacity =
            this.value / 100;

        predictionOpacityValue.innerText =
            this.value + "%";

    }}
);



// ========================================================
// SWIPE
// ========================================================

swipeLayer.addEventListener(
    "change",
    function() {{

        if (this.value === "none") {{

            swipe.style.display =
                "none";

            gtImage.style.clipPath =
                "none";

            predictionImage.style.clipPath =
                "none";

            return;

        }}


        swipe.style.display =
            "block";


        updateSwipe();

    }}
);



function updateSwipe() {{

    swipe.style.left =
        swipePosition + "%";


    const clip =
        "inset(0 " +
        (100 - swipePosition) +
        "% 0 0)";


    if (
        swipeLayer.value === "gt"
    ) {{

        gtImage.style.clipPath =
            clip;

        predictionImage.style.clipPath =
            "none";

    }}


    if (
        swipeLayer.value === "prediction"
    ) {{

        predictionImage.style.clipPath =
            clip;

        gtImage.style.clipPath =
            "none";

    }}

}}



// ========================================================
// SWIPE MOUSE
// ========================================================

swipe.addEventListener(
    "mousedown",
    function() {{

        draggingSwipe = true;

    }}
);


document.addEventListener(
    "mouseup",
    function() {{

        draggingSwipe = false;

    }}
);


document.addEventListener(
    "mousemove",
    function(event) {{

        if (!draggingSwipe)
            return;

        const rect =
            viewer.getBoundingClientRect();

        let x =
            event.clientX -
            rect.left;

        swipePosition =
            (
                x /
                rect.width
            ) * 100;

        swipePosition =
            Math.max(
                0,
                Math.min(
                    100,
                    swipePosition
                )
            );

        updateSwipe();

    }}
);



// ========================================================
// SWIPE TOUCH
// ========================================================

swipe.addEventListener(
    "touchstart",
    function() {{

        draggingSwipe = true;

    }}
);


document.addEventListener(
    "touchend",
    function() {{

        draggingSwipe = false;

    }}
);


document.addEventListener(
    "touchmove",
    function(event) {{

        if (!draggingSwipe)
            return;

        const rect =
            viewer.getBoundingClientRect();

        let x =
            event.touches[0].clientX -
            rect.left;

        swipePosition =
            (
                x /
                rect.width
            ) * 100;

        swipePosition =
            Math.max(
                0,
                Math.min(
                    100,
                    swipePosition
                )
            );

        updateSwipe();

    }}
);



// ========================================================
// RESET
// ========================================================

function resetViewer() {{

    gtToggle.checked =
        false;

    predictionToggle.checked =
        false;

    gtLayer.style.display =
        "none";

    predictionLayer.style.display =
        "none";

    gtLegend.style.display =
        "none";

    predictionLegend.style.display =
        "none";

    gtOpacity.value =
        55;

    predictionOpacity.value =
        55;

    gtImage.style.opacity =
        0.55;

    predictionImage.style.opacity =
        0.55;

    gtOpacityValue.innerText =
        "55%";

    predictionOpacityValue.innerText =
        "55%";

    swipeLayer.value =
        "none";

    swipe.style.display =
        "none";

    gtImage.style.clipPath =
        "none";

    predictionImage.style.clipPath =
        "none";

    swipePosition =
        50;

}}

</script>

</body>

</html>
"""

    components.html(
        html,
        height=height,
        scrolling=False,
    )


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def create_download_zip(
    dataset_name,
    rgb,
    gt_rgb=None,
    prediction_rgb=None,
    gt_overlay=None,
    prediction_overlay=None,
):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:

        # RGB
        z.writestr(
            f"{dataset_name}_input_rgb.png",
            rgb_array_to_png_bytes(rgb),
        )

        # GT
        if gt_rgb is not None:

            z.writestr(
                f"{dataset_name}_ground_truth.png",
                rgb_array_to_png_bytes(gt_rgb),
            )

        # Prediction
        if prediction_rgb is not None:

            z.writestr(
                f"{dataset_name}_prediction.png",
                rgb_array_to_png_bytes(
                    prediction_rgb
                ),
            )

        # GT overlay
        if gt_overlay is not None:

            z.writestr(
                f"{dataset_name}_gt_overlay.png",
                rgb_array_to_png_bytes(
                    gt_overlay
                ),
            )

        # Prediction overlay
        if prediction_overlay is not None:

            z.writestr(
                f"{dataset_name}_prediction_overlay.png",
                rgb_array_to_png_bytes(
                    prediction_overlay
                ),
            )

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ============================================================
# SESSION STATE
# ============================================================

if "selected_paths" not in st.session_state:

    st.session_state.selected_paths = set()


if "prediction_results" not in st.session_state:

    st.session_state.prediction_results = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # st.header("⚙️ Settings")

    # st.caption(
    #     f"Images:\n{IMAGE_DIR}"
    # )

    # st.caption(
    #     f"Masks:\n{MASK_DIR}"
    # )

    # st.divider()

    # st.subheader(
    #     "Ground Truth Mask Encoding"
    # )

    # mask_id_mode = st.selectbox(
    #     "Mask label IDs",
    #     [
    #         "0-based",
    #         "1-based",
    #     ],
    #     index=0,
    #     help=(
    #         "Choose how pixel values in the "
    #         "mask correspond to the class list."
    #     ),
    # )

    # st.divider()

    # st.subheader(
    #     "Prediction"
    # )

    similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=ABS_SIMILARITY_THRESHOLD,
        step=0.01,
    )


# ============================================================
# DATASET GALLERY
# ============================================================

@st.dialog(
    "Select Prithvi HLS Datasets",
    width="large",
)
def open_image_picker():

    if not tif_files:

        st.error(
            f"No TIFF files found in:\n{IMAGE_DIR}"
        )

        return

    st.write(
        "Select any number of datasets."
    )

    cols = st.columns(2)

    for idx, path in enumerate(
        tif_files
    ):

        col = cols[idx % 2]

        selected = (
            path
            in st.session_state.selected_paths
        )

        with col:

            with st.container(
                border=True
            ):

                image = (
                    process_geotiff_to_rgb(
                        path
                    )
                )

                st.image(
                    image,
                    use_container_width=True,
                )

                filename = os.path.basename(
                    path
                )

                mask_path = (
                    find_mask_for_image(
                        path
                    )
                )

                if mask_path:

                    st.caption(
                        f"🟢 Mask: "
                        f"{os.path.basename(mask_path)}"
                    )

                else:

                    st.caption(
                        "🔴 Mask not found"
                    )

                button_label = (
                    "✅ Selected"
                    if selected
                    else "➕ Select"
                )

                if st.button(
                    f"{filename} | {button_label}",
                    key=f"dataset_{idx}",
                    use_container_width=True,
                ):

                    if selected:

                        st.session_state.selected_paths.remove(
                            path
                        )

                    else:

                        st.session_state.selected_paths.add(
                            path
                        )

                    st.rerun()

    st.divider()

    if st.button(
        "Apply Selection",
        type="primary",
        use_container_width=True,
    ):

        st.rerun()


# ============================================================
# GALLERY BUTTON
# ============================================================

if st.button(
    "🗂️ Open Dataset Gallery",
    use_container_width=False,
):

    open_image_picker()


# ============================================================
# SELECTED DATASETS
# ============================================================

st.divider()

st.subheader(
    "Selected Datasets"
)


if not st.session_state.selected_paths:

    st.info(
        "No HLS datasets selected."
    )

else:

    selected_list = sorted(
        st.session_state.selected_paths
    )

    st.success(
        f"{len(selected_list)} dataset(s) selected."
    )

    preview_cols = st.columns(
        min(
            len(selected_list),
            3,
        )
    )

    for idx, path in enumerate(
        selected_list
    ):

        with preview_cols[
            idx % 3
        ]:

            image = (
                process_geotiff_to_rgb(
                    path
                )
            )

            st.image(
                image,
                caption=os.path.basename(
                    path
                ),
                use_container_width=True,
            )


# ============================================================
# PROMPT INPUT
# ============================================================

st.divider()

st.subheader(
    "🧠 Interactive Text-Prompted Prediction"
)

prompt_text = st.text_input(
    "Enter comma-separated prompts",
    value=(
        "forest, open water, "
        "corn, developed"
    ),
)

import re

user_phrases = [
    normalize_prompt(phrase)
    for phrase in re.split(r'\s*(?:,|\band\b)\s*', prompt_text, flags=re.IGNORECASE)
    if phrase.strip()
]


# ============================================================
# RUN PREDICTION
# ============================================================

if st.button(
    "🚀 Run Prediction",
    type="primary",
    use_container_width=True,
):

    if not st.session_state.selected_paths:

        st.warning(
            "Please select at least one dataset."
        )

        st.stop()

    if not user_phrases:

        st.error(
            "Enter at least one prompt."
        )

        st.stop()

    selected_list = sorted(
        st.session_state.selected_paths
    )

    tensor_list = []

    for path in selected_list:

        tensor_list.append(
            process_geotiff_to_tensor(
                path
            )
        )

    try:

        batch_images = torch.stack(
            tensor_list,
            dim=0,
        )

    except RuntimeError as error:

        st.error(
            "The selected images do not have "
            "matching dimensions.\n\n"
            f"{error}"
        )

        st.stop()

    batch_images = batch_images.to(
        device=device,
        dtype=torch.float32,
    )

    phrase_colors = get_phrase_colors(
        user_phrases
    )

    results = {}

    progress = st.progress(
        0
    )

    with st.spinner(
        "Running Prithvi VLM inference..."
    ):

        for idx, path in enumerate(
            selected_list
        ):

            single_image = (
                batch_images[
                    idx:idx + 1
                ]
            )

            # ================================================
            # MODEL
            # ================================================

            with torch.no_grad():

                outputs = prompt_model(
                    single_image,
                    prompts=user_phrases,
                )

                if (
                    isinstance(
                        outputs,
                        dict,
                    )
                    and
                    "logits"
                    in outputs
                ):

                    logits = outputs[
                        "logits"
                    ]

                elif hasattr(
                    outputs,
                    "logits",
                ):

                    logits = outputs.logits

                else:

                    logits = outputs

                scale = (
                    prompt_model.logit_scale
                    .exp()
                    .clamp(
                        max=100
                    )
                )

                raw_similarity = (
                    logits / scale
                )

                max_similarity, pred_local = (
                    torch.max(
                        raw_similarity,
                        dim=1,
                    )
                )

                pred_local = (
                    pred_local[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                max_similarity = (
                    max_similarity[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

            # ================================================
            # THRESHOLD
            # ================================================

            filtered_prediction = (
                pred_local.copy()
            )

            filtered_prediction[
                max_similarity
                < similarity_threshold
            ] = -1

            prediction_rgb = (
                colorize_labels(
                    filtered_prediction,
                    phrase_colors,
                )
            )

            # ================================================
            # RGB
            # ================================================

            pil_rgb = (
                process_geotiff_to_rgb(
                    path
                )
            )

            raw_rgb = (
                np.asarray(
                    pil_rgb
                ).astype(
                    np.float32
                ) / 255.0
            )

            # ================================================
            # GT
            # ================================================

            gt_label = None
            gt_rgb = None
            mask_path = None

            try:

                raw_gt, mask_path = (
                    load_ground_truth_label(
                        path
                    )
                )

                gt_label = (
                    raw_gt
                    # convert_mask_ids(
                    #     raw_gt,
                    #     mask_id_mode,
                    # )
                )

                gt_rgb = (
                    colorize_labels(
                        gt_label,
                        CLASS_COLORS,
                    )
                )

            except FileNotFoundError:

                pass

            # ================================================
            # OVERLAYS
            # ================================================

            gt_overlay = None

            if gt_rgb is not None:

                gt_overlay = (
                    0.5 * raw_rgb
                    +
                    0.5 * gt_rgb
                )

            prediction_overlay = (
                0.5 * raw_rgb
                +
                0.5 * prediction_rgb
            )

            # ================================================
            # LEGENDS
            # ================================================

            gt_legend = []

            if gt_label is not None:

                for label in np.unique(
                    gt_label
                ):

                    label = int(label)

                    if (
                        0 <= label
                        < len(CLASS_NAMES)
                    ):

                        gt_legend.append(
                            (
                                CLASS_NAMES[label],
                                CLASS_COLORS[label],
                            )
                        )

            prediction_legend = []

            for idx_prompt, phrase in enumerate(
                user_phrases
            ):
                print("idx_prompt", idx_prompt)

                prediction_legend.append(
                    (
                        phrase,
                        phrase_colors[
                            idx_prompt
                        ],
                    )
                )

            # ================================================
            # SAVE RESULTS
            # ================================================

            results[path] = {
                "rgb": raw_rgb,
                "gt": gt_rgb,
                "prediction": prediction_rgb,
                "gt_overlay": gt_overlay,
                "prediction_overlay": prediction_overlay,
                "gt_label": gt_label,
                "mask_path": mask_path,
                "gt_legend": gt_legend,
                "prediction_legend": prediction_legend,
                "max_similarity": max_similarity,
            }

            progress.progress(
                (idx + 1)
                / len(selected_list)
            )

    st.session_state.prediction_results = (
        results
    )

    st.success(
        "Prediction complete."
    )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.prediction_results
):

    st.divider()

    st.header(
        "🔬 Results"
    )

    for path, result in (
        st.session_state
        .prediction_results
        .items()
    ):

        dataset_name = os.path.splitext(
            os.path.basename(path)
        )[0]

        st.subheader(
            dataset_name
        )

        rgb = result["rgb"]
        gt_rgb = result["gt"]
        prediction_rgb = result["prediction"]

        gt_overlay = result[
            "gt_overlay"
        ]

        prediction_overlay = result[
            "prediction_overlay"
        ]

        gt_label = result[
            "gt_label"
        ]

        gt_legend = result[
            "gt_legend"
        ]

        prediction_legend = result[
            "prediction_legend"
        ]

        # # ====================================================
        # # OVERVIEW
        # # ====================================================

        # overview_cols = st.columns(3)

        # with overview_cols[0]:

        #     st.image(
        #         rgb,
        #         caption="Input RGB",
        #         use_container_width=True,
        #     )

        # with overview_cols[1]:

        #     if gt_rgb is not None:

        #         st.image(
        #             gt_rgb,
        #             caption="Ground Truth",
        #             use_container_width=True,
        #         )

        #     else:

        #         st.warning(
        #             "Ground truth mask not found."
        #         )

        # with overview_cols[2]:

        #     st.image(
        #         prediction_rgb,
        #         caption="Prediction",
        #         use_container_width=True,
        #     )

        # ====================================================
        # INTERACTIVE QGIS VIEWER
        # ====================================================

        st.markdown(
            "### 🗺️ Interactive Layer Viewer"
        )

        st.caption(
            "Turn Ground Truth and Prediction on/off "
            "independently. Adjust opacity or use Swipe "
            "to compare a layer against RGB."
        )

        render_layer_viewer(
            rgb=rgb,
            gt_rgb=gt_rgb,
            prediction_rgb=prediction_rgb,
            gt_legend=gt_legend,
            prediction_legend=prediction_legend,
            height=650,
        )

        # ====================================================
        # GROUND TRUTH LABELS
        # ====================================================

        if gt_label is not None:

            with st.expander(
                "Ground Truth Label Information",
                expanded=False,
            ):

                unique_labels = np.unique(
                    gt_label
                )

                label_data = []

                for label in unique_labels:

                    label = int(label)

                    if (
                        0 <= label
                        < len(CLASS_NAMES)
                    ):

                        pixel_count = int(
                            np.sum(
                                gt_label
                                == label
                            )
                        )

                        percentage = (
                            pixel_count
                            /
                            gt_label.size
                            * 100
                        )

                        label_data.append(
                            (
                                label,
                                CLASS_NAMES[label],
                                pixel_count,
                                percentage,
                            )
                        )

                if label_data:

                    st.dataframe(
                        {
                            "Label ID": [
                                x[0]
                                for x in label_data
                            ],
                            "Class": [
                                x[1]
                                for x in label_data
                            ],
                            "Pixels": [
                                x[2]
                                for x in label_data
                            ],
                            "Area %": [
                                round(
                                    x[3],
                                    2,
                                )
                                for x in label_data
                            ],
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

        # ====================================================
        # DOWNLOADS
        # ====================================================

        st.markdown(
            "### 💾 Downloads"
        )

        download_cols = st.columns(5)

        # --------------------------------------------
        # RGB
        # --------------------------------------------

        with download_cols[0]:

            st.download_button(
                "⬇️ RGB",
                data=rgb_array_to_png_bytes(
                    rgb
                ),
                file_name=(
                    f"{dataset_name}_rgb.png"
                ),
                mime="image/png",
                key=f"rgb_{dataset_name}",
                use_container_width=True,
            )

        # --------------------------------------------
        # GT
        # --------------------------------------------

        with download_cols[1]:

            if gt_rgb is not None:

                st.download_button(
                    "⬇️ GT",
                    data=rgb_array_to_png_bytes(
                        gt_rgb
                    ),
                    file_name=(
                        f"{dataset_name}_ground_truth.png"
                    ),
                    mime="image/png",
                    key=f"gt_{dataset_name}",
                    use_container_width=True,
                )

        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        with download_cols[2]:

            st.download_button(
                "⬇️ Prediction",
                data=rgb_array_to_png_bytes(
                    prediction_rgb
                ),
                file_name=(
                    f"{dataset_name}_prediction.png"
                ),
                mime="image/png",
                key=f"prediction_{dataset_name}",
                use_container_width=True,
            )

        # --------------------------------------------
        # GT Overlay
        # --------------------------------------------

        with download_cols[3]:

            if gt_overlay is not None:

                st.download_button(
                    "⬇️ GT Overlay",
                    data=rgb_array_to_png_bytes(
                        gt_overlay
                    ),
                    file_name=(
                        f"{dataset_name}_gt_overlay.png"
                    ),
                    mime="image/png",
                    key=f"gt_overlay_{dataset_name}",
                    use_container_width=True,
                )

        # --------------------------------------------
        # Prediction Overlay
        # --------------------------------------------

        with download_cols[4]:

            st.download_button(
                "⬇️ Pred Overlay",
                data=rgb_array_to_png_bytes(
                    prediction_overlay
                ),
                file_name=(
                    f"{dataset_name}_prediction_overlay.png"
                ),
                mime="image/png",
                key=f"pred_overlay_{dataset_name}",
                use_container_width=True,
            )

        # ====================================================
        # ZIP
        # ====================================================

        zip_bytes = create_download_zip(
            dataset_name=dataset_name,
            rgb=rgb,
            gt_rgb=gt_rgb,
            prediction_rgb=prediction_rgb,
            gt_overlay=gt_overlay,
            prediction_overlay=prediction_overlay,
        )

        st.download_button(
            "📦 Download All Images for This Dataset",
            data=zip_bytes,
            file_name=(
                f"{dataset_name}_results.zip"
            ),
            mime="application/zip",
            key=f"zip_{dataset_name}",
            use_container_width=True,
        )

        # ====================================================
        # SIMILARITY
        # ====================================================

        similarity = result[
            "max_similarity"
        ]

        with st.expander(
            "Prediction Statistics",
            expanded=False,
        ):

            st.write(
                f"Minimum similarity: "
                f"`{similarity.min():.4f}`"
            )

            st.write(
                f"Maximum similarity: "
                f"`{similarity.max():.4f}`"
            )

            st.write(
                f"Mean similarity: "
                f"`{similarity.mean():.4f}`"
            )

        st.divider()