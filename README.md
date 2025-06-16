# 🖼️ Image Processing Pipeline (FastAPI + Tailwind UI)

This is a **FastAPI web application** for composing and running image processing pipelines through a visually interactive UI.
It enables uploading an image, choosing processing steps, adjusting parameters, and viewing results live in the browser.

---

## 🚀 Features

* Upload and preview images 
* Compose image pipelines using modular steps
* Adjust parameters for each step (e.g. brightness, hue, blur)
* TailwindCSS enhanced glassmorphism UI with gradient animations
* Live process and view results side-by-side
* Backend powered by FastAPI and OpenCV

---

## Built-in Processing Steps

| Step        | Description             |
| ----------- | ----------------------- |
| brightness  | Adjust image brightness |
| saturation  | Adjust image saturation |
| hue         | Adjust image hue        |
| boxblur     | Apply box blur filter   |
| unsharpmask | Apply unsharp mask      |
| crop        | Crop the image          |
| rotate      | Rotate the image        |

> Steps are defined in `image_processing.py` via `ImageProcessingPipeline`.

---

## File Structure

```
├── app.py                 # FastAPI app + embedded HTML UI
├── image_processing.py    # Processing pipeline & step classes
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/Cpatel2000/GLASS_IMAGING.git
cd GLASS-IMAGING
```

2. **Set up environment**

```bash
conda create --name glass python=3.9.6 
conda activate glass
pip install -r requirements.txt
```

3. **Run the app**

```bash
python app.py
```

4. **Visit**

```
http://localhost:8000
```

---

## Security

* Intended for private/internal use


---

## 📦 Dependencies

* `fastapi`
* `uvicorn`
* `opencv-python`
* `numpy`

Install with:

```bash
pip install -r requirements.txt
```

---


