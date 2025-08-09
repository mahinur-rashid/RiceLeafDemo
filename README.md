# Rice Disease Detection Flask App

A web app for classifying rice leaf diseases using a deep learning model (EfficientNet-V2-L + attention). Upload an image to get a prediction.

## Features
- Upload rice leaf images via web UI or API
- Predicts disease class and confidence
- Uses a state-of-the-art PyTorch model
- Clean, production-ready Flask structure

## Setup

1. **Clone this repo**
2. **Install dependencies**

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

3. **Add your trained model**

Place your `best_rice_model_regularized.pth` file in the project root.

4. **Run the app**

```bash
python app.py
```

5. **Open in browser**

Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## API Usage

POST an image to `/api/predict`:

```bash
curl -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:5000/api/predict
```

## Project Structure

```
RiceLeafDemo/
├── app.py
├── model_utils.py
├── requirements.txt
├── .gitignore
├── best_rice_model_regularized.pth
├── static/
│   └── uploads/
├── templates/
│   ├── index.html
│   └── result.html
└── README.md
```

## License
MIT
