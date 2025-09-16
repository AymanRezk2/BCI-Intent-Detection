# Deployment

This folder contains code for real-time model deployment:

- `app.py` → Web interface (FastAPI / Streamlit) to interact with the model.
- `websocket_server.py` → Send predictions to external control systems.
- `edge_inference.py` → Optimized inference for edge devices (ONNX, TensorRT, Jetson, Raspberry Pi).

👉 This makes the model usable in assistive technology applications (e.g., wheelchair control, speller apps).
