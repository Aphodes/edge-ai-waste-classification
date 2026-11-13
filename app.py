"""TrashNet Surface Pro demo. Windows/Tkinter + ONNX Runtime CPU."""
import csv
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
RESULTS = ROOT / 'benchmark_results'
RESULTS.mkdir(exist_ok=True)
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
MODEL_NAMES = {
    'FP32 baseline': 'baseline.onnx',
    'Static INT8': 'static_int8.onnx',
    'Unstructured 5% + INT8': 'unstructured_5_static_int8.onnx',
    'Selective structured (6 groups)': 'selective_structured_6groups.onnx',
    'Selective structured + INT8': 'selective_structured_6groups_static_int8.onnx',
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess(rgb, size=224):
    # Match torchvision Resize((224,224)) + ToTensor + Normalize(ImageNet).
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    return np.ascontiguousarray(((array - MEAN) / STD).transpose(2, 0, 1)[None])


def load_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    shape = sess.get_inputs()[0].shape
    if len(shape) != 4 or shape[1] != 3 or (isinstance(shape[2], int) and shape[2] != 224) or (isinstance(shape[3], int) and shape[3] != 224):
        raise ValueError(f'Unexpected input shape: {shape}; expected NCHW with 224x224 images')
    if sess.get_outputs()[0].shape[-1] != 6:
        raise ValueError('Model does not have six output classes')
    return sess


class App:
    def __init__(self, root):
        self.root = root
        root.title('TrashNet | Surface Edge AI')
        root.geometry('1030x780')
        root.configure(bg='white')
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.session = None
        self.camera = None
        self.running = False
        self.last_frame_time = None
        self.model_var = tk.StringVar(value='FP32 baseline')
        self.camera_var = tk.StringVar(value='0')
        self.status_var = tk.StringVar(value='Ready. Select a model, then start the camera.')
        self.pred_var = tk.StringVar(value='—')
        self.conf_var = tk.StringVar(value='—')
        self.infer_var = tk.StringVar(value='—')
        self.fps_var = tk.StringVar(value='—')
        self.backend_var = tk.StringVar(value='CPUExecutionProvider')
        self.make_ui()
        self.root.after(60, self.poll)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def make_ui(self):
        ttk.Style().theme_use('clam')
        header = tk.Frame(self.root, bg='white')
        header.pack(fill='x', padx=25, pady=(20, 10))
        tk.Label(header, text='TrashNet  |  Edge AI Waste Classification', font=('Arial', 22, 'bold'), bg='white').pack(anchor='w')
        tk.Label(header, text='Single-item camera classification and reproducible model benchmarks', font=('Arial', 11), fg='#555', bg='white').pack(anchor='w', pady=(5, 0))
        controls = tk.Frame(self.root, bg='white')
        controls.pack(fill='x', padx=25, pady=12)
        tk.Label(controls, text='Model:', bg='white', font=('Arial', 11)).pack(side='left')
        self.model_box = ttk.Combobox(controls, textvariable=self.model_var, values=list(MODEL_NAMES), state='readonly', width=34)
        self.model_box.pack(side='left', padx=(8, 14))
        self.model_box.bind('<<ComboboxSelected>>', lambda _: self.change_model())
        tk.Label(controls, text='Camera:', bg='white').pack(side='left')
        ttk.Entry(controls, textvariable=self.camera_var, width=4).pack(side='left', padx=6)
        ttk.Button(controls, text='Start camera', command=self.start).pack(side='left', padx=5)
        ttk.Button(controls, text='Stop', command=self.stop).pack(side='left', padx=5)
        ttk.Button(controls, text='Benchmark all', command=self.benchmark).pack(side='right', padx=5)
        body = tk.Frame(self.root, bg='white')
        body.pack(fill='both', expand=True, padx=25)
        self.video = tk.Label(body, text='Camera preview\n\nPress Start camera', bg='#e9edf1', fg='#555', font=('Arial', 16), width=75, height=21)
        self.video.pack(side='left', fill='both', expand=True)
        side = tk.Frame(body, bg='white', width=235)
        side.pack(side='left', fill='y', padx=(20, 0))
        for label, variable in [('Prediction', self.pred_var), ('Confidence', self.conf_var), ('Inference time', self.infer_var), ('Camera FPS', self.fps_var), ('Execution backend', self.backend_var)]:
            tk.Label(side, text=label.upper(), font=('Arial', 9, 'bold'), fg='#666', bg='white').pack(anchor='w', pady=(20, 3))
            tk.Label(side, textvariable=variable, font=('Arial', 17 if label=='Prediction' else 13, 'bold'), bg='white', wraplength=220, justify='left').pack(anchor='w')
        tk.Label(self.root, textvariable=self.status_var, bg='#f0f2f5', fg='#222', anchor='w', font=('Arial', 10), padx=14, pady=10).pack(fill='x', padx=25, pady=(12, 10))
        tk.Label(self.root, text='Note: classifications are for six TrashNet material categories; this app does not determine local recycling eligibility.', fg='#666', bg='white', font=('Arial', 9)).pack(anchor='w', padx=25, pady=(0, 12))

    def change_model(self):
        # Never change session while the worker is using it.
        if self.running:
            self.stop()
            self.status_var.set('Camera stopped. Click Start camera to load the newly selected model.')

    def start(self):
        if self.running or (self.worker and self.worker.is_alive()):
            return
        name = self.model_var.get()
        path = MODELS / MODEL_NAMES[name]
        if not path.exists():
            messagebox.showwarning('Model missing', f'Missing {path.name}\n\nCopy the ONNX file into the models folder.')
            return
        try:
            index = int(self.camera_var.get())
        except ValueError:
            messagebox.showerror('Camera', 'Camera index must be an integer.')
            return
        self.stop_event.clear()
        self.running = True
        self.status_var.set(f'Loading {name}...')
        self.worker = threading.Thread(target=self.camera_loop, args=(path, index), daemon=True)
        self.worker.start()

    def camera_loop(self, path, index):
        cap = None
        try:
            sess = load_session(path)
            input_name = sess.get_inputs()[0].name
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError('Could not open camera. Try camera index 1 or close other camera apps.')
            self.events.put(('status', f'Running {path.name} on CPU (local inference)'))
            last = time.perf_counter()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera stopped returning frames')
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x = preprocess(rgb)
                start = time.perf_counter()
                logits = sess.run(None, {input_name: x})[0][0]
                infer_ms = (time.perf_counter() - start) * 1000
                scores = np.exp(logits - np.max(logits))
                probs = scores / scores.sum()
                idx = int(np.argmax(probs))
                now = time.perf_counter()
                fps = 1.0 / max(now - last, 1e-6)
                last = now
                # Limit preview queue to avoid latency accumulating.
                self.events.put(('frame', rgb, CLASSES[idx], float(probs[idx]), infer_ms, fps))
        except Exception as exc:
            self.events.put(('error', str(exc)))
        finally:
            if cap is not None:
                cap.release()
            self.events.put(('stopped',))

    def poll(self):
        latest = None
        try:
            while True:
                msg = self.events.get_nowait()
                if msg[0] == 'frame':
                    latest = msg
                elif msg[0] == 'status':
                    self.status_var.set(msg[1])
                elif msg[0] == 'error':
                    self.status_var.set('Error: ' + msg[1])
                elif msg[0] == 'stopped':
                    self.running = False
        except queue.Empty:
            pass
        if latest:
            _, rgb, cls, confidence, ms, fps = latest
            h, w = rgb.shape[:2]
            scale = min(720 / w, 520 / h)
            frame = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video.configure(image=img, text='')
            self.video.image = img
            self.pred_var.set(cls.title())
            self.conf_var.set(f'{confidence * 100:.1f}%')
            self.infer_var.set(f'{ms:.2f} ms')
            self.fps_var.set(f'{fps:.1f} FPS')
        self.root.after(60, self.poll)

    def stop(self):
        self.stop_event.set()
        self.status_var.set('Stopping camera...')

    def benchmark(self):
        if self.running or (self.worker and self.worker.is_alive()):
            messagebox.showinfo('Benchmark', 'Stop the camera first, then click Benchmark all.')
            return
        self.status_var.set('Benchmarking available models in isolated processes; please wait...')
        threading.Thread(target=self.benchmark_worker, daemon=True).start()

    def benchmark_worker(self):
        try:
            command = [sys.executable, str(ROOT / 'benchmark.py'), '--all', '--repeats', '500', '--trials', '5']
            proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                self.events.put(('error', (proc.stderr or proc.stdout)[-1800:]))
            else:
                self.events.put(('status', 'Benchmark complete. Results saved in benchmark_results/benchmark_summary.csv'))
        except Exception as exc:
            self.events.put(('error', str(exc)))

    def close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
