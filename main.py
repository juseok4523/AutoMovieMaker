import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def process_video_chunk(args):
    video_path, template, start_frame, end_frame, frame_rate, threshold, progress_callback = args
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    match_times = []
    
    for frame_count in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            match_times.append(frame_count / frame_rate)
        progress_callback()
    
    cap.release()
    return match_times

def find_image_in_video(video_path, template_path, threshold=0.8):
    cap = cv2.VideoCapture(video_path)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    if template is None:
        raise ValueError("Template image not found.")
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    num_workers = cpu_count()
    chunk_size = total_frames // num_workers
    progress_var.set(0)
    progress_step = 100 / total_frames
    
    def update_progress():
        progress_var.set(progress_var.get() + progress_step)

    tasks = [(video_path, template, i, min(i + chunk_size, total_frames), frame_rate, threshold, update_progress) 
             for i in range(0, total_frames, chunk_size)]
    
    with Pool(num_workers) as pool:
        results = pool.map(process_video_chunk, tasks)
    
    match_times = [time for sublist in results for time in sublist]
    
    return match_times

def select_video():
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    video_entry.delete(0, tk.END)
    video_entry.insert(0, file_path)

def select_template():
    file_path = filedialog.askopenfilename(title="Select Template Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    template_entry.delete(0, tk.END)
    template_entry.insert(0, file_path)

def start_processing():
    video_file = video_entry.get()
    template_file = template_entry.get()
    
    if not video_file or not template_file:
        messagebox.showerror("Error", "Please select both video and template image files.")
        return
    
    match_times = find_image_in_video(video_file, template_file)
    
    if match_times:
        result_label.config(text=f"Image found at times (seconds): {match_times}")
    else:
        result_label.config(text="Image not found in the video.")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Video Template Matching")

    tk.Label(root, text="Video File:").grid(row=0, column=0, padx=10, pady=10)
    video_entry = tk.Entry(root, width=50)
    video_entry.grid(row=0, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=select_video).grid(row=0, column=2, padx=10, pady=10)

    tk.Label(root, text="Template Image:").grid(row=1, column=0, padx=10, pady=10)
    template_entry = tk.Entry(root, width=50)
    template_entry.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=select_template).grid(row=1, column=2, padx=10, pady=10)

    tk.Button(root, text="Start", command=start_processing).grid(row=2, column=1, padx=10, pady=20)
    
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.grid(row=3, column=1, padx=10, pady=10)
    
    result_label = tk.Label(root, text="")
    result_label.grid(row=4, column=1, padx=10, pady=10)

    root.mainloop()