import cv2
import numpy as np
import face_recognition
import os
import csv
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import pandas as pd
import dlib
from scipy.spatial import distance as dist
import logging
from PIL import Image, ImageTk
import threading

logging.basicConfig(filename='attendance.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = {
    'image_path': 'student_faces',
    'timetable_file': 'timetable.json',
    'face_distance_threshold': 0.6,
    'blink_threshold': 2,
    'ear_threshold_min': 0.2,
    'ear_threshold_max': 0.3,
    'frame_history': 10,
    'camera_index': 0
}

if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    logging.error("Missing shape_predictor_68_face_landmarks.dat")
    raise FileNotFoundError("Missing shape_predictor_68_face_landmarks.dat")

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    logging.error(f"Dlib init failed: {e}")
    raise

def load_images():
    images, rolls = [], []
    for f in os.listdir(CONFIG['image_path']):
        img = cv2.imread(os.path.join(CONFIG['image_path'], f))
        if img is not None:
            images.append(img)
            rolls.append(os.path.splitext(f)[0])
    logging.info(f"Loaded {len(images)} images")
    return images, rolls

images, studentRollNumbers = load_images()

def find_encodings(images):
    encode_list = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            encode_list.append(encodings[0])
    return encode_list

def save_timetable(timetable):
    with open(CONFIG['timetable_file'], 'w') as f:
        json.dump(timetable, f, indent=2)

def load_timetable():
    try:
        with open(CONFIG['timetable_file'], 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_current_class(timetable):
    now = datetime.now()
    day, time = now.strftime('%A')[:3], now.strftime('%H:%M')
    if day in timetable:
        for slot in timetable[day]:
            start = datetime.strptime(slot["start"], '%H:%M').time()
            end = datetime.strptime(slot["end"], '%H:%M').time()
            current = datetime.strptime(time, '%H:%M').time()
            if start <= current <= end:
                return slot["class"]
    return None

def mark_attendance(roll, class_name):
    if not class_name:
        return None
    filename = f'Attendance_{class_name}_{datetime.now().strftime("%Y-%m-d")}.csv'
    with open(filename, 'a+', newline='') as f:
        f.seek(0)
        rolls = [row[0] for row in csv.reader(f) if row]
        if not rolls:
            csv.writer(f).writerow(['Roll Number', 'Time'])
        if roll not in rolls:
            dt = datetime.now().strftime('%H:%M:%S')
            csv.writer(f).writerow([roll, dt])
            return f"Marked {roll} in {class_name} at {dt}"
    return None

def collect_attendance_data():
    data, rolls, detailed, records = defaultdict(lambda: defaultdict(int)), defaultdict(int), defaultdict(lambda: defaultdict(int)), []
    for f in [f for f in os.listdir('.') if f.startswith('Attendance_') and f.endswith('.csv')]:
        subject, date = f.split('_')[1], f.split('_')[2].replace('.csv', '')
        with open(f, 'r') as rf:
            next(csv.reader(rf))
            for row in csv.reader(rf):
                if row:
                    roll, time = row
                    data[subject][date] += 1
                    rolls[roll] += 1
                    detailed[roll][subject] += 1
                    records.append({'Roll Number': roll, 'Subject': subject, 'Day': date, 'Time': time})
    return data, rolls, detailed, records

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def check_liveness(img, face_loc):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y1, x2, y2, x1 = face_loc
    rect = dlib.rectangle(x1, y1, x2, y2)
    shape = [(predictor(gray, rect).part(i).x, predictor(gray, rect).part(i).y) for i in range(68)]
    left_ear = eye_aspect_ratio(shape[36:42])
    right_ear = eye_aspect_ratio(shape[42:48])
    return (left_ear + right_ear) / 2.0

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LookIn")
        self.root.geometry("900x600")
        self.timetable = load_timetable()
        self.encode_list = find_encodings(images)
        self.cap = None
        self.is_running = False

        style = ttk.Style()
        style.theme_use('clam')
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill='both', expand=True)

        ttk.Label(main_frame, text="LookIn", font=("Arial", 18, "bold")).pack(pady=10)
        ttk.Label(main_frame, text="Done by 21,48,50", font=("Arial", 14, "bold")).pack(pady=10)
        self.status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(main_frame, textvariable=self.status_var).pack(pady=5)
        self.video_label = ttk.Label(main_frame)
        self.video_label.pack(pady=10)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        for i, (text, cmd) in enumerate([
            ("Start", self.start_attendance), ("Stop", self.stop_attendance),
            ("Register Face", self.register_face), ("Edit Timetable", self.edit_timetable),
            ("Analytics", self.show_analytics), ("Report", self.generate_report),
            ("Export", self.export_attendance)
        ]):
            ttk.Button(button_frame, text=text, command=cmd).grid(row=i//4, column=i%4, padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill='both', expand=True, pady=10)
        self.log_text = tk.Text(log_frame, height=5, width=50)
        self.log_text.pack(fill='both', expand=True)

    def log(self, message, level='info'):
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {message}\n")
        self.log_text.see(tk.END)
        getattr(logging, level)(message)

    def register_face(self):
        win = tk.Toplevel(self.root)
        win.title("Register Face")
        win.geometry("500x500")
        cap = cv2.VideoCapture(CONFIG['camera_index'])
        if not cap.isOpened():
            messagebox.showerror("Error", "Webcam inaccessible")
            win.destroy()
            return
        roll_var = tk.StringVar()
        status_var = tk.StringVar(value="Center face, ensure good lighting")
        captured_frame = None
        is_captured = False

        os.makedirs(CONFIG['image_path'], exist_ok=True)
        ttk.Label(win, text="Instructions: Center face, one face only").pack(pady=5)
        preview_label = ttk.Label(win)
        preview_label.pack(pady=5)
        ttk.Label(win, text="Roll Number:").pack()
        ttk.Entry(win, textvariable=roll_var).pack(pady=5)
        ttk.Label(win, textvariable=status_var).pack(pady=5)
        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)
        capture_btn = ttk.Button(btn_frame, text="Capture", command=lambda: capture_frame())
        capture_btn.pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Save", command=lambda: self.save_face(win, roll_var, captured_frame, cap, status_var)).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=lambda: [cap.release(), win.destroy()]).pack(side='left', padx=5)

        def capture_frame():
            nonlocal captured_frame, is_captured
            capture_btn.state(['disabled'])
            ret, frame = cap.read()
            capture_btn.state(['!disabled'])
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb)
                if len(faces) == 1 and (faces[0][1] - faces[0][3]) > 100:
                    captured_frame = frame
                    is_captured = True
                    img = Image.fromarray(rgb).resize((150, 112))
                    photo = ImageTk.PhotoImage(img)
                    preview_label.configure(image=photo)
                    preview_label.image = photo
                    status_var.set("Captured, click Save or Capture again")
                else:
                    status_var.set("One face required, size >100px")

        def update_preview():
            if cap.isOpened() and not is_captured:
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = face_recognition.face_locations(rgb)
                    for (top, right, bottom, left) in faces:
                        cv2.rectangle(rgb, (left, top), (right, bottom), (0, 255, 0), 2)
                    img = Image.fromarray(rgb).resize((300, 225))
                    photo = ImageTk.PhotoImage(img)
                    preview_label.configure(image=photo)
                    preview_label.image = photo
                    status_var.set(f"Detected {len(faces)} face(s)" if faces else "No face detected")
                win.after(30, update_preview)

        update_preview()
        win.protocol("WM_DELETE_WINDOW", lambda: [cap.release(), win.destroy()])

    def save_face(self, win, roll_var, frame, cap, status_var):
        roll = roll_var.get().strip().upper()
        if not roll or not roll.isalnum():
            status_var.set("Invalid roll number")
            messagebox.showerror("Error", "Enter valid alphanumeric roll number")
            return
        if not frame:
            status_var.set("No frame captured")
            messagebox.showerror("Error", "Capture a frame first")
            return
        file_path = os.path.join(CONFIG['image_path'], f"{roll}.jpg")
        if os.path.exists(file_path) and not messagebox.askyesno("Confirm", f"Overwrite {roll}?"):
            status_var.set("Cancelled")
            return
        try:
            if not os.access(CONFIG['image_path'], os.W_OK):
                raise PermissionError("Cannot write to image directory")
            cv2.imwrite(file_path, frame)
            images, rolls = load_images()
            self.encode_list = find_encodings(images)
            self.log(f"Registered {roll}")
            messagebox.showinfo("Success", f"Registered {roll}")
            cap.release()
            win.destroy()
        except Exception as e:
            status_var.set(f"Error: {str(e)}")
            self.log(f"Register error: {e}", 'error')
            messagebox.showerror("Error", str(e))

    def edit_timetable(self):
        win = tk.Toplevel(self.root)
        win.title("Timetable Editor")
        win.geometry("800x500")
        notebook = ttk.Notebook(win)
        notebook.pack(fill='both', expand=True)

        subjects_frame = ttk.Frame(notebook)
        template_frame = ttk.Frame(notebook)
        manual_frame = ttk.Frame(notebook)
        notebook.add(subjects_frame, text="Subjects")
        notebook.add(template_frame, text="Template")
        notebook.add(manual_frame, text="Manual")

        subjects = sorted(set(slot['class'] for day in self.timetable.values() for slot in day)) if self.timetable else []
        subject_var = tk.StringVar()
        subject_combo = None  # Will store manual entry subject dropdown

        s_frame = ttk.Frame(subjects_frame, padding=5)
        s_frame.pack(fill='both', expand=True)
        ttk.Label(s_frame, text="Subject:").grid(row=0, column=0)
        ttk.Entry(s_frame, textvariable=subject_var).grid(row=0, column=1)
        subject_listbox = tk.Listbox(s_frame, height=5)
        subject_listbox.grid(row=1, column=0, columnspan=2, pady=5)

        def update_subject_list():
            subject_listbox.delete(0, tk.END)
            for s in subjects:
                subject_listbox.insert(tk.END, s)

        def update_subject_dropdown():
            if subject_combo:
                subject_combo['values'] = subjects

        def add_subject():
            new_subject = subject_var.get().strip().upper()
            if new_subject and new_subject not in subjects:
                subjects.append(new_subject)
                subjects.sort()
                update_subject_list()
                update_subject_dropdown()
                self.log(f"Added {new_subject}")
                subject_var.set("")
            elif new_subject:
                messagebox.showwarning("Warning", "Subject already exists")
            else:
                messagebox.showerror("Error", "Please enter a subject")

        def delete_subject():
            sel = subject_listbox.curselection()
            if sel and not any(slot['class'] == subject_listbox.get(sel[0]) for day in self.timetable.values() for slot in day):
                subjects.remove(subject_listbox.get(sel[0]))
                update_subject_list()
                update_subject_dropdown()
                self.log(f"Deleted {subject_listbox.get(sel[0])}")

        ttk.Button(s_frame, text="Add", command=add_subject).grid(row=0, column=2, padx=5)
        ttk.Button(s_frame, text="Delete", command=delete_subject).grid(row=1, column=2)
        update_subject_list()

        t_frame = ttk.Frame(template_frame, padding=5)
        t_frame.pack(fill='both', expand=True)
        periods = [("08:00", "08:50"), ("08:50", "09:40"), ("09:50", "10:40"), ("10:40", "11:30"),
                   ("12:20", "13:10"), ("13:10", "14:00"), ("14:00", "14:50"), ("14:50", "15:40")]
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        template_subjects = {day: {i: tk.StringVar() for i in range(len(periods))} for day in days}

        for i, day in enumerate(days, 1):
            ttk.Label(t_frame, text=day).grid(row=0, column=i)
        for r, (start, end) in enumerate(periods, 1):
            ttk.Label(t_frame, text=f"{start}-{end}").grid(row=r, column=0)
            for c, day in enumerate(days, 1):
                ttk.Combobox(t_frame, textvariable=template_subjects[day][r-1], values=[""] + subjects, width=10).grid(row=r, column=c)

        def apply_template():
            for day in days:
                self.timetable[day] = [{"class": template_subjects[day][i].get().strip().upper(), "start": start, "end": end}
                                       for i, (start, end) in enumerate(periods) if template_subjects[day][i].get()]
                if not self.timetable[day]:
                    del self.timetable[day]
            save_timetable(self.timetable)
            subjects[:] = sorted(set(slot['class'] for day in self.timetable.values() for slot in day))
            update_subject_list()
            update_subject_dropdown()
            update_tree()
            self.log("Applied template")
            messagebox.showinfo("Success", "Template applied")

        ttk.Button(t_frame, text="Apply", command=apply_template).grid(row=len(periods)+1, column=0, columnspan=len(days)+1)

        m_frame = ttk.Frame(manual_frame, padding=5)
        m_frame.pack(fill='both', expand=True)
        day_var = tk.StringVar(value="Mon")
        start_var = tk.StringVar(value="22:00")
        end_var = tk.StringVar(value="23:00")
        subject_var = tk.StringVar()
        time_options = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]

        for i, (label, var, values) in enumerate([
            ("Day:", day_var, days), ("Start:", start_var, time_options), ("End:", end_var, time_options), ("Subject:", subject_var, subjects)
        ]):
            ttk.Label(m_frame, text=label).grid(row=i, column=0)
            widget = ttk.Combobox(m_frame, textvariable=var, values=values)
            widget.grid(row=i, column=1)
            if label == "Subject:":
                subject_combo = widget

        def validate_time(time_str):
            try:
                datetime.strptime(time_str, '%H:%M')
                return True
            except ValueError:
                return False

        def check_overlap(day, start, end, exclude_idx=None):
            if day not in self.timetable:
                return False
            start_dt = datetime.strptime(start, '%H:%M').time()
            end_dt = datetime.strptime(end, '%H:%M').time()
            for i, slot in enumerate(self.timetable[day]):
                if exclude_idx == i:
                    continue
                slot_start = datetime.strptime(slot["start"], '%H:%M').time()
                slot_end = datetime.strptime(slot["end"], '%H:%M').time()
                if not (end_dt <= slot_start or start_dt >= slot_end):
                    return True
            return False

        def add_entry():
            day, start, end, subject = day_var.get(), start_var.get(), end_var.get(), subject_var.get().strip().upper()
            if not all([day, start, end, subject]) or not validate_time(start) or not validate_time(end):
                messagebox.showerror("Error", "Invalid input")
                return
            start_dt, end_dt = datetime.strptime(start, '%H:%M'), datetime.strptime(end, '%H:%M')
            if end_dt <= start_dt or check_overlap(day, start, end):
                messagebox.showerror("Error", "Invalid or overlapping time")
                return
            if day not in self.timetable:
                self.timetable[day] = []
            self.timetable[day].append({"class": subject, "start": start, "end": end})
            save_timetable(self.timetable)
            if subject not in subjects:
                subjects.append(subject)
                subjects.sort()
                update_subject_dropdown()
                update_subject_list()
            update_tree()
            self.log(f"Added {subject}")
            messagebox.showinfo("Success", f"Added {subject}")

        def edit_entry():
            sel = tree.selection()
            if not sel:
                messagebox.showerror("Error", "Select entry")
                return
            day, start, end, subject = day_var.get(), start_var.get(), end_var.get(), subject_var.get().strip().upper()
            if not all([day, start, end, subject]) or not validate_time(start) or not validate_time(end):
                messagebox.showerror("Error", "Invalid input")
                return
            start_dt, end_dt = datetime.strptime(start, '%H:%M'), datetime.strptime(end, '%H:%M')
            item = tree.item(sel[0])
            old_subject, old_start = item['values'][1], item['values'][2]
            old_day = next((d for d, slots in self.timetable.items() if any(s['class'] == old_subject and s['start'] == old_start for s in slots)), None)
            if not old_day:
                messagebox.showerror("Error", "Entry not found")
                return
            idx = next(i for i, s in enumerate(self.timetable[old_day]) if s['class'] == old_subject and s['start'] == old_start)
            if end_dt <= start_dt or check_overlap(day, start, end, idx if day == old_day else None):
                messagebox.showerror("Error", "Invalid or overlapping time")
                return
            self.timetable[old_day].pop(idx)
            if not self.timetable[old_day]:
                del self.timetable[old_day]
            if day not in self.timetable:
                self.timetable[day] = []
            self.timetable[day].append({"class": subject, "start": start, "end": end})
            save_timetable(self.timetable)
            if subject not in subjects:
                subjects.append(subject)
                subjects.sort()
                update_subject_dropdown()
                update_subject_list()
            update_tree()
            self.log(f"Edited {subject}")
            messagebox.showinfo("Success", f"Updated {subject}")

        def delete_entry():
            sel = tree.selection()
            if not sel:
                messagebox.showerror("Error", "Select entry")
                return
            item = tree.item(sel[0])
            subject, start = item['values'][1], item['values'][2]
            day = next((d for d, slots in self.timetable.items() if any(s['class'] == subject and s['start'] == start for s in slots)), None)
            if not day:
                messagebox.showerror("Error", "Entry not found")
                return
            self.timetable[day] = [s for s in self.timetable[day] if not (s['class'] == subject and s['start'] == start)]
            if not self.timetable[day]:
                del self.timetable[day]
            save_timetable(self.timetable)
            update_tree()
            self.log(f"Deleted {subject}")

        btn_frame = ttk.Frame(m_frame)
        btn_frame.grid(row=4, column=0, columnspan=2)
        for text, cmd in [("Add", add_entry), ("Edit", edit_entry), ("Delete", delete_entry)]:
            ttk.Button(btn_frame, text=text, command=cmd).pack(side='left', padx=5)

        tree = ttk.Treeview(m_frame, columns=('Day', 'Subject', 'Start', 'End'), show='headings')
        for col in ('Day', 'Subject', 'Start', 'End'):
            tree.heading(col, text=col)
        tree.grid(row=5, column=0, columnspan=2, sticky='nsew')
        m_frame.rowconfigure(5, weight=1)

        def update_tree():
            tree.delete(*tree.get_children())
            current_day = datetime.now().strftime('%A')[:3]
            for day, slots in sorted(self.timetable.items()):
                for slot in slots:
                    tree.insert('', 'end', values=(day, slot['class'], slot['start'], slot['end']), tags=('current' if day == current_day else '',))
            tree.tag_configure('current', background='lightyellow')

        def on_select(event):
            sel = tree.selection()
            if sel:
                item = tree.item(sel[0])
                day_var.set(item['values'][0])
                start_var.set(item['values'][2])
                end_var.set(item['values'][3])
                subject_var.set(item['values'][1])

        tree.bind('<<TreeviewSelect>>', on_select)
        update_tree()
        update_subject_dropdown()  # Ensure dropdown is synced after UI setup

    def start_attendance(self):
        if not self.timetable or not self.encode_list:
            messagebox.showerror("Error", "Add timetable or faces")
            return
        self.cap = cv2.VideoCapture(CONFIG['camera_index'])
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Webcam error")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.is_running = True
        self.status_var.set("Status: Running")
        self.log("Started")
        threading.Thread(target=self.attendance_loop, daemon=True).start()

    def attendance_loop(self):
        blinks, ears = defaultdict(int), defaultdict(list)
        frame_count = 0
        while self.is_running and self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                self.log("Webcam failed", 'error')
                break
            frame_count += 1
            img_s = cv2.resize(img, (0, 0), None, 0.2, 0.2)
            img_s_rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
            current_class = get_current_class(self.timetable)
            cv2.putText(img, f"Class: {current_class or 'None'}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0) if current_class else (0, 0, 255), 2)

            if frame_count % 5 == 0:
                faces = face_recognition.face_locations(img_s_rgb)
                encodes = face_recognition.face_encodings(img_s_rgb, faces)
                for encode, loc in zip(encodes, faces):
                    matches = face_recognition.compare_faces(self.encode_list, encode)
                    distances = face_recognition.face_distance(self.encode_list, encode)
                    if matches and (match_idx := np.argmin(distances)) and distances[match_idx] < CONFIG['face_distance_threshold']:
                        roll = studentRollNumbers[match_idx]
                        y1, x2, y2, x1 = [x * 5 for x in loc]
                        if frame_count % 10 == 0:
                            ear = check_liveness(img, (y1, x2, y2, x1))
                            ears[roll].append(ear)
                            if len(ears[roll]) > CONFIG['frame_history']:
                                ears[roll].pop(0)
                                if max(ears[roll]) > CONFIG['ear_threshold_max'] and min(ears[roll]) < CONFIG['ear_threshold_min']:
                                    blinks[roll] += 1
                        if blinks[roll] >= CONFIG['blink_threshold']:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, roll, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                            if status := mark_attendance(roll, current_class):
                                self.status_var.set(f"Status: {status}")
                                self.log(status)
                                blinks[roll], ears[roll] = 0, []
                        else:
                            cv2.putText(img, f"Blink {blinks[roll]}/{CONFIG['blink_threshold']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if frame_count % 5 == 0:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb).resize((300, 225))
                photo = ImageTk.PhotoImage(img_pil)
                self.video_label.configure(image=photo)
                self.video_label.image = photo

    def stop_attendance(self):
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Status: Stopped")
        self.log("Stopped")

    def show_analytics(self):
        win = tk.Toplevel(self.root)
        win.title("Analytics")
        win.geometry("800x400")
        notebook = ttk.Notebook(win)
        notebook.pack(fill='both', expand=True)

        def plot_tab(tab, plot_func):
            frame = ttk.Frame(tab)
            frame.pack(fill='both', expand=True)
            fig = plt.Figure(figsize=(8, 3))
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.get_tk_widget().pack(fill='both', expand=True)
            plot_func(*collect_attendance_data(), fig)
            canvas.draw()

        for name, func in [("Summary", lambda s, r, d, rec, f: plt.bar(s.keys(), [sum(d.values()) for d in s.values()], figure=f)),
                           ("Roll", lambda s, r, d, rec, f: pd.DataFrame(rec)['Roll Number'].value_counts().plot(kind='bar', figure=f)),
                           ("Subject", lambda s, r, d, rec, f: pd.DataFrame(rec)['Subject'].value_counts().plot(kind='bar', figure=f))]:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=name)
            plot_tab(tab, func)

    def generate_report(self):
        _, _, _, records = collect_attendance_data()
        if records:
            df = pd.DataFrame(records)
            report_file = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.groupby(['Roll Number', 'Subject']).size().unstack(fill_value=0).to_csv(report_file)
            self.log(f"Generated {report_file}")
            messagebox.showinfo("Success", f"Saved {report_file}")

    def export_attendance(self):
        file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if file:
            with open(file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Roll Number', 'Subject', 'Day', 'Time'])
                for fname in [f for f in os.listdir('.') if f.startswith('Attendance_') and f.endswith('.csv')]:
                    subject, date = fname.split('_')[1], fname.split('_')[2].replace('.csv', '')
                    with open(fname, 'r') as rf:
                        next(csv.reader(rf))
                        for row in csv.reader(rf):
                            if row:
                                writer.writerow([row[0], subject, date, row[1]])
                self.log(f"Exported {file}")
                messagebox.showinfo("Success", f"Exported {file}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
