import os

# Import for email functionality
import smtplib
import subprocess
import threading
import tkinter as tk
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ultralytics import YOLO

# --- Configuration Constants ---
EXCEL_FILE = "crowd_log.xlsx"
MODEL = YOLO("yolov8n.pt") # Ensure this model path is correct or accessible

# --- Global Variables ---
stop_detection = False
video_path = None
# Global references for matplotlib canvas and toolbar in dashboard
canvas_dashboard = None
fig_dashboard = None
# Global references for matplotlib canvas in heatmap
canvas_heatmap = None
fig_heatmap = None

# Cooldown period for email alerts (e.g., 2 minutes = 120 seconds)
ALERT_COOLDOWN_SECONDS = 120 # Changed from 300 to 120 for 2 minutes
last_alert_time = datetime.min # Initialize to a very old time to allow immediate first alert

# --- SMTP Configuration ---
SENDER_EMAIL = "rishi.sridhar25@gmail.com"
# IMPORTANT: Use an App Password if you're using Gmail with 2-Factor Authentication
# Follow Google's instructions to generate one: https://support.google.com/accounts/answer/185833
SENDER_PASSWORD = "wffu oheu pspg whae" # <<< REPLACE THIS WITH YOUR GENERATED APP PASSWORD
RECEIVER_EMAIL = "rishi.sridhar25@gmail.com" # Can be different or the same as sender
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587 # For TLS

# --- Helper Functions ---

def send_alert_email(current_count, limit):
    """Sends an email notification when the crowd limit is exceeded."""
    msg = MIMEText(f"Crowd Limit Exceeded!\n\nCurrent Count: {current_count}\nLimit: {limit}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    msg['Subject'] = "CROWD ALERT: Limit Exceeded!"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    try:
        # Connect to the SMTP server
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Start TLS encryption
            server.login(SENDER_EMAIL, SENDER_PASSWORD) # Login with your app password
            server.send_message(msg)
        print(f"Email alert sent successfully! Count: {current_count}, Limit: {limit}")
    except Exception as e:
        print(f"Failed to send email alert: {e}")
        messagebox.showerror("Email Error", f"Failed to send alert email: {e}\n"
                              "Please check SENDER_EMAIL, SENDER_PASSWORD (App Password), "
                              "2-Step Verification, and network connection.")


def log_to_excel(count, limit, alert_triggered):
    """Logs crowd detection data to an Excel file."""
    now = datetime.now()
    new_data = pd.DataFrame([{
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "People Count": count,
        "Limit": limit,
        "Alert Triggered": "Yes" if alert_triggered else "No"
    }])
    if not os.path.exists(EXCEL_FILE):
        new_data.to_excel(EXCEL_FILE, index=False)
    else:
        try:
            df = pd.read_excel(EXCEL_FILE)
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_excel(EXCEL_FILE, index=False)
        except Exception as e:
            messagebox.showwarning("Excel Write Error", f"Could not write to Excel: {e}. Attempting to overwrite.")
            try:
                # Fallback: if concat fails (e.g., file corruption, odd format), try to overwrite
                # This might lose previous data if the file was indeed corrupted or malformed
                new_data.to_excel(EXCEL_FILE, index=False)
            except Exception as e_overwrite:
                messagebox.showerror("Critical Excel Error", f"Failed to write to Excel even with overwrite: {e_overwrite}")


def open_excel_file():
    """Opens the Excel log file using the default system application."""
    try:
        if os.name == 'nt':  # For Windows
            os.startfile(EXCEL_FILE)
        elif os.name == 'posix':  # For macOS or Linux
            subprocess.call(['xdg-open', EXCEL_FILE]) # Linux
        else:
            subprocess.call(['open', EXCEL_FILE]) # macOS
    except FileNotFoundError:
        messagebox.showerror("File Error", f"The file '{EXCEL_FILE}' was not found.")
    except Exception as e:
        messagebox.showerror("Excel Error", f"Could not open Excel file: {e}")

# --- Detection Functions ---

def start_detection():
    """Starts the real-time crowd detection process."""
    global stop_detection, last_alert_time
    stop_detection = False
    
    limit_str = limit_entry.get()
    if not limit_str.isdigit():
        messagebox.showerror("Input Error", "Enter a valid numeric limit for the crowd.")
        return
    limit = int(limit_str)

    source = 0 if video_path is None else video_path # 0 for webcam, otherwise video file path
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        messagebox.showerror("Video Error", f"Could not open video source '{source}'.\n"
                              "Please check camera connection/drivers or verify video file path and format.")
        return

    # Reset last_alert_time to allow an immediate alert if conditions are met
    last_alert_time = datetime.now() - timedelta(seconds=ALERT_COOLDOWN_SECONDS + 1)

    while not stop_detection:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from video source. Stopping detection.")
            break # Exit loop if no frame is read (e.g., end of video, camera disconnected)

        results = MODEL(frame) # Perform object detection
        # YOLOv8 returns results where class 0 typically corresponds to 'person'
        people_count = sum(1 for result in results[0].boxes.cls if int(result) == 0)
        
        annotated_frame = results[0].plot() # Get the frame with bounding boxes and labels

        # Display count on frame
        cv2.putText(annotated_frame, f'Count: {people_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        alert_triggered = False
        if people_count > limit:
            cv2.putText(annotated_frame, 'ALERT: Crowd Exceeded!', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            alert_status_label.config(text="⚠ ALERT", fg="red")
            alert_triggered = True

            # Send email alert if cooldown period has passed
            current_time = datetime.now()
            if (current_time - last_alert_time).total_seconds() > ALERT_COOLDOWN_SECONDS:
                threading.Thread(target=send_alert_email, args=(people_count, limit)).start()
                last_alert_time = current_time # Update last alert time
        else:
            alert_status_label.config(text="✅ Safe", fg="green")

        # Update GUI labels
        people_count_label.config(text=str(people_count))
        
        # Log data to Excel
        log_to_excel(people_count, limit, alert_triggered)

        # Display the annotated frame (resizing for better fit)
        display_width = 800
        display_height = int(annotated_frame.shape[0] * (display_width / annotated_frame.shape[1]))
        resized_frame = cv2.resize(annotated_frame, (display_width, display_height))
        cv2.imshow("Crowd Detection", resized_frame)

        # Check for 'q' key press or stop_detection flag
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_detection:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    alert_status_label.config(text="🛑 Stopped", fg="black")
    
    # Refresh dashboard after detection stops to include all new logs
    root.after(100, refresh_dashboard) # Use root.after to safely update GUI from another thread


def stop_detection_func():
    """Sets the global flag to stop the detection loop."""
    global stop_detection
    stop_detection = True
    alert_status_label.config(text="🛑 Stopped", fg="black")

def browse_video():
    """Opens a file dialog to select a video file for detection."""
    global video_path
    selected_path = filedialog.askopenfilename(filetypes=[("Video files", ".mp4 *.avi *.mov *.mkv"), ("All files", ".*")])
    if selected_path:
        video_path = selected_path
        video_label.config(text=os.path.basename(video_path))
    else:
        video_path = None # Reset to webcam if selection is cancelled
        video_label.config(text="Webcam (Default)")

# --- Dashboard Functions ---

def refresh_dashboard():
    """Refreshes the data displayed in the dashboard (Treeview and graph)."""
    # Clear existing treeview data
    for i in dashboard_tree.get_children():
        dashboard_tree.delete(i)

    if not os.path.exists(EXCEL_FILE):
        # If excel file doesn't exist, show message in graph area too
        plot_crowd_count_line_graph(pd.DataFrame()) # Pass empty DataFrame
        return

    try:
        df = pd.read_excel(EXCEL_FILE)
        
        # Ensure consistent column order and handling missing columns gracefully
        expected_columns = ["Date", "Time", "People Count", "Limit", "Alert Triggered"]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = "N/A" # Fill missing columns with N/A or appropriate default

        for _, row in df.iterrows():
            dashboard_tree.insert('', 'end', values=(
                row["Date"],
                row["Time"],
                row["People Count"],
                row["Limit"],
                row["Alert Triggered"]
            ))

        # Update the line graph with the loaded data
        plot_crowd_count_line_graph(df)

    except Exception as e:
        messagebox.showerror("Dashboard Data Error", f"Could not read or process Excel data for dashboard: {e}")
        plot_crowd_count_line_graph(pd.DataFrame()) # Clear graph on error


def plot_crowd_count_line_graph(df):
    """Generates and embeds a line graph of People Count over Time."""
    global canvas_dashboard, fig_dashboard

    # Clear previous plot if exists
    for widget in graph_frame.winfo_children():
        widget.destroy()
    if fig_dashboard:
        plt.close(fig_dashboard) # Close the old figure to free memory

    if df.empty:
        tk.Label(graph_frame, text="No data to display in graph.", font=("Arial", 10)).pack(pady=20)
        return

    # Ensure 'Date', 'Time', 'People Count' columns exist
    if not all(col in df.columns for col in ["Date", "Time", "People Count"]):
        messagebox.showerror("Graph Data Error", "Missing required columns for plotting (Date, Time, People Count).")
        tk.Label(graph_frame, text="Missing data columns for graph.", font=("Arial", 10, "italic")).pack(pady=20)
        return

    # Combine Date and Time into DateTime for plotting
    df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors='coerce')
    
    # Convert 'People Count' to numeric, handling errors
    df["People Count"] = pd.to_numeric(df["People Count"], errors='coerce')
    
    # Drop rows where DateTime or People Count could not be converted
    df.dropna(subset=['DateTime', 'People Count'], inplace=True)

    if df.empty: # Check again after cleaning
        tk.Label(graph_frame, text="No valid numeric people count data or datetime to display.", font=("Arial", 10)).pack(pady=20)
        return

    # Sort by DateTime to ensure correct line plot progression
    df = df.sort_values(by="DateTime")

    # Create the plot
    fig_dashboard, ax = plt.subplots(figsize=(8, 3)) # Adjusted size for dashboard
    sns.lineplot(x="DateTime", y="People Count", data=df, ax=ax, marker='o', color='blue', linewidth=1.5,
                 label="People Count")
    
    # Optionally add Limit line
    if "Limit" in df.columns and pd.to_numeric(df["Limit"], errors='coerce').max() > 0:
        limit_val = pd.to_numeric(df["Limit"], errors='coerce').dropna().iloc[0] if not df["Limit"].isnull().all() else None
        if limit_val is not None:
            ax.axhline(y=limit_val, color='red', linestyle='--', label=f'Limit ({int(limit_val)})')

    ax.set_title("People Count Over Time", fontsize=12)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("People Count", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', fontsize=8)
    
    # Improve x-axis date formatting based on the range of data
    # If data spans multiple days, show Date & Time. If single day, just Time.
    if df['DateTime'].dt.date.nunique() > 1:
        date_format = '%Y-%m-%d %H:%M'
    else:
        date_format = '%H:%M:%S'
    
    formatter = plt.matplotlib.dates.DateFormatter(date_format)
    ax.xaxis.set_major_formatter(formatter)

    fig_dashboard.tight_layout() # Adjust layout to prevent labels overlapping

    # Embed the plot in Tkinter
    canvas_dashboard = FigureCanvasTkAgg(fig_dashboard, master=graph_frame)
    canvas_dashboard.draw()
    canvas_dashboard.get_tk_widget().pack(expand=True, fill='both')


# --- Heatmap Functions ---

def show_heatmap():
    """Generates and embeds a second-wise heatmap of crowd density."""
    global canvas_heatmap, fig_heatmap

    # Clear previous plot if exists
    for widget in heatmap_frame.winfo_children():
        widget.destroy()
    if fig_heatmap:
        plt.close(fig_heatmap) # Close the old figure to free memory

    if not os.path.exists(EXCEL_FILE):
        messagebox.showwarning("Data Missing", "Excel log file not found for heatmap.")
        return

    try:
        df = pd.read_excel(EXCEL_FILE)
    except Exception as e:
        messagebox.showerror("Excel Read Error", f"Could not read Excel file for heatmap: {e}")
        return

    if df.empty:
        messagebox.showinfo("No Data", "The Excel log is empty for heatmap.")
        return

    # Ensure necessary columns exist
    if not all(col in df.columns for col in ["Date", "Time", "People Count"]):
        messagebox.showerror("Heatmap Data Error", "Missing 'Date', 'Time', or 'People Count' columns in Excel file.")
        return

    # Combine Date and Time into DateTime
    df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors='coerce')
    df.set_index("DateTime", inplace=True)

    # Convert 'People Count' to numeric, coercing errors to NaN and then filling
    df["People Count"] = pd.to_numeric(df["People Count"], errors='coerce').fillna(0)
    df.dropna(subset=['People Count'], inplace=True) # Ensure no NaNs from datetime conversion either

    if df.empty:
        tk.Label(heatmap_frame, text="No valid numeric data for heatmap.", font=("Arial", 10)).pack(pady=20)
        return
    
    # Sort by DateTime to ensure correct chronological order for the heatmap columns
    df = df.sort_index()

    # Create a DataFrame for the heatmap. This will be a single row with columns as timestamps.
    heatmap_data = pd.DataFrame([df['People Count'].values], columns=df.index)
    
    fig_heatmap, ax = plt.subplots(figsize=(15, 3)) # Wider for more seconds

    sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax, cbar=True, yticklabels=False)

    ax.set_title("Crowd Density Heatmap (Second-wise)", fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("") # No y-label needed for a single row

    # Improve x-axis labels for readability. Show labels every N seconds.
    num_points = len(df.index)
    
    # Determine interval for ticks based on number of data points
    # Aim for roughly 10-20 ticks for readability
    if num_points > 20:
        interval = max(1, num_points // 15) # Show about 15 labels
    else:
        interval = 1 # Show all labels if data is sparse

    if num_points > 0:
        tick_positions = range(0, num_points, interval)
        tick_labels = [df.index[i].strftime('%H:%M:%S') for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xticks([]) # No ticks if no data

    fig_heatmap.tight_layout() # Adjust layout to prevent labels overlapping

    # Embed the heatmap in Tkinter
    canvas_heatmap = FigureCanvasTkAgg(fig_heatmap, master=heatmap_frame)
    canvas_heatmap.draw()
    canvas_heatmap.get_tk_widget().pack(expand=True, fill='both')


# --- GUI Layout ---
root = tk.Tk()
root.title("Crowd Detection System with Dashboard")
root.geometry("1000x650") # Set initial window size

# Create a notebook (tabs)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=5, pady=5) # Add padding to notebook

# --- Detection Tab ---
detection_tab = tk.Frame(notebook, bg='#e0f2f7') # Light blue background
notebook.add(detection_tab, text="🎥 Detection")

tk.Label(detection_tab, text="YOLOv8 Crowd Monitoring System", font=("Helvetica", 18, "bold"), bg='#e0f2f7', fg='#004d40').pack(pady=15)

top_frame = tk.Frame(detection_tab, bg='#e0f2f7')
top_frame.pack(pady=10)

tk.Label(top_frame, text="Crowd Limit:", font=("Arial", 12), bg='#e0f2f7').pack(side=tk.LEFT, padx=5)
limit_entry = tk.Entry(top_frame, font=("Arial", 12), width=7, bd=2, relief="groove")
limit_entry.pack(side=tk.LEFT, padx=10)
limit_entry.insert(0, "10") # Default limit

tk.Button(top_frame, text="📁 Browse Video", command=browse_video, font=("Arial", 10, "bold"), bg='#c8e6c9', fg='#1b5e20', relief="raised", bd=2).pack(side=tk.LEFT, padx=10)
video_label = tk.Label(top_frame, text="Webcam (Default)", font=("Arial", 10), bg='#e0f2f7', fg='#333333')
video_label.pack(side=tk.LEFT)

button_frame = tk.Frame(detection_tab, bg='#e0f2f7')
button_frame.pack(pady=20)
tk.Button(button_frame, text="▶ Start Detection", font=("Arial", 14, "bold"), bg='#4CAF50', fg='white', relief="raised", bd=3, command=lambda: threading.Thread(target=start_detection).start()).pack(side=tk.LEFT, padx=20)
tk.Button(button_frame, text="⏹ Stop Detection", font=("Arial", 14, "bold"), bg='#f44336', fg='white', relief="raised", bd=3, command=stop_detection_func).pack(side=tk.LEFT, padx=20)

info_frame = tk.Frame(detection_tab, bg='#fffde7', bd=2, relief="solid") # Light yellow background for info
info_frame.pack(pady=15, padx=20, fill='x')

tk.Label(info_frame, text="👥 Current Count:", font=("Arial", 14), bg='#fffde7').grid(row=0, column=0, padx=10, pady=5)
people_count_label = tk.Label(info_frame, text="0", font=("Arial", 14, "bold"), bg='#fffde7', fg='#0d47a1')
people_count_label.grid(row=0, column=1, padx=10, pady=5)

tk.Label(info_frame, text="⚠ Alert Status:", font=("Arial", 14), bg='#fffde7').grid(row=0, column=2, padx=(30, 10), pady=5)
alert_status_label = tk.Label(info_frame, text="Waiting...", font=("Arial", 14, "bold"), bg='#fffde7', fg='black')
alert_status_label.grid(row=0, column=3, padx=10, pady=5)
info_frame.grid_columnconfigure(1, weight=1) # Allow count label to expand
info_frame.grid_columnconfigure(3, weight=1) # Allow alert label to expand


# --- Dashboard Tab ---
dashboard_tab = tk.Frame(notebook, bg='#e8f5e9') # Light green background
notebook.add(dashboard_tab, text="📊 Dashboard")

# Frame for the Treeview (log table)
table_frame = tk.Frame(dashboard_tab, bg='#e8f5e9')
table_frame.pack(side=tk.TOP, fill="both", expand=True, padx=10, pady=(10, 5)) # Adjust pack for layout

columns = ("Date", "Time", "People Count", "Limit", "Alert Triggered")
dashboard_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
for col in columns:
    dashboard_tree.heading(col, text=col)
    dashboard_tree.column(col, anchor=tk.CENTER, width=100) # Set a default width
dashboard_tree.pack(expand=True, fill="both")

# Scrollbar for the Treeview
scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=dashboard_tree.yview)
dashboard_tree.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Buttons for dashboard actions
btns = tk.Frame(dashboard_tab, bg='#e8f5e9')
btns.pack(pady=5)
tk.Button(btns, text="🔁 Refresh Data", command=refresh_dashboard, font=("Arial", 10), bg='#bbdefb', relief="raised", bd=2).pack(side=tk.LEFT, padx=10)
tk.Button(btns, text="📂 Open in Excel", command=open_excel_file, font=("Arial", 10), bg='#bbdefb', relief="raised", bd=2).pack(side=tk.LEFT)

# Frame for the Line Graph
graph_frame = tk.Frame(dashboard_tab, bd=2, relief="groove", bg='white') # Added border for visual separation
graph_frame.pack(side=tk.BOTTOM, fill="both", expand=True, padx=10, pady=(5, 10))


# --- Heatmap Tab ---
heatmap_tab = tk.Frame(notebook, bg='#ffe0b2') # Light orange background
notebook.add(heatmap_tab, text="🔥 Heatmap")

tk.Label(heatmap_tab, text="Crowd Density Heatmap Visualization", font=("Helvetica", 16, "bold"), bg='#ffe0b2', fg='#e65100').pack(pady=15)

heatmap_frame = tk.Frame(heatmap_tab, bg='white', bd=2, relief="groove")
heatmap_frame.pack(expand=True, fill='both', padx=10, pady=10)
tk.Button(heatmap_tab, text="📊 Generate Heatmap", command=show_heatmap, font=("Arial", 12), bg='#ffcc80', relief="raised", bd=2).pack(pady=10)


# --- Initial Setup ---
refresh_dashboard() # Call on startup to populate dashboard and initial graph
root.mainloop()
