import tkinter as tk
from tkinter import ttk
import threading
import time

def simulate_progress():
    progress['maximum'] = 100
    for i in range(101):
        time.sleep(0.1)  # Simulate some work being done
        progress['value'] = i
        window.update_idletasks()

def start_progress_in_thread():
    start_button.config(state=tk.DISABLED)  # Disable the start button during progress

    # Create a new thread for the progress bar
    progress_thread = threading.Thread(target=simulate_progress)
    progress_thread.start()

    # Check the progress thread's status and re-enable the button when it's done
    window.after(100, check_progress_thread, progress_thread)

def check_progress_thread(thread):
    if thread.is_alive():
        window.after(100, check_progress_thread, thread)
    else:
        start_button.config(state=tk.NORMAL)  # Re-enable the start button after progress

# Create the main application window
window = tk.Tk()
window.title("Progress Bar Example")

# Create a progress bar widget
progress = ttk.Progressbar(window, orient=tk.HORIZONTAL, length=300, mode='determinate')
progress.pack(pady=20)

# Create a "Start" button to trigger the progress simulation
start_button = tk.Button(window, text="Start Progress", command=start_progress_in_thread)
start_button.pack(pady=10)

window.mainloop()