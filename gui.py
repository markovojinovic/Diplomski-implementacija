import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ml import *


# Global variables
window_close = False


def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        generate_graph(df)
        KNNprediction(df)


def generate_graph(df):
    x_column = df[df.columns[0]]
    y_column = df[df.columns[1]]

    plt.figure(figsize=(8, 6))
    plt.plot(x_column, y_column, 'x')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title('Reprezentacija zavisnosti promenljivih')
    plt.grid(True)

    # Create a FigureCanvasTkAgg instance
    canvas = FigureCanvasTkAgg(plt.gcf(), master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    # Non-visible state for upload button
    upload_button.pack_forget()
    # upload_button.pack(pady=20)----------------------vraca dugme na displej---------------------------


def start_program():
    global window_close
    window_close = True
    intro_window.destroy()


def on_closing():
    intro_window.destroy()
    window.quit()


# ==================================== Create the intro window ===================================

intro_window = tk.Tk()
intro_window.title("Diplomski rad")
intro_window.iconbitmap("logo.ico")
intro_window.minsize(400, 300)

# Create widgets for the intro screen
intro_label = tk.Label(intro_window, text="Autor: Marko Vojinović 2019/0559")
intro_label.pack(pady=10)
intro_label = tk.Label(intro_window, text="Univerzitet u Beogradu, Elektrotehnićki Fakultet")
intro_label.pack(pady=10)

start_program_button = tk.Button(intro_window, text="Otpočnite sa radom", command=start_program,
                                 font=('Arial', 12), bg='#4CAF50', fg='white',
                                 activebackground='#45a049', activeforeground='white',
                                 padx=10, pady=5)
start_program_button.pack(pady=20)

# Run the intro screen
intro_window.mainloop()

# ==================================== Create the main window ===================================

if window_close:
    window = tk.Tk()
    window.title("Diplomski rad")
    window.iconbitmap("logo.ico")
    # Set minimum width and height
    window.minsize(400, 300)

    # Create the upload button with customized style
    upload_button = tk.Button(window, text="Unesite set podataka u CSV formatu", command=upload_csv,
                              font=('Arial', 12), bg='#4CAF50', fg='white',
                              activebackground='#45a049', activeforeground='white',
                              padx=10, pady=5)
    upload_button.pack(pady=20)

    # Start the GUI event loop
    window.mainloop()
