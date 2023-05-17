import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        generate_graph(df)


def generate_graph(df):
    x_column = df['Pregnancies']
    y_column = df['Glucose']

    plt.figure(figsize=(8, 6))
    plt.plot(x_column, y_column, 'x')
    plt.xlabel('Pregnancies')
    plt.ylabel('Glucose')
    plt.title('Reprezentacija zavisnosti trudnoce i glukoze')
    plt.grid(True)

    # Create a FigureCanvasTkAgg instance
    canvas = FigureCanvasTkAgg(plt.gcf(), master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


# Create the main window
window = tk.Tk()
window.title("Diplomski rad")
# Set minimum width and height
window.minsize(400, 300)

# Create the upload button
upload_button = tk.Button(window, text="Prilozite CSV fajl", command=upload_csv)
upload_button.pack(pady=20)

# Start the GUI event loop
window.mainloop()