import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ml import*


def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        generate_graph(df)
        KNNprediction(df)


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
    # Non-visible state for upload button
    upload_button.pack_forget()
    # upload_button.pack(pady=20)----------------------vraca dugme na displej---------------------------


# ==================================== Create the main window ===================================

window = tk.Tk()
window.title("Diplomski rad")
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
