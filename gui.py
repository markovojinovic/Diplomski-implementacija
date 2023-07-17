import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from ml import *

# Set the backend explicitly to TkAgg
matplotlib.use('TkAgg')

# Global variables
window_close = False
selected_algorithm = 1
parameter_knn = 3
number_of_epochs_neural = 100
number_of_hidden_layers_neural = 2
hidden_layer_function_neural = "relu"
output_layer_function_neural = "sigmoid"
optimizer_neural = "adam"
loss_function_neural = "binary_crossentropy"
max_dept_decision_tree = -1


def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        generate_graph(df)
        global selected_algorithm
        if selected_algorithm == 1:
            global parameter_knn
            KNNprediction(df, parameter_knn)
        elif selected_algorithm == 3:
            global number_of_epochs_neural
            global number_of_hidden_layers_neural
            global hidden_layer_function_neural
            global output_layer_function_neural
            global optimizer_neural
            global loss_function_neural
            NeuralNetwork(df, number_of_hidden_layers_neural, hidden_layer_function_neural,
                          output_layer_function_neural, optimizer_neural, loss_function_neural, number_of_epochs_neural)
        elif selected_algorithm == 2:
            global max_dept_decision_tree
            DecisionTree(df, max_dept_decision_tree)


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
    # Non-visible state for upload button and select menu
    treci_red.pack_forget()
    if selected_algorithm == 1 or selected_algorithm == 2:
        drugi_red.pack_forget()
        treci_red.pack_forget()
    if selected_algorithm == 3:
        drugi_red_1.pack_forget()
        drugi_red_2.pack_forget()
        drugi_red_3.pack_forget()
        drugi_red_4.pack_forget()
        drugi_red_5.pack_forget()
        drugi_red_6.pack_forget()


def start_program():
    global window_close
    window_close = True
    intro_window.destroy()


def on_closing():
    window.quit()


def on_menu_select(event):
    global selected_algorithm
    selected_item = dropdown_algorithm.get()
    if selected_item == "KNN":
        selected_algorithm = 1
    elif selected_item == "Stablo odlučivanja":
        selected_algorithm = 2
    else:
        selected_algorithm = 3


def validate_integer_for_knn(text):
    global parameter_knn
    global selected_algorithm
    if text.isdigit() or text == "":
        if text != "":
            parameter_knn = int(text)
            if parameter_knn > 10 or parameter_knn < 3:
                messagebox.showwarning("Opseg izvan preporučenog",
                                       "Preporučen opseg za KNN algoritam je izmedju 3 i 10!")
        return True
    else:
        messagebox.showerror("Pogrešan unos", "Unesite cele brojeve, veće od 0")
        return False


def validate_integer_for_decision_tree(text):
    global max_dept_decision_tree
    global selected_algorithm
    if text.isdigit() or text == "":
        if text != "":
            max_dept_decision_tree = int(text)
            if max_dept_decision_tree < 3:
                messagebox.showwarning("Izvan preporučenog opsega",
                                       "Preporučeno je da najveća dubina stabla odlučivanja ne bude manja od 3!")
        return True
    else:
        messagebox.showerror("Pogrešan unos", "Unesite cele brojeve, veće od 0")
        return False


def validate_integer_for_neural(text, chosen):
    global number_of_epochs_neural
    global number_of_hidden_layers_neural
    if text.isdigit() or text == "":
        if text != "":
            if chosen == '1':
                number_of_hidden_layers_neural = int(text)
                if number_of_hidden_layers_neural > 3:
                    messagebox.showwarning("Izvan preporučenog opsega",
                                           "Ukoliko na neuralnu mrežu primenite više sakrivenih slojeva, "
                                           "moguće je da proces treniranja traje dugo")
            elif chosen == '2':
                number_of_epochs_neural = int(text)
                if number_of_epochs_neural > 150:
                    messagebox.showwarning("Izvan preporučenog opsega",
                                           "Ukoliko na neuralnu mrežu primenite više epoha, "
                                           "moguće je da proces treniranja traje dugo")
        return True
    else:
        messagebox.showerror("Pogrešan unos", "Unesite cele brojeve, veće od 0")
        return False


def on_function_select_neural(event, chosen):
    if chosen == '1':
        global hidden_layer_function_neural
        selected_item = dropdown_hidden_layer_neural.get()
        hidden_layer_function_neural = selected_item
    elif chosen == '2':
        global output_layer_function_neural
        selected_item = dropdown_output_layer_neural.get()
        output_layer_function_neural = selected_item
    elif chosen == '3':
        global optimizer_neural
        selected_item = dropdown_optimizer_neural.get()
        optimizer_neural = selected_item
    else:
        global loss_function_neural
        selected_item = dropdown_loss_neural.get()
        loss_function_neural = selected_item


# ==================================== Create the intro window ===================================

intro_window = tk.Tk()
intro_window.title("Diplomski rad")
intro_window.iconbitmap("logo.ico")
intro_window.minsize(400, 300)

# Create widgets for the intro screen
intro_label = tk.Label(intro_window, text="Autor: Marko Vojinović 2019/0559")
intro_label.pack(pady=10)
intro_label = tk.Label(intro_window, text="Univerzitet u Beogradu, Elektrotehnićki Fakultet")
intro_label.pack()

# Create a frame as the container
prvi_red = tk.Frame(intro_window)
prvi_red.pack(pady=10)
label = tk.Label(prvi_red, text="Odaberite algoritam za mašinsko učenje: ")
label.pack(side="left")
# Create a style for the dropdown
style = ttk.Style()
style.configure("TCombobox", foreground="red", background="lightgray", padding=6)
# Create a dropdown menu
dropdown_algorithm = ttk.Combobox(prvi_red, values=["KNN", "Stablo odlučivanja", "Neuralna mreža"], state="readonly",
                                  style="TCombobox")
dropdown_algorithm.pack(side="left")
# Set a default value for the dropdown
dropdown_algorithm.set("KNN")
# Bind the event when a menu item is selected
dropdown_algorithm.bind("<<ComboboxSelected>>", on_menu_select)

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
    window.minsize(600, 450)

    if selected_algorithm == 1:  # KNN
        # Create a frame as the container
        drugi_red = tk.Frame(window)
        drugi_red.pack(pady=10)
        # Create a validation command to allow only integers
        validate_command = (window.register(validate_integer_for_knn), '%P')
        # Create an input box for integers
        label = tk.Label(drugi_red, text="Unesite argument KNN algoritma:")
        label.pack(side="left")
        entry = tk.Entry(drugi_red, validate="key", validatecommand=validate_command)
        entry.insert(0, str(parameter_knn))
        entry.pack(side="left")

    if selected_algorithm == 2:  # Decision tree
        # Create a frame as the container
        drugi_red = tk.Frame(window)
        drugi_red.pack(pady=10)
        # Create a validation command to allow only integers
        validate_command = (window.register(validate_integer_for_decision_tree), '%P')
        # Create an input box for integers
        label = tk.Label(drugi_red, text="Unesite najveću dubinu stabla odlučivanja:")
        label.pack(side="left")
        entry = tk.Entry(drugi_red, validate="key", validatecommand=validate_command)
        entry.pack(side="left")

    if selected_algorithm == 3:  # Neural network

        # Create a style for the dropdown
        style = ttk.Style()
        style.configure("TCombobox", foreground="red", background="lightgray", padding=6)

        # Create a frame as the container
        drugi_red_1 = tk.Frame(window)
        drugi_red_1.pack(pady=10)
        # Create a validation command to allow only integers
        validate_command = (window.register(validate_integer_for_neural), '%P', '1')
        # Create an input box for integers
        label_hidden_layer_neural = tk.Label(drugi_red_1, text="Unesite broj sakrivenih slojeva:")
        label_hidden_layer_neural.pack(side="left")
        entry_hidden_layers_neural = tk.Entry(drugi_red_1, validate="key", validatecommand=validate_command)
        entry_hidden_layers_neural.insert(0, str(number_of_hidden_layers_neural))
        entry_hidden_layers_neural.pack(side="left")

        # Create a frame as the container
        drugi_red_2 = tk.Frame(window)
        drugi_red_2.pack(pady=10)
        label_hidden_layer_neural = tk.Label(drugi_red_2, text="Odaberite funkciju za obradu sakrivenog sloja:")
        label_hidden_layer_neural.pack(side="left")
        # Create a dropdown menu
        dropdown_hidden_layer_neural = ttk.Combobox(drugi_red_2, values=["relu", "sigmoid", "softmax", "tanh"],
                                                    state="readonly",
                                                    style="TCombobox")
        dropdown_hidden_layer_neural.pack(side="left")
        # Set a default value for the dropdown
        dropdown_hidden_layer_neural.set("relu")
        dropdown_hidden_layer_neural.bind("<<ComboboxSelected>>", lambda event: on_function_select_neural(event, '1'))

        # Create a frame as the container
        drugi_red_3 = tk.Frame(window)
        drugi_red_3.pack(pady=10)
        # Create an input box for integers
        label_output_layer_neural = tk.Label(drugi_red_3, text="Odaberite funkciju obrade izlaznog sloja:")
        label_output_layer_neural.pack(side="left")
        # Create a dropdown menu
        dropdown_output_layer_neural = ttk.Combobox(drugi_red_3, values=["relu", "sigmoid", "softmax", "tanh"],
                                                    state="readonly",
                                                    style="TCombobox")
        dropdown_output_layer_neural.pack(side="left")
        # Set a default value for the dropdown
        dropdown_output_layer_neural.set("sigmoid")
        dropdown_output_layer_neural.bind("<<ComboboxSelected>>", lambda event: on_function_select_neural(event, '2'))

        # Create a frame as the container
        drugi_red_4 = tk.Frame(window)
        drugi_red_4.pack(pady=10)
        # Create an input box for integers
        label_optimizer_neural = tk.Label(drugi_red_4, text="Odaberite optimizator:")
        label_optimizer_neural.pack(side="left")
        # Create a dropdown menu
        dropdown_optimizer_neural = ttk.Combobox(drugi_red_4, values=["adam", "SGD", "RMSprop"], state="readonly",
                                                 style="TCombobox")
        dropdown_optimizer_neural.pack(side="left")
        # Set a default value for the dropdown
        dropdown_optimizer_neural.set("adam")
        dropdown_optimizer_neural.bind("<<ComboboxSelected>>", lambda event: on_function_select_neural(event, '3'))

        # Create a frame as the container
        drugi_red_5 = tk.Frame(window)
        drugi_red_5.pack(pady=10)
        # Create an input box for integers
        label_loss_neural = tk.Label(drugi_red_5, text="Odaberite funkciju obrade gubitka:")
        label_loss_neural.pack(side="left")
        dropdown_loss_neural = ttk.Combobox(drugi_red_5,
                                            values=["binary_crossentropy", "mse", "categorical_crossentropy"],
                                            state="readonly",
                                            style="TCombobox")
        dropdown_loss_neural.pack(side="left")
        # Set a default value for the dropdown
        dropdown_loss_neural.set("binary_crossentropy")
        dropdown_loss_neural.bind("<<ComboboxSelected>>", lambda event: on_function_select_neural(event, '4'))

        # Create a frame as the container
        drugi_red_6 = tk.Frame(window)
        drugi_red_6.pack(pady=10)
        # Create a validation command to allow only integers
        validate_command = (window.register(validate_integer_for_neural), '%P', '2')
        # Create an input box for integers
        label_epochs_neural = tk.Label(drugi_red_6, text="Unesite broj epoha:")
        label_epochs_neural.pack(side="left")
        entry_epochs_neural = tk.Entry(drugi_red_6, validate="key", validatecommand=validate_command)
        entry_epochs_neural.insert(0, str(number_of_epochs_neural))
        entry_epochs_neural.pack(side="left")

    # Create a frame as the container
    treci_red = tk.Frame(window)
    treci_red.pack(pady=10)
    # Create the upload button with customized style
    upload_button = tk.Button(treci_red, text="Unesite set podataka u CSV formatu", command=upload_csv,
                              font=('Arial', 12), bg='#4CAF50', fg='white',
                              activebackground='#45a049', activeforeground='white',
                              padx=10, pady=5)
    upload_button.pack(side="left")

    window.protocol("WM_DELETE_WINDOW", on_closing)
    # Start the GUI event loop
    window.mainloop()

    # Explicitly quit the Tkinter event loop
    window.quit()
