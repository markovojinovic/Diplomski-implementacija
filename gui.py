import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from ml import *

# TODO: problemi oko reakcije dugmeta za export na to
#  (kad je unet model ne treba da ga ima, ali ako se izmeni, treba da se pojavi)
#  kao i to da se obavesti korinik kad je krenulo izracunavanje za modele, a pre nego se zavrsi
#  videti sta sve moze da se dodaje od parametara za algoritme, na dokumentacijama, i ako fali nesto - dodati

matplotlib.use('TkAgg')

# Global variables
window_close = False
selected_algorithm = 1

number_of_neighbours_knn = 5
weight_knn = "uniform"
algorithm_knn = "auto"
leaf_size_knn = 30
p_knn = 2
metric_knn = "minkowski"

number_of_epochs_neural = 100
number_of_hidden_layers_neural = 2
hidden_layer_function_neural = "relu"
output_layer_function_neural = "sigmoid"
optimizer_neural = "adam"
loss_function_neural = "binary_crossentropy"

max_dept_decision_tree = None
criterion_decision_tree = "gini"
splitter_decision_tree = "best"
random_state_decision_tree = None

model = None
prediction_parameter1 = ""
prediction_parameter2 = ""
df = None
new_x_points = []
new_y_points = []
peti_red = None
middle_frame = None
prediction_output = ""
real_output = ""
exported = False
loaded_export = False
changed_loaded = False
answer_buttons = None
question_label = None


def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:

        upload_csv_button.destroy()

        global exported
        global loaded_export
        global changed_loaded

        if (not exported and not loaded_export) or changed_loaded:
            treci_red.pack(pady=10)
            save_model_button = tk.Button(treci_red, text="Sačuvajte model", command=save_model,
                                          font=('Arial', 12), bg='#4CAF50', fg='white',
                                          activebackground='#45a049', activeforeground='white',
                                          padx=10, pady=5)
            save_model_button.pack(side="left")

        global df
        df = pd.read_csv(file_path)
        generate_graph()

        global prediction_parameter1
        global prediction_parameter2

        global selected_algorithm
        global model
        if selected_algorithm == 1:
            global number_of_neighbours_knn
            global leaf_size_knn
            global p_knn
            global weight_knn
            global metric_knn
            global algorithm_knn
            model = train_knn(df, number_of_neighbours_knn, leaf_size_knn, p_knn, weight_knn, metric_knn, algorithm_knn)
        elif selected_algorithm == 3:
            global number_of_epochs_neural
            global number_of_hidden_layers_neural
            global hidden_layer_function_neural
            global output_layer_function_neural
            global optimizer_neural
            global loss_function_neural
            model = train_neural(df, number_of_hidden_layers_neural, hidden_layer_function_neural,
                                 output_layer_function_neural, optimizer_neural, loss_function_neural,
                                 number_of_epochs_neural)
        elif selected_algorithm == 2:
            global max_dept_decision_tree
            global criterion_decision_tree
            global splitter_decision_tree
            global random_state_decision_tree
            model = train_decision_tree(df, max_dept_decision_tree, criterion_decision_tree, splitter_decision_tree,
                                        random_state_decision_tree)

        window.state('zoomed')


def predict_model():
    global prediction_parameter1
    global prediction_parameter2
    global new_x_points
    global new_y_points
    global prediction_output
    global model

    new_x_points.append(int(prediction_parameter1))
    new_y_points.append(int(prediction_parameter2))

    prediciton_result = -1
    if selected_algorithm == 1:
        prediciton_result = predict_knn(model, int(prediction_parameter1), int(prediction_parameter2))
    elif selected_algorithm == 2:
        prediciton_result = predict_decision_tree(model, int(prediction_parameter1), int(prediction_parameter2))
    elif selected_algorithm == 3:
        prediciton_result = predict_neural(model, int(prediction_parameter1), int(prediction_parameter2))

    prediction_output = str(prediciton_result[0])
    generate_graph()

    return prediciton_result


def retrain_model():
    global real_output
    global number_of_epochs_neural
    global loaded_export
    global exported
    global changed_loaded
    global answer_buttons
    global question_label
    global peti_red
    global prediction_parameter2
    global prediction_parameter1
    global prediction_output

    question_label.destroy()

    if real_output != "":
        answer_buttons.destroy()

        if loaded_export or exported:
            changed_loaded = True
            treci_red.pack(pady=10)
            save_model_button = tk.Button(treci_red, text="Sačuvajte model", command=save_model,
                                          font=('Arial', 12), bg='#4CAF50', fg='white',
                                          activebackground='#45a049', activeforeground='white',
                                          padx=10, pady=5)
            save_model_button.pack(side="left")
            prediction_output = ""
            prediction_parameter2 = ""
            prediction_parameter1 = ""
            prediction_parameter1 = ""
            generate_graph()

        if selected_algorithm == 2:
            retrain_decision_tree(model, int(prediction_parameter1), int(prediction_parameter2), int(real_output))
        elif selected_algorithm == 3:
            retrain_neural(model, int(prediction_parameter1), int(prediction_parameter2), int(real_output),
                           number_of_epochs_neural)

    else:
        question_label = tk.Label(peti_red, text="Model ne može biti pretreniran praznim izlazom")
        question_label.grid(row=0, column=5, padx=30, pady=5)

    return


def repack_for_retrain():
    global peti_red
    global answer_buttons
    global question_label
    global real_output
    global selected_algorithm

    if selected_algorithm == 1:
        answer_buttons.destroy()
        question_label.destroy()
        question_label = tk.Label(peti_red,
                                  text="Za knn algoritam, nije moguće izvršiti pretreniranje sa jednom ulaznom vrednosti")
        question_label.grid(row=0, column=5, padx=30, pady=5)
    else:
        question_label.destroy()
        question_label = tk.Label(peti_red, text="Unesite realnu vrednost izlaza, i pretrenirajte model")
        question_label.grid(row=0, column=5, padx=30, pady=5)

        answer_buttons.destroy()
        answer_buttons = tk.Frame(peti_red)
        answer_buttons.grid(row=1, column=5, padx=30, pady=5)

        validate_command1 = (window.register(validate_real_output), '%P')
        entry1 = tk.Entry(answer_buttons, validate="key", validatecommand=validate_command1)
        entry1.insert(0, str(real_output))
        entry1.grid(row=0, column=0, padx=30, pady=5)

        retrain_button = tk.Button(answer_buttons, text="Pretrenirajte model", command=retrain_model,
                                   font=('Arial', 12), bg='#4CAF50', fg='white',
                                   activebackground='#45a049', activeforeground='white',
                                   padx=10, pady=5)
        retrain_button.grid(row=0, column=1, padx=30, pady=5)

    return


def repack_for_good_prediction():
    global peti_red
    global answer_buttons
    global question_label

    answer_buttons.destroy()
    question_label.destroy()
    question_label = tk.Label(peti_red, text="Nakon dobrog predvidjanja model ostaje isti")
    question_label.grid(row=0, column=5, padx=30, pady=5)

    return


def load_model():
    global selected_algorithm
    global model
    global df
    global loaded_export
    global exported
    global changed_loaded

    loaded_export = True
    loaded_file = None

    if selected_algorithm == 1:
        loaded_file = load_knn()
    elif selected_algorithm == 3:
        loaded_file, df = load_neural()
    elif selected_algorithm == 2:
        loaded_file = load_decision_tree()

    if selected_algorithm == 1 or selected_algorithm == 2:
        model = loaded_file['model']
        df = loaded_file['df']
    else:
        model = loaded_file

    if model is not None and df is not None:
        upload_csv_button.destroy()
        if (not exported and not loaded_export) or changed_loaded:
            treci_red.pack(pady=10)
            save_model_button = tk.Button(treci_red, text="Sačuvajte model", command=save_model,
                                          font=('Arial', 12), bg='#4CAF50', fg='white',
                                          activebackground='#45a049', activeforeground='white',
                                          padx=10, pady=5)
            save_model_button.pack(side="left")

        generate_graph()
        window.state('zoomed')


def save_model():
    global selected_algorithm
    global model
    global exported
    global df

    exported = True

    if selected_algorithm == 1:
        model = save_knn(model, df)
    elif selected_algorithm == 3:
        model = save_neural(model, df)
    elif selected_algorithm == 2:
        model = save_decision_tree(model, df)


def generate_graph():
    global df
    global new_x_points
    global new_y_points
    global peti_red
    global middle_frame

    x_column = df[df.columns[0]]
    y_column = df[df.columns[1]]

    plt.figure(figsize=(8, 6))

    plt.plot(x_column, y_column, 'x', color='blue', label='Podaci iz seta')

    plt.plot(new_x_points, new_y_points, 'o', color='red', label='Vaši podaci')

    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title('Reprezentacija zavisnosti promenljivih')
    plt.grid(True)
    plt.legend()

    if middle_frame is not None:
        middle_frame.pack_forget()
    middle_frame = tk.Frame(window)
    middle_frame.pack(expand=True, fill=tk.BOTH)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=middle_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    cetvrti_red.pack_forget()
    drugi_red_2.pack_forget()
    drugi_red_3.pack_forget()
    drugi_red_4.pack_forget()
    if selected_algorithm == 1:
        drugi_red.pack_forget()
        drugi_red_5.pack_forget()
        drugi_red_6.pack_forget()
    if selected_algorithm == 2:
        drugi_red.pack_forget()
    if selected_algorithm == 3:
        drugi_red_1.pack_forget()
        drugi_red_5.pack_forget()
        drugi_red_6.pack_forget()

    if peti_red is not None:
        peti_red.pack_forget()
    peti_red = tk.Frame(window)
    peti_red.pack(pady=10)

    label1 = tk.Label(peti_red, text="Unesite vrednost prve kolone za predvidjanje:")
    label1.grid(row=0, column=0, padx=5, pady=5)
    validate_command1 = (window.register(validate_parameter1_for_prediction), '%P')
    entry1 = tk.Entry(peti_red, validate="key", validatecommand=validate_command1)
    entry1.insert(0, str(prediction_parameter1))
    entry1.grid(row=0, column=1, padx=30, pady=5)

    label1 = tk.Label(peti_red, text="Unesite vrednost druge kolone za predvidjanje:")
    label1.grid(row=1, column=0, padx=5, pady=5)
    validate_command1 = (window.register(validate_parameter2_for_prediction), '%P')
    entry1 = tk.Entry(peti_red, validate="key", validatecommand=validate_command1)
    entry1.insert(0, str(prediction_parameter2))
    entry1.grid(row=1, column=1, padx=30, pady=5)

    predict_model_button = tk.Button(peti_red, text="Predvidite izlaz", command=predict_model,
                                     font=('Arial', 12), bg='#4CAF50', fg='white',
                                     activebackground='#45a049', activeforeground='white',
                                     padx=10, pady=5)
    predict_model_button.grid(row=1, column=2, padx=30, pady=5)

    if prediction_output != "":
        global answer_buttons
        global question_label

        bold_font = ('Arial', 12, 'bold')

        output_label = tk.Label(peti_red, text="Izlaz predvidjanja : " + prediction_output, fg="red", font=bold_font)
        output_label.grid(row=1, column=3, padx=30, pady=5)
        output_value = tk.Label(peti_red, text="", fg="blue")
        output_value.grid(row=1, column=4, padx=30, pady=5)

        question_label = tk.Label(peti_red, text="Da li su izlazna predvidjanja tačna?")
        question_label.grid(row=0, column=5, padx=30, pady=5)

        answer_buttons = tk.Frame(peti_red)
        answer_buttons.grid(row=0, column=5, padx=30, pady=5)
        yes_button = tk.Button(answer_buttons, text="Da", command=repack_for_good_prediction,
                               font=('Arial', 12), bg='#4CAF50', fg='white',
                               activebackground='#45a049', activeforeground='white',
                               padx=10, pady=5)
        yes_button.grid(row=0, column=0, padx=30, pady=5)
        no_button = tk.Button(answer_buttons, text="Ne", command=repack_for_retrain,
                              font=('Arial', 12), bg='#4CAF50', fg='white',
                              activebackground='#45a049', activeforeground='white',
                              padx=10, pady=5)
        no_button.grid(row=0, column=1, padx=30, pady=5)
        answer_buttons.grid(row=1, column=5, padx=30, pady=5)


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


def validate_number_of_neibhours_knn(text):
    global number_of_neighbours_knn
    global selected_algorithm
    if text.isdigit() or text == "":
        if text != "":
            number_of_neighbours_knn = int(text)
            if number_of_neighbours_knn > 10 or number_of_neighbours_knn < 3:
                messagebox.showwarning("Opseg izvan preporučenog",
                                       "Preporučen opseg za KNN algoritam je izmedju 3 i 10!")
        return True
    else:
        messagebox.showerror("Pogrešan unos", "Unesite cele brojeve, veće od 0")
        return False


def validate_integer_knn(text):
    global number_of_neighbours_knn
    global selected_algorithm
    if not text.isdigit() and text != "":
        messagebox.showerror("Pogrešan unos", "Unesite cele brojeve, veće od 0")
        return False
    else:
        global leaf_size_knn
        leaf_size_knn = int(text)
        return True


def validate_parameter1_for_prediction(text):
    global prediction_parameter1
    if text.isdigit():
        prediction_parameter1 = int(text)
        return True
    else:
        return False


def validate_parameter2_for_prediction(text):
    global prediction_parameter2
    if text.isdigit():
        prediction_parameter2 = int(text)
        return True
    else:
        return False


def validate_real_output(text):
    global real_output
    if text.isdigit():
        real_output = int(text)
        return True
    else:
        return False


def validate_dept_for_decision_tree(text):
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


def validate_integer_for_decision_tree(text):
    if not text.isdigit() and text != "":
        messagebox.showerror("Pogrešan unos", "Unesite cele brojeve, veće od 0")
        return False
    else:
        global random_state_decision_tree
        random_state_decision_tree = int(text)
        return True


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


def on_function_select_knn(event, chosen):
    if chosen == '1':
        global weight_knn
        selected_item = dropdown_weight_knn.get()
        weight_knn = selected_item
    elif chosen == '2':
        global algorithm_knn
        selected_item = dropdown_algorithm_knn.get()
        algorithm_knn = selected_item
    else:
        global metric_knn
        selected_item = dropdown_metric_knn.get()
        metric_knn = selected_item


def on_function_select_decision_tree(event, chosen):
    if chosen == '1':
        global criterion_decision_tree
        selected_item = dropdown_criterion_decision_tree.get()
        criterion_decision_tree = selected_item
    else:
        global splitter_decision_tree
        selected_item = dropdown_splitter_decision_tree.get()
        splitter_decision_tree = selected_item


# ==================================== Create the intro window ===================================

intro_window = tk.Tk()
intro_window.title("Diplomski rad")
intro_window.iconbitmap("logo.ico")
intro_window.minsize(400, 300)

intro_label = tk.Label(intro_window, text="Autor: Marko Vojinović 2019/0559")
intro_label.pack(pady=10)
intro_label = tk.Label(intro_window, text="Univerzitet u Beogradu, Elektrotehnički Fakultet")
intro_label.pack()

prvi_red = tk.Frame(intro_window)
prvi_red.pack(pady=10)
label = tk.Label(prvi_red, text="Odaberite algoritam za mašinsko učenje: ")
label.pack(side="left")
style = ttk.Style()
style.configure("TCombobox", foreground="red", background="lightgray", padding=6)
dropdown_algorithm = ttk.Combobox(prvi_red, values=["KNN", "Stablo odlučivanja", "Neuralna mreža"], state="readonly",
                                  style="TCombobox")
dropdown_algorithm.pack(side="left")
dropdown_algorithm.set("KNN")
dropdown_algorithm.bind("<<ComboboxSelected>>", on_menu_select)

start_program_button = tk.Button(intro_window, text="Otpočnite sa radom", command=start_program,
                                 font=('Arial', 12), bg='#4CAF50', fg='white',
                                 activebackground='#45a049', activeforeground='white',
                                 padx=10, pady=5)
start_program_button.pack(pady=20)

intro_window.mainloop()

# ==================================== Create the main window ===================================

if window_close:
    window = tk.Tk()
    window.title("Diplomski rad")
    window.iconbitmap("logo.ico")
    window.minsize(600, 450)

    style = ttk.Style()
    style.configure("TCombobox", foreground="red", background="lightgray", padding=6)

    if selected_algorithm == 1:  # KNN

        drugi_red = tk.Frame(window)
        drugi_red.pack(pady=10)
        validate_command = (window.register(validate_number_of_neibhours_knn), '%P')
        label = tk.Label(drugi_red, text="Unesite broj komšija:")
        label.pack(side="left")
        entry = tk.Entry(drugi_red, validate="key", validatecommand=validate_command)
        entry.insert(0, str(number_of_neighbours_knn))
        entry.pack(side="left")

        drugi_red_2 = tk.Frame(window)
        drugi_red_2.pack(pady=10)
        label_weight_knn = tk.Label(drugi_red_2, text="Odaberite funkciju za težinsku obradu:")
        label_weight_knn.pack(side="left")
        dropdown_weight_knn = ttk.Combobox(drugi_red_2, values=["uniform", "distance"],
                                           state="readonly",
                                           style="TCombobox")
        dropdown_weight_knn.pack(side="left")
        dropdown_weight_knn.set("uniform")
        dropdown_weight_knn.bind("<<ComboboxSelected>>", lambda event: on_function_select_knn(event, '1'))

        drugi_red_3 = tk.Frame(window)
        drugi_red_3.pack(pady=10)
        label_algorithm_knn = tk.Label(drugi_red_3, text="Odaberite algoritam unutar funkcije:")
        label_algorithm_knn.pack(side="left")
        dropdown_algorithm_knn = ttk.Combobox(drugi_red_3, values=["auto", "ball_tree", "kd_tree", "brute"],
                                              state="readonly",
                                              style="TCombobox")
        dropdown_algorithm_knn.pack(side="left")
        dropdown_algorithm_knn.set("auto")
        dropdown_algorithm_knn.bind("<<ComboboxSelected>>", lambda event: on_function_select_knn(event, '2'))

        drugi_red_4 = tk.Frame(window)
        drugi_red_4.pack(pady=10)
        validate_command = (window.register(validate_integer_knn), '%P')
        label1 = tk.Label(drugi_red_4, text="Unesite broj listova:")
        label1.pack(side="left")
        entry1 = tk.Entry(drugi_red_4, validate="key", validatecommand=validate_command)
        entry1.insert(0, str(leaf_size_knn))
        entry1.pack(side="left")

        drugi_red_5 = tk.Frame(window)
        drugi_red_5.pack(pady=10)
        validate_command = (window.register(validate_integer_knn), '%P')
        label2 = tk.Label(drugi_red_5, text="Unesite parametar za Minkowski udaljenost:")
        label2.pack(side="left")
        entry2 = tk.Entry(drugi_red_5, validate="key", validatecommand=validate_command)
        entry2.insert(0, str(p_knn))
        entry2.pack(side="left")

        drugi_red_6 = tk.Frame(window)
        drugi_red_6.pack(pady=10)
        label_metric_knn = tk.Label(drugi_red_6, text="Odaberite metriku:")
        label_metric_knn.pack(side="left")
        dropdown_metric_knn = ttk.Combobox(drugi_red_6, values=["minkowski", "euclidean", "haversine", "manhattan"],
                                           state="readonly",
                                           style="TCombobox")
        dropdown_metric_knn.pack(side="left")
        dropdown_metric_knn.set("minkowski")
        dropdown_metric_knn.bind("<<ComboboxSelected>>", lambda event: on_function_select_knn(event, '3'))

    if selected_algorithm == 2:  # Decision tree

        drugi_red = tk.Frame(window)
        drugi_red.pack(pady=10)
        validate_command = (window.register(validate_dept_for_decision_tree), '%P')
        label = tk.Label(drugi_red, text="Unesite najveću dubinu stabla odlučivanja:")
        label.pack(side="left")
        entry = tk.Entry(drugi_red, validate="key", validatecommand=validate_command)
        entry.pack(side="left")

        drugi_red_2 = tk.Frame(window)
        drugi_red_2.pack(pady=10)
        label_criterion_decision_tree = tk.Label(drugi_red_2, text="Odaberite funkciju za kriterijum stabla:")
        label_criterion_decision_tree.pack(side="left")
        dropdown_criterion_decision_tree = ttk.Combobox(drugi_red_2, values=["gini", "entropy", "log_loss"],
                                                        state="readonly",
                                                        style="TCombobox")
        dropdown_criterion_decision_tree.pack(side="left")
        dropdown_criterion_decision_tree.set("gini")
        dropdown_criterion_decision_tree.bind("<<ComboboxSelected>>",
                                              lambda event: on_function_select_decision_tree(event, '1'))

        drugi_red_3 = tk.Frame(window)
        drugi_red_3.pack(pady=10)
        label_splitter_decision_tree = tk.Label(drugi_red_3, text="Odaberite funkciju za podelu stabla:")
        label_splitter_decision_tree.pack(side="left")
        dropdown_splitter_decision_tree = ttk.Combobox(drugi_red_3, values=["best", "random"],
                                                       state="readonly",
                                                       style="TCombobox")
        dropdown_splitter_decision_tree.pack(side="left")
        dropdown_splitter_decision_tree.set("best")
        dropdown_splitter_decision_tree.bind("<<ComboboxSelected>>",
                                             lambda event: on_function_select_decision_tree(event, '2'))

        drugi_red_4 = tk.Frame(window)
        drugi_red_4.pack(pady=10)
        validate_command = (window.register(validate_integer_for_decision_tree), '%P')
        label = tk.Label(drugi_red_4, text="Unesite nasumičan broj za obradu stabla:")
        label.pack(side="left")
        entry = tk.Entry(drugi_red_4, validate="key", validatecommand=validate_command)
        entry.pack(side="left")

    if selected_algorithm == 3:  # Neural network

        drugi_red_1 = tk.Frame(window)
        drugi_red_1.pack(pady=10)
        validate_command = (window.register(validate_integer_for_neural), '%P', '1')
        label_hidden_layer_neural = tk.Label(drugi_red_1, text="Unesite broj sakrivenih slojeva:")
        label_hidden_layer_neural.pack(side="left")
        entry_hidden_layers_neural = tk.Entry(drugi_red_1, validate="key", validatecommand=validate_command)
        entry_hidden_layers_neural.insert(0, str(number_of_hidden_layers_neural))
        entry_hidden_layers_neural.pack(side="left")

        drugi_red_2 = tk.Frame(window)
        drugi_red_2.pack(pady=10)
        label_hidden_layer_neural = tk.Label(drugi_red_2, text="Odaberite funkciju za obradu sakrivenog sloja:")
        label_hidden_layer_neural.pack(side="left")
        dropdown_hidden_layer_neural = ttk.Combobox(drugi_red_2, values=["relu", "sigmoid", "softmax", "tanh"],
                                                    state="readonly",
                                                    style="TCombobox")
        dropdown_hidden_layer_neural.pack(side="left")
        dropdown_hidden_layer_neural.set("relu")
        dropdown_hidden_layer_neural.bind("<<ComboboxSelected>>", lambda event: on_function_select_neural(event, '1'))

        drugi_red_3 = tk.Frame(window)
        drugi_red_3.pack(pady=10)
        label_output_layer_neural = tk.Label(drugi_red_3, text="Odaberite funkciju obrade izlaznog sloja:")
        label_output_layer_neural.pack(side="left")
        dropdown_output_layer_neural = ttk.Combobox(drugi_red_3, values=["relu", "sigmoid", "softmax", "tanh"],
                                                    state="readonly",
                                                    style="TCombobox")
        dropdown_output_layer_neural.pack(side="left")
        dropdown_output_layer_neural.set("sigmoid")
        dropdown_output_layer_neural.bind("<<ComboboxSelected>>", lambda event: on_function_select_neural(event, '2'))

        drugi_red_4 = tk.Frame(window)
        drugi_red_4.pack(pady=10)
        label_optimizer_neural = tk.Label(drugi_red_4, text="Odaberite optimizator:")
        label_optimizer_neural.pack(side="left")
        dropdown_optimizer_neural = ttk.Combobox(drugi_red_4, values=["adam", "SGD", "RMSprop"], state="readonly",
                                                 style="TCombobox")
        dropdown_optimizer_neural.pack(side="left")
        dropdown_optimizer_neural.set("adam")
        dropdown_optimizer_neural.bind("<<ComboboxSelected>>", lambda event: on_function_select_neural(event, '3'))

        drugi_red_5 = tk.Frame(window)
        drugi_red_5.pack(pady=10)
        label_loss_neural = tk.Label(drugi_red_5, text="Odaberite funkciju obrade gubitka:")
        label_loss_neural.pack(side="left")
        dropdown_loss_neural = ttk.Combobox(drugi_red_5,
                                            values=["binary_crossentropy", "mse", "categorical_crossentropy"],
                                            state="readonly",
                                            style="TCombobox")
        dropdown_loss_neural.pack(side="left")
        dropdown_loss_neural.set("binary_crossentropy")
        dropdown_loss_neural.bind("<<ComboboxSelected>>", lambda event: on_function_select_neural(event, '4'))

        drugi_red_6 = tk.Frame(window)
        drugi_red_6.pack(pady=10)
        validate_command = (window.register(validate_integer_for_neural), '%P', '2')
        label_epochs_neural = tk.Label(drugi_red_6, text="Unesite broj epoha:")
        label_epochs_neural.pack(side="left")
        entry_epochs_neural = tk.Entry(drugi_red_6, validate="key", validatecommand=validate_command)
        entry_epochs_neural.insert(0, str(number_of_epochs_neural))
        entry_epochs_neural.pack(side="left")

    treci_red = tk.Frame(window)
    treci_red.pack(pady=10)
    upload_csv_button = tk.Button(treci_red, text="Unesite set podataka u CSV formatu", command=upload_csv,
                                  font=('Arial', 12), bg='#4CAF50', fg='white',
                                  activebackground='#45a049', activeforeground='white',
                                  padx=10, pady=5)
    upload_csv_button.pack(side="left")

    cetvrti_red = tk.Frame(window)
    cetvrti_red.pack(pady=10)

    upload_model_button = tk.Button(cetvrti_red, text="Unesite postojeci model", command=load_model,
                                    font=('Arial', 12), bg='#4CAF50', fg='white',
                                    activebackground='#45a049', activeforeground='white',
                                    padx=10, pady=5)
    upload_model_button.pack(side="left")

    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()

    window.quit()
