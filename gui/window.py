import tkinter as tk
from tkinter import filedialog
from tkinter import ttk


class Architecture:

    def __init__(self):
        pass

    def train(self):
        pass


class Training:

    def __init__(self, model, arch, epochs):
        self.model = model
        self.arch = arch
        self.epochs = epochs

    def run(self):
        self.arch.train()


def main():
    root = tk.Tk()
    root.geometry("400x300")
    root.title("Experiments")
    selected_path = tk.StringVar()
    elements = []
    frame = []

    def open_file_dialog():
        global selected_path
        file_path = filedialog.askopenfilename()
        selected_path.set(file_path)

    def fe_train_model():
        init_frame()

        inital_label = tk.Label(frame, text="Model to train")
        inital_label.pack()
        elements.append(inital_label)

        models = ["ResNet-50", "VGG-16"]
        for i in range(8):
            models.append(f"EfficientNet-B{i}")

        selected_model = tk.StringVar()
        liste = ttk.Combobox(root, values=models, state="readonly")
        liste.set("Select an Option")

        def get_chosen_option(event):
            selected_option = liste.get()
            print(f"Selected Option: {selected_option}")

        liste.bind('<<ComboboxSelected>>', get_chosen_option)
        liste.pack()
        elements.append(liste)

        inital_label = tk.Label(frame, text="Choose Training setup")
        inital_label.pack()
        elements.append(inital_label)

        training_setups = ["Classification", "SimSiam", "MoCov1", "MoCov2", "MoCov3", "BYOL", "SimCLR",
                           "MAE"]

        def get_chosen_optionTS(event):
            selected_option = listeTS.get()
            print(f"Selected Option: {selected_option}")

        selected_TS = tk.StringVar()
        listeTS = ttk.Combobox(root, values=training_setups, state="readonly")
        listeTS.set("Select an Option")
        listeTS.bind('<<ComboboxSelected>>', get_chosen_optionTS)
        listeTS.pack()
        elements.append(listeTS)

        def check_chosen_option():
            selected_option = liste.get()
            if selected_option not in models:
                return
            else:
                feature_extraction()

        inital_label = tk.Label(frame, text="Epochs:")
        inital_label.pack()
        elements.append(inital_label)

        global epochs_var
        epochs_var = tk.StringVar(value="100")
        epochs = ttk.Entry(frame, textvariable=epochs_var)
        epochs.pack()
        elements.append(epochs)

        inital_label = tk.Label(frame, text="Image size:")
        inital_label.pack()
        elements.append(inital_label)

        global img_size_var
        img_size_var = tk.StringVar(value="256")
        img_size = ttk.Entry(frame, textvariable=img_size_var)
        img_size.pack()
        elements.append(img_size)

        initial_button = tk.Button(frame, text="OK", command=check_chosen_option)
        initial_button.pack()
        elements.append(initial_button)

        initial_button = tk.Button(frame, text="Back", command=feature_extraction)
        initial_button.pack()
        elements.append(initial_button)

    def fe_extract_features():
        init_frame()
        # Add a button to open a file dialog and store the selected path
        file_button = tk.Button(frame, text="Select File", command=open_file_dialog)
        file_button.pack()
        elements.append(file_button)

        initial_button = tk.Button(root, text="OK", command=feature_extraction)
        initial_button.pack()
        elements.append(initial_button)

        initial_button = tk.Button(root, text="Back", command=feature_extraction)
        initial_button.pack()
        elements.append(initial_button)

    def feature_extraction():
        init_frame()

        # Display new text and buttons
        inital_label = tk.Label(frame, text="Choose experiment")
        inital_label.pack()
        elements.append(inital_label)

        new_button = tk.Button(frame, text="Train model", command=fe_train_model)
        new_button.pack()
        elements.append(new_button)

        new_button = tk.Button(frame, text="Extract features", command=fe_extract_features)
        new_button.pack()
        elements.append(new_button)

        initial_button = tk.Button(root, text="Back", command=welcome_window)
        initial_button.pack()
        elements.append(initial_button)

    def anomaly_detection():
        init_frame()

        inital_label = tk.Label(frame, text="Algorithm to run")
        inital_label.pack()
        elements.append(inital_label)

        algo = ["KNN"]

        selected_model = tk.StringVar()
        liste = ttk.Combobox(root, values=algo, state="readonly")
        liste.set("Select an Option")

        def get_chosen_option(event):
            selected_option = liste.get()
            print(f"Selected Option: {selected_option}")

        liste.bind('<<ComboboxSelected>>', get_chosen_option)
        liste.pack()
        elements.append(liste)

        def check_chosen_option():
            selected_option = liste.get()
            if selected_option not in algo:
                return
            else:
                ad_knn()

        initial_button = tk.Button(root, text="OK", command=check_chosen_option)
        initial_button.pack()
        elements.append(initial_button)

        initial_button = tk.Button(root, text="Back", command=feature_extraction)
        initial_button.pack()
        elements.append(initial_button)

    def ad_knn():
        init_frame()
        label = tk.Label(root, text="Select an option:")
        label.pack()
        elements.append(label)

        var = tk.StringVar()
        radio_one_k = tk.Radiobutton(root, text="one k", variable=var, value="one k")
        radio_one_k.pack()
        elements.append(radio_one_k)

        radio_multiple_k = tk.Radiobutton(root, text="multiple k", variable=var, value="multiple k")
        radio_multiple_k.pack()
        elements.append(radio_multiple_k)

        entry = tk.Entry(root)
        entry.pack()
        elements.append(entry)

        file_button = tk.Button(frame, text="Select Features File", command=open_file_dialog)
        file_button.pack()
        elements.append(file_button)

        initial_button = tk.Button(root, text="OK", command=lambda x: x)
        initial_button.pack()
        elements.append(initial_button)
        initial_button = tk.Button(root, text="Back", command=feature_extraction)
        initial_button.pack()
        elements.append(initial_button)

    def destroy_all():
        for elt in elements:
            elt.destroy()

    def welcome_window():
        init_frame()

        initial_label = tk.Label(root, text="Choose experiment")
        initial_label.pack()
        elements.append(initial_label)

        fe_button = tk.Button(root, text="Features extraction", command=feature_extraction)
        fe_button.pack()
        elements.append(fe_button)

        ad_button = tk.Button(root, text="Anomaly detection", command=anomaly_detection)
        ad_button.pack()
        elements.append(ad_button)

    def init_frame():
        destroy_all()
        frame = tk.Frame(root)
        frame.pack()
        elements.append(frame)

    welcome_window()

    # Run the GUI application
    root.mainloop()

    # Access the selected path outside the main loop
    print("Selected Path:", selected_path.get())


if __name__ == "__main__":
    main()
