import tkinter as tk
from tkinter import ttk

# Create the main window
root = tk.Tk()
root.title("Default Value Example")

# Create a StringVar to hold the default value
default_value = tk.StringVar(value="Default Text")

# Create a ttk.Entry widget with the default value
entry = ttk.Entry(root, textvariable=default_value)
entry.pack()

# Run the tkinter main loop
root.mainloop()
