import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd

class CostTableSelector:
    def __init__(self, parent):
        self.frame = tk.Frame(parent)

        # Label
        tk.Label(self.frame, text="Health Cost Table:", font=("Arial", 12)).grid(row=0, column=0, sticky="w")

        # Entry for path
        self.entry = tk.Entry(self.frame, width=70, font=("Arial", 12))
        self.entry.grid(row=0, column=1, padx=5)
        self.entry.insert(0, r"C:\Users\74007\Downloads\Stanford University\0_input_data\health_cost_table.xlsx")

        # Browse button
        browse_btn = tk.Button(self.frame, text="Browse", font=("Arial", 12), command=self.browse)
        browse_btn.grid(row=0, column=2, padx=5)

        # Input button
        input_btn = tk.Button(self.frame, text="Input", font=("Arial", 12), command=self.load_table)
        input_btn.grid(row=0, column=3, padx=5)

        # Dropdown
        self.combo_var = tk.StringVar()
        self.combo = ttk.Combobox(self.frame, textvariable=self.combo_var, font=("Arial", 12), state="readonly")
        self.combo.grid(row=1, column=1, columnspan=2, pady=5, sticky="w")
        self.combo.bind("<<ComboboxSelected>>", self.display_cost)

        # Label to show cost
        self.cost_label = tk.Label(self.frame, text="", font=("Arial", 12))
        self.cost_label.grid(row=2, column=1, columnspan=2, sticky="w")

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, path)

    def load_table(self):
        path = self.entry.get()
        try:
            df = pd.read_excel(path)
            if "region" in df.columns:
                self.df = df
                self.combo['values'] = df['region'].dropna().unique().tolist()
                self.combo.current(0)
                self.display_cost()
        except Exception as e:
            print("Failed to load file:", e)

    def display_cost(self, event=None):
        selected = self.combo_var.get()
        try:
            value = self.df.loc[self.df['region'] == selected, 'cost_value'].values[0]
            self.cost_label.config(text=f"cost = {value}")
        except:
            self.cost_label.config(text="")
