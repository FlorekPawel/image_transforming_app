import tkinter as tk
from tkinter import filedialog, StringVar, Scale
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from plots import *
from filters import *

class App():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Viewer")
        self.root.geometry("1100x800")
        self.root.resizable(width=False, height=False)
        self.root.configure(bg="gray20")
        
        self.frame = tk.Frame(self.root, bg="gray20")
        self.frame.pack(pady=20)         
        
        self.image_label_1 = tk.Label(self.frame, text="Upload image", width=35, height=20, bg="gray")
        self.image_label_1.grid(row=0, column=0, padx=20)
        
        self.upload_button = tk.Button(self.frame, text="Upload Image", command=self.upload_image)
        self.upload_button.grid(row=1, column=0, padx=10, pady=10)
        
        self.image_label_2 = tk.Label(self.frame, text="Transfrom image", width=35, height=20, bg="gray")
        self.image_label_2.grid(row=0, column=2, padx=20)
        
        self.transform_frame = tk.Frame(self.frame, bg="gray20")
        self.transform_frame.grid(row=0, column=1, padx=10)
        
        self.transform_options = [
            "Grayscale", 
            "Binary",
            "Brightness", 
            "Contrast",
            "Negative",
            "Binarisation",
            "Averaging",
            "Median",
            "Kuwahara",
            "Gaussian",
            "Sharpening",
            "High Pass",
            "Ridge",
            "Roberts",
            "Prewitt",
            "Sobel",
            "Scharr",
            "Laplace",
        ]
        self.selected_option = StringVar(self.root)
        self.selected_option.set(self.transform_options[0])
        self.dropdown_menu = tk.OptionMenu(self.transform_frame, self.selected_option, *self.transform_options, command=self.update_slider)
        self.dropdown_menu.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=10)
        
        self.slider = ttk.Scale(self.transform_frame, state="disabled", value=1, from_=1, to=10, orient=tk.HORIZONTAL, command=self.update_slider_label)
        self.slider.grid(row=1, column=0, sticky="w", padx=0, pady=10)
        
        self.slider_value_label = tk.Label(self.transform_frame, text="0", width=5)
        self.slider_value_label.grid(row=1, column=1, pady=10)
        
        self.transform_button = tk.Button(self.transform_frame, text="Transform Image", command=self.transform_image)
        self.transform_button.config(width=12)
        self.transform_button.grid(row=2, column=0, sticky="w", padx=0, pady=(5,5))
        
        self.swap_button = tk.Button(self.transform_frame, text="Swap Images", command=self.swap_images)
        self.swap_button.config(width=12)
        self.swap_button.grid(row=3, column=0, sticky="w", padx=0)
        
        self.save_button = tk.Button(self.frame, text="Save Image", command=self.save_image)
        self.save_button.grid(row=1, column=2, padx=10, pady=10)

        self.plot_frame = tk.Frame(self.frame, bg="gray20")
        self.plot_frame.grid(row=2, column=1, columnspan=2, padx=10, pady=20)
        
    def upload_image(self):
        
        file_path = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpeg")])
        if file_path:
            self.image = Image.open(file_path)
            self.image = self.image.resize((320, 320), Image.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(self.image)

            self.image_label_1.destroy()
            self.image_label_1 = tk.Label(self.frame, image=self.image_tk)
            self.image_label_1.grid(row=0, column=0, padx=20)
                                
            plot_hist(self.image, self.plot_frame) 
                      
            self.plot_option_frame = tk.Frame(self.frame, bg="gray20")
            self.plot_option_frame.grid(row=2, column=0, padx=(0,20), sticky="e")
            
            self.plot_options = ["Histogram", "Horizontal Projection", "Vertical Projection"]
            self.selected_plot = StringVar(self.root)
            self.selected_plot.set(self.plot_options[0])
            self.dropdown_menu_plot = tk.OptionMenu(self.plot_option_frame, self.selected_plot, *self.plot_options)
            self.dropdown_menu_plot.grid(row=0, column=0, padx=(0, 20))
            self.dropdown_menu_plot.config(width=15)
            
            self.image_options = ["Original", "Transformed"]
            self.selected_image = StringVar(self.root)
            self.selected_image.set(self.image_options[0])
            self.dropdown_menu_image = tk.OptionMenu(self.plot_option_frame, self.selected_image, *self.image_options)
            self.dropdown_menu_image.grid(row=1, column=0, padx=(0, 20))
            self.dropdown_menu_image.config(width=15)
            self.dropdown_menu_image.config(state="disabled")
            
            self.plot_button = tk.Button(self.plot_option_frame, text="Generate plot", command=self.plot)
            self.plot_button.config(width=10)
            self.plot_button.grid(row=2, column=0, sticky="w", padx=0, pady=10)
                    
    def save_image(self):
        if hasattr(self, 'image_tk_transformed'):
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpeg")])
            if file_path:
                self.image_tk_transformed._PhotoImage__photo.write(file_path)
        else:
            tk.messagebox.showerror("Error", "No transformed image to save.")
                    
    def swap_images(self):
        if hasattr(self, 'image_tk_transformed'):
            img_width, img_height = self.image_tk.width(), self.image_tk.height()

            self.image_label_1.destroy()
            self.image_label_1 = tk.Label(self.frame, image=self.image_tk_transformed, width=img_width, height=img_height)
            self.image_label_1.grid(row=0, column=0, padx=20)
            
            self.image_tk = self.image_tk_transformed
            self.image = self.image_transformed
            del self.image_tk_transformed
            del self.image_transformed
            
            self.image_label_2.destroy()
            self.image_label_2 = tk.Label(self.frame, text="Transfrom image", width=35, height=20, bg="gray")
            self.image_label_2.grid(row=0, column=2, padx=10)
            
            self.dropdown_menu_image.config(state="normal")
        else:
            tk.messagebox.showerror("Error", "No transformed image to swap.")
                
    def update_slider_label(self, value):
        self.slider_value_label.config(text=str(int(float(value))))
        
    def update_slider(self, value):
        if value == "Grayscale":
            self.slider.config(state="disabled")
            
        elif value == "Binary":
            self.slider.config(state="normal", value=100, from_=0, to=255)

        elif value == "Brightness":
            self.slider.config(state="normal", value=0, from_=-255, to=255)

        elif value == "Contrast":
            self.slider.config(state="normal", value=1, from_=0.1, to=4)

        elif value == "Negative":
            self.slider.config(state="disabled")

        elif value == "Binarisation":
            self.slider.config(state="normal", value=0, from_=0, to=255)

        elif value == "Averaging":
            self.slider.config(state="normal", value=1, from_=1, to=10)

        elif value == "Median":
            self.slider.config(state="normal", value=1, from_=1, to=10)

        elif value == "Gaussian":
            self.slider.config(state="normal", value=1, from_=0.1, to=5)

        elif value == "Sharpening":
            self.slider.config(state="normal", value=1, from_=1, to=20)

        elif value == "Roberts":
            self.slider.config(state="disabled")

        elif value == "Sobel":
            self.slider.config(state="disabled")

        elif value == "High Pass":
            self.slider.config(state="normal", value=10, from_=4, to=20)
        
        elif value == "Laplace":
            self.slider.config(state="disabled")
            
        elif value == "Prewitt":
            self.slider.config(state="disabled")

        elif value == "Kuwahara":
            self.slider.config(state="normal", value=1, from_=1, to=10)
            
        elif value == "Ridge":
            self.slider.config(state="disabled")
            
        elif value == "Scharr":
            self.slider.config(state="disabled")
        
        self.update_slider_label(self.slider.get())
            
    def transform_image(self):
        if hasattr(self, 'image_tk'):
            selected_option = self.selected_option.get()
            slider_value = self.slider.get()
            
            filters = {
                "Grayscale": grayscale,
                "Binary": binary,
                "Brightness": brightness_correction,
                "Contrast": contrast_correction,
                "Negative": negative_filter,
                "Binarisation": binarisation,
                "Averaging": averaging,
                "Median": median,
                "Kuwahara": kuwahara,
                "Ridge": ridge,
                "Gaussian": gaussian,
                "Sharpening": sharpening,
                "Roberts": roberts,
                "Sobel": sobel,
                "High Pass": high_pass,
                "Laplace": laplace,
                "Prewitt": prewitt,
                "Scharr": scharr, 
            }
            
            self.image_transformed = filters[selected_option](self.image, slider_value)
            
            self.image_transformed = self.image_transformed.resize((320, 320), Image.LANCZOS)
            self.image_tk_transformed = ImageTk.PhotoImage(self.image_transformed)

            self.image_label_2.destroy()
            self.image_label_2 = tk.Label(self.frame, image=self.image_tk_transformed)
            self.image_label_2.grid(row=0, column=2, padx=20)
            
            self.dropdown_menu_image.config(state="normal")
        else:
            tk.messagebox.showerror("Error", "No image to transfrom.")
        
    def plot(self):
        selected_plot = self.selected_plot.get() 
        if hasattr(self, 'selected_image'):
            selected_image = self.selected_image.get()
        else:
            selected_image = "Original"        
        image = self.image if selected_image == "Original" else self.image_transformed
        
        
        if selected_plot == "Histogram":
            plot_hist(image, self.plot_frame)
        elif selected_plot == "Horizontal Projection":
            plot_horizontal_projection(image, self.plot_frame)
        elif selected_plot == "Vertical Projection":
            plot_vertical_projection(image, self.plot_frame)
        
    def run(self):
        self.root.mainloop()
        
App().run()