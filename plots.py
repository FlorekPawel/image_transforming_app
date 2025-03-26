import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import tkinter as tk


def plot_hist(image, plot_frame):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    
    if image:
        image_array = np.array(image)
        
        if image_array.ndim == 3: 
            r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

            ax.hist(r.ravel(), bins=128, color='red', alpha=0.5, label='Red')
            ax.hist(g.ravel(), bins=128, color='green', alpha=0.5, label='Green')
            ax.hist(b.ravel(), bins=128, color='blue', alpha=0.5, label='Blue')
        else:  
            ax.hist(image_array.ravel(), bins=128, color='gray', alpha=0.5, label='Gray')

        ax.set_title("Histogram")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 255)

        for widget in plot_frame.winfo_children():
            widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def plot_vertical_projection(image, plot_frame):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    if image:
        image_array = np.array(image)
        
        if image_array.ndim == 3: 
            vertical_projection_red = np.sum(image_array[:, :, 0], axis=1)
            vertical_projection_green = np.sum(image_array[:, :, 1], axis=1)
            vertical_projection_blue = np.sum(image_array[:, :, 2], axis=1)

            vertical_projection_red = vertical_projection_red / vertical_projection_red.max()
            vertical_projection_green = vertical_projection_green / vertical_projection_green.max()
            vertical_projection_blue = vertical_projection_blue / vertical_projection_blue.max()

            ax.plot(vertical_projection_red, range(len(vertical_projection_red)), color="red", linewidth=2, label='Red')
            ax.plot(vertical_projection_green, range(len(vertical_projection_green)), color="green", linewidth=2, label='Green')
            ax.plot(vertical_projection_blue, range(len(vertical_projection_blue)), color="blue", linewidth=2, label='Blue')
        else: 
            vertical_projection_gray = np.sum(image_array, axis=1)
            vertical_projection_gray = vertical_projection_gray / vertical_projection_gray.max()
            ax.plot(vertical_projection_gray, range(len(vertical_projection_gray)), color="gray", linewidth=2, label='Gray')

        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Vertical Projection")

        for widget in plot_frame.winfo_children():
            widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
def plot_horizontal_projection(image, plot_frame):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    if image:
        image_array = np.array(image)
        
        if image_array.ndim == 3: 
            horizontal_projection_red = np.sum(image_array[:, :, 0], axis=0)
            horizontal_projection_green = np.sum(image_array[:, :, 1], axis=0)
            horizontal_projection_blue = np.sum(image_array[:, :, 2], axis=0)

            horizontal_projection_red = horizontal_projection_red / horizontal_projection_red.max()
            horizontal_projection_green = horizontal_projection_green / horizontal_projection_green.max()
            horizontal_projection_blue = horizontal_projection_blue / horizontal_projection_blue.max()

            ax.plot(range(len(horizontal_projection_red)), horizontal_projection_red, color="red", linewidth=2, label='Red')
            ax.plot(range(len(horizontal_projection_green)), horizontal_projection_green, color="green", linewidth=2, label='Green')
            ax.plot(range(len(horizontal_projection_blue)), horizontal_projection_blue, color="blue", linewidth=2, label='Blue')
        else: 
            horizontal_projection_gray = np.sum(image_array, axis=0)
            horizontal_projection_gray = horizontal_projection_gray / horizontal_projection_gray.max()
            ax.plot(range(len(horizontal_projection_gray)), horizontal_projection_gray, color="gray", linewidth=2, label='Gray')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Horizontal Projection")

        for widget in plot_frame.winfo_children():
            widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)