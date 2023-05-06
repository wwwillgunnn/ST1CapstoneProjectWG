# Created by William Gunn u3258398
import tkinter as tk
from PIL import ImageTk
from asteroid_model import Prediction


class AsteroidGUI:
    def __init__(self):

        # Create the main window.
        self.main_window = tk.Tk()
        self.main_window.title("Asteroid Hazard Predictor")
        self.main_window.geometry("1000x650")

        """Create background image"""
        # Create background image
        # Define image
        bg = ImageTk.PhotoImage(file="space.jpeg")
        # Create canvas
        self.canvas = tk.Canvas(self.main_window, width=1000, height=600)
        self.canvas.pack(fill="both", expand=True)
        # Set image
        self.canvas.create_image(0, 0, image=bg, anchor="nw")
        self.canvas.create_text(500, 100, text="Asteroid Hazard Predictor", font=("Helvetica bold", 50), fill="White")

        """Create the widgets for inputs"""
        # Diameter min
        self.canvas.create_text(400, 212, text="Estimated minimum diameter:", font=("Helvetica", 20), fill="White")
        self.est_diameter_min_entry = tk.Entry(self.main_window, bg="white", fg="black", width=20)
        self.text_window = self.canvas.create_window(600, 200, anchor="nw", window=self.est_diameter_min_entry)

        # Diameter max
        self.canvas.create_text(400, 262, text="Estimated maximum diameter:", font=("Helvetica", 20), fill="White")
        self.est_diameter_max_entry = tk.Entry(self.main_window, bg="white", fg="black", width=20)
        self.text_window = self.canvas.create_window(600, 250, anchor="nw", window=self.est_diameter_max_entry)

        # Relative velocity
        self.canvas.create_text(400, 312, text="Relative velocity:", font=("Helvetica", 20), fill="White")
        self.relative_velocity_entry = tk.Entry(self.main_window, bg="white", fg="black", width=20)
        self.text_window = self.canvas.create_window(600, 300, anchor="nw", window=self.relative_velocity_entry)

        # Miss distance
        self.canvas.create_text(400, 362, text="Miss distance:", font=("Helvetica", 20), fill="White")
        self.miss_distance_entry = tk.Entry(self.main_window, bg="white", fg="black", width=20)
        self.text_window = self.canvas.create_window(600, 350, anchor="nw", window=self.miss_distance_entry)

        # Absolute magnitude
        self.canvas.create_text(400, 412, text="Absolute magnitude:", font=("Helvetica", 20), fill="White")
        self.absolute_magnitude_entry = tk.Entry(self.main_window, bg="white", fg="black", width=20)
        self.text_window = self.canvas.create_window(600, 400, anchor="nw", window=self.absolute_magnitude_entry)

        # Buttons
        self.button1 = tk.Button(self.main_window, text="Calculate", command=self.calculate)
        self.button2 = tk.Button(self.main_window, text="Quit", command=self.main_window.destroy)
        self.button1_window = self.canvas.create_window(450, 470, anchor="nw", window=self.button1)
        self.button1_window = self.canvas.create_window(550, 470, anchor="nw", window=self.button2)

        # Enter the tkinter main loop.
        tk.mainloop()

    def calculate(self):
        est_diameter_min = float(self.est_diameter_min_entry.get())
        est_diameter_max = float(self.est_diameter_max_entry.get())
        relative_velocity = float(self.relative_velocity_entry.get())
        miss_distance = float(self.miss_distance_entry.get())
        absolute_magnitude = float(self.absolute_magnitude_entry.get())

        asteroid_info = (est_diameter_min, est_diameter_max, relative_velocity, miss_distance, absolute_magnitude)

        hazardous_prediction = Prediction.best_model.predict([asteroid_info])
        self.canvas.create_text(500, 520, text=f"This prediction has an accuracy of: {Prediction.model_accuracy:.0%}",
                                font=("Helvetica", 20), fill="White")

        if hazardous_prediction == [0]:
            self.canvas.create_text(500, 570, text="The asteroid is safe", font=("Helvetica bold", 30), fill="White")
            # (1279, 1279, 62571, 44783, 358)
        else:
            self.canvas.create_text(500, 570, text="The asteroid is hazardous!", font=("Helvetica bold", 30),
                                    fill="White")
            # (1072, 1072, 49418, 35823, 565)


myGUI = AsteroidGUI()
