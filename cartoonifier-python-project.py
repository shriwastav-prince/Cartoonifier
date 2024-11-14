import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Radiobutton, StringVar, Button, Label
from PIL import ImageTk, Image
import cv2
import numpy as np

# Global variables
undo_stack = []
redo_stack = []
cartoon_image = None  # To store the cartoonified image
cap = None  # For the webcam capture
captured_image = None  # To store the captured image

def add_to_undo_stack(image):
    global undo_stack
    undo_stack.append(image.copy())

def undo():
    global cartoon_image
    if undo_stack:
        redo_stack.append(cartoon_image)
        cartoon_image = undo_stack.pop()
        display_image(cartoon_image)

def redo():
    global cartoon_image
    if redo_stack:
        add_to_undo_stack(cartoon_image)
        cartoon_image = redo_stack.pop()
        display_image(cartoon_image)

def cartoonify(image):
    # convert the image to RGB format
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # converting an image to grayscale
    grayScaleImage = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # applying median blur to smoothen an image
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)

    # retrieving the edges for cartoon effect by using thresholding technique
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 9, 9)

    # applying bilateral filter to remove noise and keep edge sharp as required
    colorImage = cv2.bilateralFilter(original_image, 9, 300, 300)

    # masking edged image with our "BEAUTIFY" image
    cartoon_image = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)

    # Return the cartoonified image
    return cartoon_image

def display_image(image):
    try:
        # Resize the image to fit the display panel
        ReSized = cv2.resize(image, (720, 405))  # Increase the size to 720x405 (or adjust as needed)
        im = Image.fromarray(ReSized)
        imgtk = ImageTk.PhotoImage(image=im)
        panel.config(image=imgtk)
        panel.image = imgtk

    except Exception as e:
        messagebox.showerror("Error", str(e))

def start_webcam():
    global cap
    cap = cv2.VideoCapture(0)  # Initialize webcam capture
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    def show_frame():
        global cap
        if cap is None or not cap.isOpened():
            return
        ret, frame = cap.read()  # Read frame from the webcam
        if ret:
            cartoon_frame = cartoonify(frame)  # Apply cartoonify function to the frame
            display_image(cartoon_frame)
            panel.after(10, show_frame)  # Update the frame every 10 milliseconds

    show_frame()

def stop_webcam():
    global cap
    if cap:
        cap.release()  # Release the webcam
        cap = None
        cv2.destroyAllWindows()
        panel.config(image='')  # Clear the image panel

def capture_image():
    global cap, captured_image, cartoon_image
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "Webcam is not started.")
        return

    ret, frame = cap.read()  # Capture the frame
    if ret:
        captured_image = frame  # Store the captured frame
        cartoon_image = cartoonify(captured_image)  # Cartoonify the captured image
        add_to_undo_stack(cartoon_image)
        display_image(cartoon_image)
        messagebox.showinfo("Info", "Image captured and cartoonified.")

def upload_image():
    global cartoon_image
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        cartoon_image = cartoonify(image)
        add_to_undo_stack(cartoon_image)
        display_image(cartoon_image)

def save_image():
    global cartoon_image
    if cartoon_image is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Info", "Image saved successfully.")
    else:
        messagebox.showerror("Error", "No image to save.")

def translate_image():
    global cartoon_image
    tx = simpledialog.askinteger("Input", "Enter translation in x direction:", minvalue=-500, maxvalue=500)
    ty = simpledialog.askinteger("Input", "Enter translation in y direction:", minvalue=-500, maxvalue=500)
    if tx is not None and ty is not None:
        chosen_image = cartoon_image
        if chosen_image is not None:
            rows, cols, _ = chosen_image.shape
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            cartoon_image = cv2.warpAffine(chosen_image, M, (cols, rows))
            add_to_undo_stack(chosen_image)
            display_image(cartoon_image)

def rotate_image():
    global cartoon_image
    angle = simpledialog.askfloat("Input", "Enter rotation angle:", minvalue=-360, maxvalue=360)
    if angle is not None:
        chosen_image = cartoon_image
        if chosen_image is not None:
            rows, cols, _ = chosen_image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            cartoon_image = cv2.warpAffine(chosen_image, M, (cols, rows))
            add_to_undo_stack(chosen_image)
            display_image(cartoon_image)

def scale_image():
    global cartoon_image
    XS = simpledialog.askfloat("Input", "Enter scale factor along x: ", minvalue=0.1, maxvalue=10.0)
    YS = simpledialog.askfloat("Input", "Enter scale factor along y: ", minvalue=0.1, maxvalue=10.0)
    if XS or YS is not None:
        chosen_image = cartoon_image
        if chosen_image is not None:
            cartoon_image = cv2.resize(chosen_image, None, fx=XS, fy=YS)
            add_to_undo_stack(chosen_image)
            display_image(cartoon_image)

def change_color():
    global cartoon_image
    chosen_image = cartoon_image
    if chosen_image is not None:
        hsv_image = cv2.cvtColor(chosen_image, cv2.COLOR_RGB2HSV)
        h = simpledialog.askinteger("Input", "Enter new hue value (0-179):", minvalue=0, maxvalue=179)
        s = simpledialog.askinteger("Input", "Enter new saturation value (0-255):", minvalue=0, maxvalue=255)
        v = simpledialog.askinteger("Input", "Enter new brightness value (0-255):", minvalue=0, maxvalue=255)
        if h is not None and s is not None and v is not None:
            hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] + h, 0, 179)  # Hue
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + s, 0, 255)  # Saturation
            hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + v, 0, 255)  # Brightness
            cartoon_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
            add_to_undo_stack(chosen_image)
            display_image(cartoon_image)

def change_texture():
    global cartoon_image
    chosen_image = cartoon_image
    if chosen_image is not None:
        texture_var = StringVar(value='None')
        texture_window = Toplevel(top)
        texture_window.title("Select Texture")

        textures = ['None', 'Blur', 'Sharpen']
        for texture in textures:
            Radiobutton(texture_window, text=texture, variable=texture_var, value=texture).pack(anchor=tk.W)

        Button(texture_window, text="Apply", command=lambda: apply_texture(texture_var.get(), chosen_image)).pack()

def apply_texture(choice, chosen_image):
    global cartoon_image
    if chosen_image is not None:
        if choice == 'Blur':
            ksize = simpledialog.askinteger("Input", "Enter kernel size for blur (odd number):", minvalue=1)
            if ksize is not None:
                if ksize % 2 == 0:
                    ksize += 1  # Ensure the kernel size is odd
                cartoon_image = cv2.GaussianBlur(chosen_image, (ksize, ksize), 0)
                add_to_undo_stack(chosen_image)
                display_image(cartoon_image)
        elif choice == 'Sharpen':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            cartoon_image = cv2.filter2D(chosen_image, -1, kernel)
            add_to_undo_stack(chosen_image)
            display_image(cartoon_image)

def apply_filter():
    global cartoon_image
    chosen_image = cartoon_image
    if chosen_image is not None:
        filter_var = StringVar(value='None')
        filter_window = Toplevel(top)
        filter_window.title("Select Filter")

        filters = ['None', 'Grayscale', 'Sepia', 'Edge Detection']
        for filter_name in filters:
            Radiobutton(filter_window, text=filter_name, variable=filter_var, value=filter_name).pack(anchor=tk.W)

        Button(filter_window, text="Apply", command=lambda: apply_selected_filter(filter_var.get(), chosen_image)).pack()

def apply_selected_filter(choice, chosen_image):
    global cartoon_image
    if chosen_image is not None:
        if choice == 'Grayscale':
            cartoon_image = cv2.cvtColor(chosen_image, cv2.COLOR_RGB2GRAY)
            cartoon_image = cv2.cvtColor(cartoon_image, cv2.COLOR_GRAY2RGB)
        elif choice == 'Sepia':
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            cartoon_image = cv2.transform(chosen_image, sepia_filter)
            cartoon_image = np.clip(cartoon_image, 0, 255).astype(np.uint8)
        elif choice == 'Edge Detection':
            gray = cv2.cvtColor(chosen_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            cartoon_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        add_to_undo_stack(chosen_image)
        display_image(cartoon_image)

def detect_faces():
    global cartoon_image
    if cartoon_image is not None:
        gray = cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Post-processing to filter out non-faces based on size and aspect ratio
        valid_faces = []
        for (x, y, w, h) in faces:
            # Perform additional checks if necessary
            if w > 0 and h > 0 and w / h > 0.5 and h / w > 0.5:  # Adjust aspect ratio thresholds as needed
                valid_faces.append((x, y, w, h))
        
        # Draw rectangles on valid face detections
        for (x, y, w, h) in valid_faces:
            cv2.rectangle(cartoon_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        add_to_undo_stack(cartoon_image)
        display_image(cartoon_image)

# Create the main window
top = tk.Tk()
top.geometry("960x600")
top.title("Image Cartoonifier")

# Update the bottom frame and right frame with slate background color
bottom_frame = tk.Frame(top, height=60, bg='slate gray')
bottom_frame.pack(side="bottom", fill="x")

right_frame = tk.Frame(top, width=180, bg='slate gray')
right_frame.pack(side="right", fill="y")

# Create a frame for the image display (left side)
left_frame = tk.Frame(top, width=480, height=540)
left_frame.pack(side="left", fill="both", expand="yes")

# Create a panel to display the image
panel = Label(left_frame)
panel.pack(side="top", fill="both", expand="yes")

# Add buttons to the bottom frame
upload_button = Button(bottom_frame, text="Upload", command=upload_image)
upload_button.pack(side="left", padx=5, pady=5)

undo_button = Button(bottom_frame, text="Undo", command=undo)
undo_button.pack(side="left", padx=5, pady=5)

redo_button = Button(bottom_frame, text="Redo", command=redo)
redo_button.pack(side="left", padx=5, pady=5)

save_button = Button(bottom_frame, text="Save", command=save_image)
save_button.pack(side="left", padx=5, pady=5)

# Add operational buttons to the right frame
start_button = Button(right_frame, text="Start Webcam", command=start_webcam)
start_button.pack(side="top", padx=5, pady=5)

stop_button = Button(right_frame, text="Stop Webcam", command=stop_webcam)
stop_button.pack(side="top", padx=5, pady=5)

capture_button = Button(right_frame, text="Capture", command=capture_image)
capture_button.pack(side="top", padx=5, pady=5)

translate_button = Button(right_frame, text="Translate", command=translate_image)
translate_button.pack(side="top", padx=5, pady=5)

rotate_button = Button(right_frame, text="Rotate", command=rotate_image)
rotate_button.pack(side="top", padx=5, pady=5)

scale_button = Button(right_frame, text="Scale", command=scale_image)
scale_button.pack(side="top", padx=5, pady=5)

color_button = Button(right_frame, text="Color", command=change_color)
color_button.pack(side="top", padx=5, pady=5)

texture_button = Button(right_frame, text="Texture", command=change_texture)
texture_button.pack(side="top", padx=5, pady=5)

filter_button = Button(right_frame, text="Filter", command=apply_filter)
filter_button.pack(side="top", padx=5, pady=5)

face_button = Button(right_frame, text="Detect Faces", command=detect_faces)
face_button.pack(side="top", padx=5, pady=5)

# Start the main event loop
top.mainloop()
