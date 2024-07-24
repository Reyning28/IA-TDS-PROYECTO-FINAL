import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import requests
from io import BytesIO

# Configuración de Azure Custom Vision
prediction_key = '8082e045167b4c4b9e16d1d781712481'
endpoint = 'https://cncustomvisiondemo-prediction.cognitiveservices.azure.com/'
project_id = 'b1e504e1-71ec-421c-b432-fbb7e75efbb0'
publish_iteration_name = 'Iteration1'
predictor = CustomVisionPredictionClient(endpoint, ApiKeyCredentials(in_headers={"Prediction-key": prediction_key}))

def select_image():
    path = filedialog.askopenfilename()
    if path:
        load_image(path)

def load_image(path):
    img = Image.open(path)
    img = img.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk  # Guarda una referencia para evitar que la imagen sea recolectada por el recolector de basura
    panel.image_path = path
    panel.image_data = None
    result_label.config(text="[Standby]")

def load_image_from_url():
    url = url_entry.get()
    try:
        response = requests.get(url)
        response.raise_for_status()
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        img = img.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk  # Guarda una referencia para evitar que la imagen sea recolectada por el recolector de basura
        panel.image_path = None
        panel.image_data = img_data.getvalue()
        result_label.config(text="[Standby]")
    except requests.exceptions.RequestException as e:
        result_label.config(text=f"Error al cargar la imagen: {e}")

def classify_image():
    if hasattr(panel, "image_path") and panel.image_path:
        with open(panel.image_path, "rb") as image_file:
            results = predictor.classify_image(project_id, publish_iteration_name, image_file.read())
    elif hasattr(panel, "image_data") and panel.image_data:
        results = predictor.classify_image(project_id, publish_iteration_name, panel.image_data)
    else:
        result_label.config(text="No hay imagen cargada.")
        return
    
    best_prediction = max(results.predictions, key=lambda p: p.probability)
    result_label.config(text=f"{best_prediction.tag_name}: {best_prediction.probability * 100:.2f}%")

# Configuración de la GUI
root = tk.Tk()
root.title("CN Demo - Cliente")
root.geometry("600x600")
root.config(bg="#333333")

# Configuración del título
title = tk.Label(root, text="CN Demo V1.0", font=("Helvetica", 14, "bold"), fg="white", bg="#333333")
title.pack(pady=10)

# Frame principal para la imagen y los botones
main_frame = tk.Frame(root, bg="#333333")
main_frame.pack(pady=10)

# Definir dimensiones del panel
panel_width = 250
panel_height = 290

# Frame para la imagen
image_frame = tk.Frame(main_frame, bg="#333333", width=panel_width, height=panel_height)
image_frame.grid(row=0, column=0, padx=10, pady=10)
image_frame.pack_propagate(False)  # Evitar que el frame cambie de tamaño

# Panel de la imagen con tamaño fijo
panel = tk.Label(image_frame, bg="#444444", width=panel_width, height=panel_height, bd=2, relief="solid")
panel.pack(expand=True, fill="both")

# Frame para los botones
btn_frame = tk.Frame(main_frame, bg="#333333")
btn_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Botón para seleccionar imagen
select_button = tk.Button(btn_frame, text="Seleccionar imagen", command=select_image, bg="#555555", fg="white", font=("Helvetica", 10))
select_button.pack(pady=5, fill="x")

# Entrada de URL para cargar imagen desde URL
url_entry = tk.Entry(btn_frame, bg="#555555", fg="white", font=("Helvetica", 10))
url_entry.pack(pady=5, fill="x")

# Botón para cargar imagen desde URL
load_url_button = tk.Button(btn_frame, text="Cargar desde URL", command=load_image_from_url, bg="#555555", fg="white", font=("Helvetica", 10))
load_url_button.pack(pady=5, fill="x")

# Botón para clasificar imagen
classify_button = tk.Button(btn_frame, text="Clasificar imagen", command=classify_image, bg="#555555", fg="white", font=("Helvetica", 10))
classify_button.pack(pady=5, fill="x")

# Etiqueta para mostrar el resultado
result_label = tk.Label(root, text="Resultado:\n[Standby]", font=("Helvetica", 10), fg="white", bg="#333333", bd=2, relief="solid", width=30, height=2)
result_label.pack(pady=10)

root.mainloop()