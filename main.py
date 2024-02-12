from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io


app = FastAPI()

modelo = tf.keras.models.load_model('saved_model/d121_big_v4_BaseModel.h5')
class_names = ["cardboard","glass","metal","organic","paper","plastic","trash"]

# Cargar el modelo SavedModel
modelo_dir = 'saved_model'  # Asegúrate de que este es el camino correcto al directorio del modelo SavedModel
modelo = tf.saved_model.load(modelo_dir)
inferencia = modelo.signatures["serving_default"]

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((224, 224))  # Ajusta este tamaño según las necesidades de tu modelo.
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # Ajusta la preprocesación según tu modelo.

    predictions = modelo.predict(image_array)
    max_pred_index = np.argmax(predictions)  # Obtener el índice de la mayor predicción
    pred_nombre = class_names[max_pred_index]  # Mapear el índice a un nombre de clase
    max_pred_value = predictions[0][max_pred_index]  # Obtener el valor de la predicción más alta

    return JSONResponse(content={"prediction": pred_nombre, "confidence": float(max_pred_value)})


@app.post("/predict2")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((224, 224))  # Ajusta este tamaño según las necesidades de tu modelo.
    image_array = np.expand_dims(np.array(image), axis=0)  # Ajusta la preprocesación según tu modelo.

    image_array = image_array.astype(np.uint8)
    # Preparar el tensor de entrada para el modelo SavedModel
    input_tensor = tf.convert_to_tensor(image_array, dtype=tf.uint8)

    # Realizar la inferencia
    predictions = inferencia(input_tensor)

    # El modelo SavedModel devuelve un diccionario. Asumiremos que tu modelo tiene una sola salida.
    output_name = list(predictions.keys())[0]  # Cambia esto según el nombre de la salida en tu modelo
    predictions = predictions[output_name].numpy()

    max_pred_index = np.argmax(predictions)  # Obtener el índice de la mayor predicción
    pred_nombre = class_names[max_pred_index]  # Mapear el índice a un nombre de clase
    max_pred_value = predictions[0][max_pred_index]  # Obtener el valor de la predicción más alta

    return JSONResponse(content={"prediction": pred_nombre, "confidence": float(max_pred_value)})

if __name__ == '__main__':
    app.run(debug=True)
