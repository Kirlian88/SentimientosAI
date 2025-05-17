from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import requests

# Clase para representar el resultado del sentimiento
class Sentimiento:
    def __init__(self, nombre, confianza):
        self.nombre = nombre
        self.confianza = confianza

    def __str__(self):
        return f"Sentimiento: {self.nombre} (Confianza: {self.confianza:.2f})"

# Analizador con modelo multiclase
class AnalizadorSentimientosMulticlase:
    def __init__(self, modelo='cardiffnlp/twitter-roberta-base-sentiment'):
        print("⏳ Cargando modelo multiclase con neutralidad...")
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelo)
        self.labels = self._cargar_etiquetas(modelo)
        print("✅ Modelo cargado.")

    def _cargar_etiquetas(self, modelo):
        # Cargar etiquetas desde Hugging Face
        url = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
        response = requests.get(url)
        labels = {}
        if response.status_code == 200:
            for i, line in enumerate(response.text.splitlines()):
                labels[i] = line.strip()
        else:
            labels = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Fallback
        return labels

    def predecir(self, texto):
        inputs = self.tokenizer(texto, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()

        idx_max = np.argmax(scores)
        sentimiento = self.labels[idx_max]
        confianza = float(scores[idx_max])
        return Sentimiento(sentimiento, confianza)

# Función principal
def main():
    print("=== Analizador de Sentimientos Multiclase ===")
    analizador = AnalizadorSentimientosMulticlase()

    while True:
        texto = input("\nIngrese el texto a analizar (o 'salir' para terminar): ").strip()
        if texto.lower() == "salir":
            print("👋 ¡Hasta luego!")
            break
        resultado = analizador.predecir(texto)
        print(resultado)

# Ejecutar si es script principal
if __name__ == "__main__":
    main()
