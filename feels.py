import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pyttsx3
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import requests
import threading
import time
import random

# Clase para representar un sentimiento
class Sentimiento:
    def __init__(self, nombre, confianza):
        self.nombre = nombre
        self.confianza = confianza

# Analizador multiclase con modelo de Hugging Face
class AnalizadorSentimientos:
    def __init__(self):
        modelo = 'cardiffnlp/twitter-roberta-base-sentiment'
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelo)
        self.labels = self._cargar_etiquetas(modelo)

    def _cargar_etiquetas(self, modelo):
        url = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
        try:
            response = requests.get(url)
            return {i: line.strip().capitalize() for i, line in enumerate(response.text.splitlines())}
        except:
            return {0: "Negativo", 1: "Neutral", 2: "Positivo"}

    def predecir(self, texto):
        inputs = self.tokenizer(texto, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()
        idx_max = np.argmax(scores)
        return Sentimiento(self.labels[idx_max], float(scores[idx_max]))

# Interfaz de Luna
class LunaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Luna - Tu Asistente Emocional")
        self.analizador = AnalizadorSentimientos()

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('voice', self._voz_espanol())

        self._crear_interfaz()

    def _voz_espanol(self):
        for voice in self.engine.getProperty('voices'):
            if "spanish" in voice.name.lower():
                return voice.id
        return self.engine.getProperty('voice')

    def _crear_interfaz(self):
        self.avatar_neutral = ImageTk.PhotoImage(Image.open("luna.png").resize((180, 180)))
        self.avatar_hablando = ImageTk.PhotoImage(Image.open("luna_habla.png").resize((180, 180)))
        self.avatar_label = tk.Label(self.root, image=self.avatar_neutral)
        self.avatar_label.pack(pady=10)

        self.entry = tk.Entry(self.root, width=50, font=("Arial", 12))
        self.entry.pack(pady=10)
        self.entry.insert(0, "¿Cómo te sientes hoy?")

        tk.Button(self.root, text="Analizar emoción", command=self.analizar_sentimiento).pack(pady=5)
        tk.Button(self.root, text="Motívame", command=self.motivar).pack(pady=5)

        self.resultado_label = tk.Label(self.root, text="", font=("Arial", 12), wraplength=400)
        self.resultado_label.pack(pady=10)

    def hablar(self, mensaje):
        stop_animacion = threading.Event()

        def hablar_con_animacion():
            self.engine.say(mensaje)
            self.engine.runAndWait()
            stop_animacion.set()  # Señal para detener animación

        def animar_boca():
            time.sleep(0.2)
            while not stop_animacion.is_set():
                self.avatar_label.config(image=self.avatar_hablando)
                time.sleep(0.3)
                self.avatar_label.config(image=self.avatar_neutral)
                time.sleep(0.2)
            self.avatar_label.config(image=self.avatar_neutral)

        threading.Thread(target=hablar_con_animacion).start()
        threading.Thread(target=animar_boca).start()

    def analizar_sentimiento(self):
        texto = self.entry.get().strip()
        if not texto:
            messagebox.showwarning("Aviso", "Por favor escribe algo.")
            return

        resultado = self.analizador.predecir(texto)
        mensaje = f"Detecté una emoción {resultado.nombre.lower()} con un {resultado.confianza * 100:.1f}% de certeza."
        self.resultado_label.config(text=mensaje)
        self.hablar(mensaje)

    def motivar(self):
        frases = [
            "Recuerda que cada día es una nueva oportunidad.",
            "Estás haciendo lo mejor que puedes, y eso es suficiente.",
            "Confía en ti. Has superado cosas más difíciles.",
            "Todo va a estar bien, incluso si ahora parece difícil.",
            "Respira hondo, estás avanzando más de lo que crees.",
            "Eres más fuerte de lo que imaginas.",
            "Tus emociones importan. Yo estoy aquí para ti."
        ]
        mensaje = random.choice(frases)
        self.resultado_label.config(text=mensaje)
        self.hablar(mensaje)

# Ejecutar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = LunaApp(root)
    root.mainloop()
