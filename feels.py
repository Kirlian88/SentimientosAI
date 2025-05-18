import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import pyttsx3
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
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

# Analizador de sentimientos multilenguaje
class AnalizadorSentimientos:
    def __init__(self):
        self.modelos = {
            'en': {
                'modelo': 'cardiffnlp/twitter-roberta-base-sentiment',
                'labels': {0: "Negativo", 1: "Neutral", 2: "Positivo"}
            },
            'es': {
                'modelo': 'finiteautomata/beto-sentiment-analysis',
                'labels': {0: "Negativo", 1: "Neutral", 2: "Positivo"}
            }
        }
        self.tokenizers = {}
        self.models = {}

        for lang, info in self.modelos.items():
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(info['modelo'])
            self.models[lang] = AutoModelForSequenceClassification.from_pretrained(info['modelo'])

    def detectar_idioma(self, texto):
        try:
            idioma = detect(texto)
            return 'es' if idioma.startswith('es') else 'en'
        except:
            return 'en'

    def predecir(self, texto):
        idioma = self.detectar_idioma(texto)
        tokenizer = self.tokenizers[idioma]
        model = self.models[idioma]
        labels = self.modelos[idioma]['labels']

        inputs = tokenizer(texto, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()
        idx_max = np.argmax(scores)
        return Sentimiento(labels[idx_max], float(scores[idx_max]))

# Interfaz de Luna
class LunaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌙 Luna - Tu Asistente Emocional")
        self.root.configure(bg="#f8f9fa")

        self.analizador = AnalizadorSentimientos()

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('voice', self._voz_espanol())

        self._estilizar()
        self._crear_interfaz()

    def _voz_espanol(self):
        for voice in self.engine.getProperty('voices'):
            if "spanish" in voice.name.lower() or "es" in voice.id.lower():
                return voice.id
        return self.engine.getProperty('voice')

    def _estilizar(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TButton",
                        font=("Segoe UI", 11, "bold"),
                        foreground="white",
                        background="#007bff",
                        padding=10)
        style.map("TButton",
                  background=[("active", "#0056b3")])

    def _crear_interfaz(self):
        # Avatar
        self.avatar_neutral = ImageTk.PhotoImage(Image.open("luna.png").resize((180, 180)))
        self.avatar_hablando = ImageTk.PhotoImage(Image.open("luna_habla.png").resize((180, 180)))
        self.avatar_label = tk.Label(self.root, image=self.avatar_neutral, bg="#f8f9fa")
        self.avatar_label.pack(pady=10)

        # Entrada de texto
        self.entry = tk.Entry(self.root, width=50, font=("Segoe UI", 12))
        self.entry.pack(pady=10)
        self.entry.insert(0, "¿Cómo te sientes hoy? / How do you feel today?")

        # Botones con ttk
        frame_botones = tk.Frame(self.root, bg="#f8f9fa")
        frame_botones.pack(pady=5)

        ttk.Button(frame_botones, text="Analizar emoción", command=self.analizar_sentimiento).grid(row=0, column=0, padx=10)
        ttk.Button(frame_botones, text="Motívame", command=self.motivar).grid(row=0, column=1, padx=10)

        # Resultado
        self.resultado_label = tk.Label(self.root, text="", font=("Segoe UI", 11), wraplength=450, bg="#f8f9fa", justify="center")
        self.resultado_label.pack(pady=15)

    def hablar(self, mensaje):
        stop_animacion = threading.Event()

        def hablar_con_animacion():
            self.engine.say(mensaje)
            self.engine.runAndWait()
            stop_animacion.set()

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

# Ejecutar app
if __name__ == "__main__":
    root = tk.Tk()
    app = LunaApp(root)
    root.mainloop()
