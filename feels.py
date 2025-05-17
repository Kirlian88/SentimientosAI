from transformers import pipeline

# Clase para representar el resultado del sentimiento
class Sentimiento:
    def __init__(self, nombre, confianza):
        self.nombre = nombre
        self.confianza = confianza

    def __str__(self):
        return f"Sentimiento: {self.nombre} (Confianza: {self.confianza:.2f})"

# Analizador de sentimientos con modelo preentrenado de Hugging Face
class AnalizadorSentimientosTransformers:
    def __init__(self, modelo='distilbert-base-uncased-finetuned-sst-2-english'):
        print("⏳ Cargando modelo de análisis de sentimientos...")
        self.nlp = pipeline("sentiment-analysis", model=modelo)
        print("✅ Modelo cargado.")

    def predecir(self, texto):
        resultado = self.nlp(texto)[0]
        nombre = "Positivo" if resultado["label"] == "POSITIVE" else "Negativo"
        confianza = resultado["score"]
        return Sentimiento(nombre, confianza)

# Función principal
def main():
    print("=== Analizador de Sentimientos con Transformers ===")
    analizador = AnalizadorSentimientosTransformers()

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
