import pickle  # Para guardar y cargar objetos en archivos binarios
from collections import defaultdict  # Para crear diccionarios con valores por defecto
import pandas as pd

# Clase para representar el resultado de sentimiento
class Sentimiento:
    def __init__(self, nombre, confianza):
        self.nombre = nombre  # Nombre del sentimiento (ej. "Felicidad")
        self.confianza = confianza  # Nivel de confianza en la predicción (0.0 a 1.0)

    def __str__(self):
        return f"Sentimiento: {self.nombre} (Confianza: {self.confianza:.2f})"

# Analizador de Sentimientos básico sin aprendizaje
class AnalizadorSentimientosBase:
    def _corregir_errores_ortograficos(self, texto):
        return texto  # Método de corrección aún no implementado

    def _predecir_texto(self, texto):
        texto = self._corregir_errores_ortograficos(texto).lower()
        if "feliz" in texto or "eufórico" in texto:
            return Sentimiento("Felicidad", 0.9)
        elif "triste" in texto or "ganas de nada" in texto:
            return Sentimiento("Tristeza", 0.9)
        else:
            return Sentimiento("Neutral", 0.5)

    def predecir(self, texto):
        return self._predecir_texto(texto)

# Analizador con Aprendizaje Incremental
class AnalizadorSentimientosAprendizaje(AnalizadorSentimientosBase):
    def __init__(self, modelo_path="modelo_sentimientos.pkl"):
        self.modelo_path = modelo_path
        self.ejemplos = defaultdict(list)
        self._cargar_modelo()

    def _cargar_modelo(self):
        try:
            with open(self.modelo_path, "rb") as f:
                self.ejemplos = pickle.load(f)
        except (FileNotFoundError, EOFError):
            self.ejemplos = defaultdict(list)

    def _guardar_modelo(self):
        with open(self.modelo_path, "wb") as f:
            pickle.dump(self.ejemplos, f)

    def ensenar(self, texto, sentimiento):
        texto = self._corregir_errores_ortograficos(texto)
        self.ejemplos[sentimiento].append(texto)
        self._guardar_modelo()

    def _predecir_texto(self, texto):
        texto = self._corregir_errores_ortograficos(texto).lower()
        for sentimiento, frases in self.ejemplos.items():
            for frase in frases:
                if frase.lower() in texto or texto in frase.lower():
                    return Sentimiento(sentimiento, 0.99)
        return super()._predecir_texto(texto)

# Carga ejemplos desde Feels.csv
def cargar_datos_desde_csv(analizador, ruta_csv="Feels.csv"):
    try:
        df = pd.read_csv(ruta_csv)
        for _, fila in df.iterrows():
            texto = str(fila["text"]).strip()
            sentimiento = str(fila["sentiment"]).strip()
            if texto and sentimiento:
                analizador.ensenar(texto, sentimiento)
        print(f"✅ Se cargaron {len(df)} frases desde '{ruta_csv}'")
    except FileNotFoundError:
        print(f"⚠️ No se encontró el archivo '{ruta_csv}'")
    except Exception as e:
        print(f"⚠️ Error al cargar el CSV: {e}")

# Función principal de la aplicación
def main():
    print("=== Bienvenido al Analizador de Sentimientos ===")
    print("Seleccione el modo de funcionamiento:")
    print("1. Modo básico")
    print("2. Modo con aprendizaje incremental")
    modo = input("Opción (1/2): ").strip()

    if modo == "2":
        analizador = AnalizadorSentimientosAprendizaje()
        cargar_datos_desde_csv(analizador)  # Cargar frases desde CSV
        while True:
            print("\nOpciones:")
            print("1. Analizar texto")
            print("2. Enseñar nueva frase")
            print("3. Salir")
            opcion = input("Seleccione una opción: ").strip()

            if opcion == "1":
                texto = input("Ingrese el texto a analizar: ")
                resultado = analizador.predecir(texto)
                print(resultado)
            elif opcion == "2":
                texto = input("Ingrese la nueva frase: ")
                sentimiento = input("¿Qué sentimiento representa?: ")
                analizador.ensenar(texto, sentimiento)
                print("✅ ¡Nuevo ejemplo guardado!")
            elif opcion == "3":
                print("Gracias por usar el analizador. ¡Hasta luego!")
                break
            else:
                print("❌ Opción no válida.")
    else:
        analizador = AnalizadorSentimientosBase()
        while True:
            texto = input("\nIngrese el texto a analizar (o escriba 'salir' para terminar): ")
            if texto.lower() == "salir":
                print("¡Hasta luego!")
                break
            resultado = analizador.predecir(texto)
            print(resultado)

# Ejecuta la función principal si se llama el script directamente
if __name__ == "__main__":
    main()
