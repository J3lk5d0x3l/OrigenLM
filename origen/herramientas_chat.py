"""Herramientas Web y MCP en el chat de PESOS PROPIOS de OrigenLM.

Se inicializan solamente cuando el usuario usa un comando explícito. El
modelo todavía no está entrenado para decidir llamadas a herramientas.
"""
from .agente.config import cargar_config
from .agente.cli import preparar, mostrar, herramienta_desde_linea


def es_comando_herramienta(texto: str) -> bool:
    return texto == "/herramientas" or texto.startswith(("/buscar ", "/pagina ", "/usar "))


class SesionHerramientas:
    def __init__(self):
        config = cargar_config()
        self.registro, self.gestor = preparar(config)

    def ejecutar(self, texto: str):
        if texto == "/herramientas":
            mostrar(self.registro)
        else:
            print("Herramienta >", herramienta_desde_linea(texto, self.registro))

    def cerrar(self):
        if self.gestor:
            self.gestor.cerrar()
