<div align="center">

# OrigenLM

**Inteligencia artificial creada desde Colombia.**

Un proyecto independiente para desarrollar modelos de lenguaje con pesos propios, investigación abierta y una comunidad de colaboradores de Latinoamérica.

[![Tests](https://github.com/J3lk5d0x3l/OrigenLM/actions/workflows/tests.yml/badge.svg)](https://github.com/J3lk5d0x3l/OrigenLM/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-555555.svg)](LICENSE)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/J3lk5d0x3l)

[**Comenzar**](#comenzar) · [**Colaborar**](#colaborar) · [**Patrocinar**](#patrocinar) · [**Discord**](https://discord.vyrron.net)

</div>

---

## El proyecto

**OrigenLM** busca construir una inteligencia artificial desarrollada desde cero, con identidad colombiana y participación abierta de la comunidad. El código incluye modelos Transformer, tokenizadores propios, herramientas de entrenamiento, evaluación y una interfaz experimental por terminal.

El propósito es avanzar hacia un asistente accesible y útil, construido con transparencia y colaboración.

> OrigenLM sigue en investigación. El código público no incluye los checkpoints entrenados ni implica que exista ya un asistente de producción.

## Comenzar

Requiere Python 3.11 o superior.

```bash
git clone https://github.com/J3lk5d0x3l/OrigenLM.git
cd OrigenLM
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m origen.cli estado
```

En Codespaces también puedes utilizar el comando `origen` para abrir el proyecto una vez instalado el entorno. Consulta la [guía de desarrollo](docs/DESARROLLO.md) para preparar datos y trabajar con los entrenadores.

## Estructura

```text
origen/       Arquitecturas, tokenizadores y herramientas
scripts/      Procesamiento de datos, entrenamiento y utilidades
dataset/      Ejemplos públicos y datos de evaluación
tests/        Pruebas automatizadas del proyecto
docs/         Documentación técnica
train*.py     Entrenadores
chat*.py      Interfaces de conversación experimentales
```

## Colaborar

¿Te gustaría mejorar el modelo, aportar datos autorizados, programar herramientas o trabajar en la infraestructura?

**[Únete a nuestro Discord](https://discord.vyrron.net)** y cuéntanos que quieres colaborar en OrigenLM, en qué área te gustaría participar y qué experiencia tienes. Coordinamos propuestas y después trabajamos mediante *issues* y *pull requests*.

Lee la [guía para colaboradores](CONTRIBUTING.md) antes de enviar cambios.

## Patrocinar

OrigenLM necesita infraestructura, almacenamiento y capacidad de cómputo para seguir creciendo. Los patrocinios apoyan su investigación y el objetivo de hacer el modelo cada vez más accesible públicamente.

**[Apoya el desarrollo en GitHub Sponsors](https://github.com/sponsors/J3lk5d0x3l)**

---

<div align="center">
  <sub>
    Código bajo <a href="LICENSE">licencia MIT</a> ·
    Nombre e identidad de OrigenLM sujetos a la <a href="TRADEMARKS.md">política de marca</a> ·
    <a href="https://discord.vyrron.net">Comunidad Discord</a>
  </sub>
</div>
