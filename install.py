#!/usr/bin/env python3
"""
Script de instalación para Soccer Agent
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Imprimir banner de bienvenida"""
    print("""
⚽ Soccer Agent - Instalador
============================
Un agente inteligente para análisis completo de futbolistas
utilizando datos de StatsBomb, machine learning y LLMs.
    """)

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando versión de Python...")
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True

def install_dependencies():
    """Instalar dependencias"""
    print("\n📦 Instalando dependencias...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def setup_environment():
    """Configurar archivo de entorno"""
    print("\n🔧 Configurando variables de entorno...")
    
    env_file = Path(".env")
    example_file = Path("env_example.txt")
    
    if env_file.exists():
        print("⚠️  El archivo .env ya existe")
        response = input("¿Deseas sobrescribirlo? (y/N): ").lower()
        if response != 'y':
            print("✅ Configuración de entorno saltada")
            return True
    
    if not example_file.exists():
        print("❌ No se encontró el archivo env_example.txt")
        return False
    
    # Copiar archivo de ejemplo
    shutil.copy(example_file, env_file)
    print("✅ Archivo .env creado desde env_example.txt")
    
    # Solicitar API key
    print("\n🔑 Configuración de API Key:")
    print("Para usar el agente, necesitas una API key de OpenAI.")
    print("Puedes obtenerla en: https://platform.openai.com/api-keys")
    
    api_key = input("Ingresa tu API key de OpenAI (o presiona Enter para configurar después): ").strip()
    
    if api_key:
        # Actualizar archivo .env
        with open(env_file, 'r') as f:
            content = f.read()
        
        content = content.replace("your_openai_api_key_here", api_key)
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ API key configurada en .env")
    else:
        print("⚠️  Recuerda configurar tu API key en el archivo .env antes de usar el agente")
    
    return True

def create_directories():
    """Crear directorios necesarios"""
    print("\n📁 Creando directorios...")
    
    directories = ["models", "logs", "reports", "examples"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directorio {directory}/ creado")
    
    return True

def test_installation():
    """Probar la instalación"""
    print("\n🧪 Probando instalación...")
    
    try:
        # Importar módulos principales
        import pandas as pd
        import numpy as np
        from statsbombpy import sb
        from mplsoccer import Sbopen
        
        print("✅ Módulos principales importados correctamente")
        
        # Probar StatsBomb
        competitions = sb.competitions()
        if not competitions.empty:
            print(f"✅ Conexión a StatsBomb exitosa ({len(competitions)} competiciones)")
        else:
            print("⚠️  No se pudieron obtener competiciones de StatsBomb")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Advertencia durante la prueba: {e}")
        return True

def show_next_steps():
    """Mostrar próximos pasos"""
    print("""
🎉 ¡Instalación completada!

📋 Próximos pasos:
1. Si no configuraste tu API key, edita el archivo .env
2. Ejecuta el ejemplo básico: python examples/basic_usage.py
3. Abre el notebook demo: jupyter notebook notebooks/soccer_agent_demo.ipynb
4. Usa la CLI: python -m src.cli analyze "Nombre del Jugador"

📚 Documentación:
- README.md: Guía completa del proyecto
- notebooks/soccer_agent_demo.ipynb: Demo interactivo
- examples/: Ejemplos de uso

🔧 Comandos útiles:
- soccer-agent analyze "Jugador"     # Análisis general
- soccer-agent predict "Jugador"     # Predicción de potencial
- soccer-agent chat "Pregunta"       # Chat interactivo
- soccer-agent competitions          # Ver competiciones disponibles

¡Disfruta analizando fútbol con IA! ⚽🤖
    """)

def main():
    """Función principal"""
    print_banner()
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Instalar dependencias
    if not install_dependencies():
        sys.exit(1)
    
    # Configurar entorno
    if not setup_environment():
        sys.exit(1)
    
    # Crear directorios
    if not create_directories():
        sys.exit(1)
    
    # Probar instalación
    if not test_installation():
        print("⚠️  La instalación se completó pero hay algunos problemas")
        print("   Revisa los errores anteriores")
    
    # Mostrar próximos pasos
    show_next_steps()

if __name__ == "__main__":
    main() 