#!/usr/bin/env python3
"""
Ejemplo básico de uso del Soccer Agent
"""

import sys
import os
sys.path.append('..')

from src.soccer_agent import SoccerAgent
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def main():
    """Ejemplo básico de uso"""
    print("⚽ Soccer Agent - Ejemplo Básico")
    print("=" * 50)
    
    # Verificar API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: No se encontró OPENAI_API_KEY")
        print("Por favor, crea un archivo .env con tu API key de OpenAI")
        return
    
    # Inicializar agente
    print("🤖 Inicializando agente...")
    agent = SoccerAgent()
    
    # Jugador de ejemplo
    player_name = "Dusan Tadic"
    
    print(f"\n🔍 Analizando jugador: {player_name}")
    
    # 1. Análisis general
    print("\n1️⃣ Análisis General de Rendimiento:")
    print("-" * 40)
    general_analysis = agent.analyze_player_general_performance(player_name)
    
    if 'error' not in general_analysis:
        print("✅ Análisis general completado")
        print(f"📊 Métricas encontradas: {len(general_analysis.get('metrics_summary', {}))}")
    else:
        print(f"❌ Error en análisis general: {general_analysis['error']}")
    
    # 2. Predicción de potencial
    print("\n2️⃣ Predicción de Potencial:")
    print("-" * 40)
    potential_analysis = agent.predict_player_potential(player_name, years_ahead=3)
    
    if 'error' not in potential_analysis:
        print("✅ Predicción completada")
        print(f"📈 Rendimiento actual: {potential_analysis['current_performance']:.3f}")
        print(f"🎯 Confianza: {potential_analysis['confidence']:.2%}")
    else:
        print(f"❌ Error en predicción: {potential_analysis['error']}")
    
    # 3. Chat interactivo
    print("\n3️⃣ Chat Interactivo:")
    print("-" * 40)
    chat_response = agent.chat_with_agent("¿Cuáles son las principales fortalezas de este jugador?")
    print("✅ Chat completado")
    print(f"💬 Respuesta: {chat_response[:100]}...")
    
    # 4. Explorar datos disponibles
    print("\n4️⃣ Exploración de Datos:")
    print("-" * 40)
    competitions = agent.get_available_competitions()
    print(f"🏆 Competiciones disponibles: {len(competitions)}")
    
    if competitions:
        # Mostrar primera competición
        first_comp = competitions[0]
        print(f"📋 Ejemplo: {first_comp.get('competition_name', 'N/A')} ({first_comp.get('country_name', 'N/A')})")
        
        # Obtener partidos de la primera competición
        matches = agent.get_available_matches(
            competition_id=first_comp['competition_id'], 
            season_id=first_comp.get('season_id', 3)
        )
        print(f"⚽ Partidos disponibles: {len(matches)}")
    
    print("\n✅ Ejemplo completado exitosamente!")
    print("\n💡 Próximos pasos:")
    print("   • Ejecuta el notebook demo: jupyter notebook notebooks/soccer_agent_demo.ipynb")
    print("   • Usa la CLI: python -m src.cli analyze 'Nombre del Jugador'")
    print("   • Personaliza el análisis según tus necesidades")

if __name__ == "__main__":
    main() 