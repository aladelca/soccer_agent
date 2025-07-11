"""
Command Line Interface for Soccer Agent
"""

import argparse
import json
import sys
from typing import Optional
from .soccer_agent import SoccerAgent


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Soccer Agent - Análisis inteligente de futbolistas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  soccer-agent analyze "Dusan Tadic"
  soccer-agent match "Dusan Tadic" 7532
  soccer-agent predict "Dusan Tadic" --years 5
  soccer-agent chat "¿Cuáles son las fortalezas de este jugador?"
  soccer-agent report "Dusan Tadic" --match-id 7532
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando: analyze
    analyze_parser = subparsers.add_parser('analyze', help='Análisis general de rendimiento')
    analyze_parser.add_argument('player_name', help='Nombre del jugador')
    analyze_parser.add_argument('--output', '-o', help='Archivo de salida (JSON)')
    
    # Comando: match
    match_parser = subparsers.add_parser('match', help='Análisis de rendimiento en partido específico')
    match_parser.add_argument('player_name', help='Nombre del jugador')
    match_parser.add_argument('match_id', type=int, help='ID del partido')
    match_parser.add_argument('--output', '-o', help='Archivo de salida (JSON)')
    
    # Comando: predict
    predict_parser = subparsers.add_parser('predict', help='Predicción de potencial')
    predict_parser.add_argument('player_name', help='Nombre del jugador')
    predict_parser.add_argument('--years', '-y', type=int, default=3, help='Años hacia adelante (default: 3)')
    predict_parser.add_argument('--output', '-o', help='Archivo de salida (JSON)')
    
    # Comando: chat
    chat_parser = subparsers.add_parser('chat', help='Chat interactivo con el agente')
    chat_parser.add_argument('message', help='Mensaje para el agente')
    
    # Comando: report
    report_parser = subparsers.add_parser('report', help='Reporte comprehensivo')
    report_parser.add_argument('player_name', help='Nombre del jugador')
    report_parser.add_argument('--match-id', '-m', type=int, help='ID del partido (opcional)')
    report_parser.add_argument('--output', '-o', help='Archivo de salida (JSON)')
    
    # Comando: competitions
    competitions_parser = subparsers.add_parser('competitions', help='Listar competiciones disponibles')
    
    # Comando: matches
    matches_parser = subparsers.add_parser('matches', help='Listar partidos de una competición')
    matches_parser.add_argument('competition_id', type=int, help='ID de la competición')
    matches_parser.add_argument('season_id', type=int, help='ID de la temporada')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Inicializar agente
        print("🤖 Inicializando Soccer Agent...")
        agent = SoccerAgent()
        
        if args.command == 'analyze':
            handle_analyze(agent, args)
        elif args.command == 'match':
            handle_match(agent, args)
        elif args.command == 'predict':
            handle_predict(agent, args)
        elif args.command == 'chat':
            handle_chat(agent, args)
        elif args.command == 'report':
            handle_report(agent, args)
        elif args.command == 'competitions':
            handle_competitions(agent)
        elif args.command == 'matches':
            handle_matches(agent, args)
            
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def handle_analyze(agent: SoccerAgent, args):
    """Manejar comando analyze"""
    print(f"🔍 Analizando rendimiento general de {args.player_name}...")
    
    result = agent.analyze_player_general_performance(args.player_name)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        if 'suggestions' in result:
            print(f"💡 Sugerencias: {result['suggestions']}")
        return
    
    print("\n📊 ANÁLISIS GENERAL:")
    print("=" * 50)
    print(result['analysis'])
    
    if args.output:
        save_json(result, args.output)
        print(f"💾 Resultado guardado en: {args.output}")


def handle_match(agent: SoccerAgent, args):
    """Manejar comando match"""
    print(f"⚽ Analizando rendimiento de {args.player_name} en partido {args.match_id}...")
    
    result = agent.analyze_match_performance(args.player_name, args.match_id)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        if 'suggestions' in result:
            print(f"💡 Sugerencias: {result['suggestions']}")
        return
    
    print("\n🎯 ANÁLISIS DEL PARTIDO:")
    print("=" * 50)
    print(result['analysis'])
    
    if args.output:
        save_json(result, args.output)
        print(f"💾 Resultado guardado en: {args.output}")


def handle_predict(agent: SoccerAgent, args):
    """Manejar comando predict"""
    print(f"🔮 Prediciendo potencial de {args.player_name} ({args.years} años adelante)...")
    
    result = agent.predict_player_potential(args.player_name, args.years)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print("\n🚀 ANÁLISIS DE POTENCIAL:")
    print("=" * 50)
    print(result['analysis'])
    
    print(f"\n📊 Rendimiento actual: {result['current_performance']:.3f}")
    print(f"🎯 Confianza: {result['confidence']:.2%}")
    
    if args.output:
        save_json(result, args.output)
        print(f"💾 Resultado guardado en: {args.output}")


def handle_chat(agent: SoccerAgent, args):
    """Manejar comando chat"""
    print(f"💬 Chat con el agente: {args.message}")
    
    response = agent.chat_with_agent(args.message)
    
    print("\n🤖 Respuesta del agente:")
    print("=" * 50)
    print(response)


def handle_report(agent: SoccerAgent, args):
    """Manejar comando report"""
    print(f"📋 Generando reporte comprehensivo para {args.player_name}...")
    
    result = agent.get_comprehensive_report(args.player_name, args.match_id)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print("\n📄 RESUMEN EJECUTIVO:")
    print("=" * 50)
    print(result['summary'])
    
    if args.output:
        save_json(result, args.output)
        print(f"💾 Reporte guardado en: {args.output}")


def handle_competitions(agent: SoccerAgent):
    """Manejar comando competitions"""
    print("🏆 Competiciones disponibles:")
    
    competitions = agent.get_available_competitions()
    
    if not competitions:
        print("❌ No se pudieron obtener las competiciones")
        return
    
    print(f"\nTotal: {len(competitions)} competiciones")
    print("\nPrimeras 10 competiciones:")
    print("-" * 80)
    print(f"{'ID':<5} {'Nombre':<30} {'País':<20}")
    print("-" * 80)
    
    for comp in competitions[:10]:
        print(f"{comp.get('competition_id', 'N/A'):<5} {comp.get('competition_name', 'N/A')[:29]:<30} {comp.get('country_name', 'N/A')[:19]:<20}")


def handle_matches(agent: SoccerAgent, args):
    """Manejar comando matches"""
    print(f"⚽ Partidos de competición {args.competition_id}, temporada {args.season_id}:")
    
    matches = agent.get_available_matches(args.competition_id, args.season_id)
    
    if not matches:
        print("❌ No se pudieron obtener los partidos")
        return
    
    print(f"\nTotal: {len(matches)} partidos")
    print("\nPrimeros 10 partidos:")
    print("-" * 100)
    print(f"{'ID':<8} {'Local':<25} {'Visitante':<25} {'Resultado':<10}")
    print("-" * 100)
    
    for match in matches[:10]:
        home = match.get('home_team_name', 'N/A')[:24]
        away = match.get('away_team_name', 'N/A')[:24]
        score = f"{match.get('home_score', '?')}-{match.get('away_score', '?')}"
        print(f"{match.get('match_id', 'N/A'):<8} {home:<25} {away:<25} {score:<10}")


def save_json(data: dict, filename: str):
    """Guardar datos en archivo JSON"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Error guardando archivo: {e}")


if __name__ == "__main__":
    main() 