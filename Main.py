import warnings
warnings.filterwarnings('ignore')
from data_collection import PlayerDataCollector
from ml_models import PlayerPerformancePredictor

def main():
    """Main function to run the player performance prediction system."""
    print("=" * 50)
    print("Player Performance Prediction System")
    print("=" * 50)
    
    # Ask user for player information
    search_type = input("Search by name or ID? (name/id): ").lower()
    
    if search_type == 'name':
        player_name = input("Enter player name: ")
        collector = PlayerDataCollector(player_name=player_name)
    elif search_type == 'id':
        try:
            player_id = int(input("Enter player ID: "))
            player_id = float(player_id)
            collector = PlayerDataCollector(player_id=player_id)
        except ValueError:
            print("Invalid ID. Must be a number.")
            return
    else:
        print("Invalid choice. Please choose 'name' or 'id'.")
        return
    
    # Set number of matches
    try:
        max_matches = int(input("Maximum number of matches to analyze (5-15 recommended): "))
        collector.max_matches = max_matches
    except ValueError:
        print("Using default value of 15 matches.")
    
    # Collect player data
    print("\nCollecting player data...")
    if not collector.collect_player_data():
        print("Failed to collect player data. Exiting.")
        return
    
    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    if not collector.calculate_performance_metrics():
        print("Failed to calculate performance metrics. Exiting.")
        return
    
    # Create predictor with the collected metrics
    predictor = PlayerPerformancePredictor(collector.performance_metrics)
    
    # Visualize current performance
    print("\nGenerating performance visualization...")
    predictor.visualize_performance(player_name=collector.full_name)
    
    # Train models
    print("\nTraining prediction models...")
    if predictor.train_models():
        # Predict next performance and check for decline
        print("\nPredicting future performance...")
        predictor.predict_next_performance()
    else:
        print("Failed to train prediction models. Not enough data.")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()