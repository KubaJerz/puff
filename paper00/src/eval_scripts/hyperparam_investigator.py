"""
Hyperparameter Investigator

Analyzes hyperparameter sweep results and generates comprehensive reports
of model performance, ranking, and hyperparameter impact analysis.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import statistics
import toml


class HyperparameterInvestigator:
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.models_data = []
        self.top_models = {}
        self.bottom_models = {}
        self.hyperparameter_analysis = {}
        
    def validate_directory(self) -> bool:
        """Validate that the experiment directory exists and has required structure."""
        if not self.experiment_dir.exists():
            print(f"Error: Directory '{self.experiment_dir}' does not exist.")
            return False
            
        # Find the main experiment TOML file
        toml_files = list(self.experiment_dir.glob("*.toml"))
        if not toml_files:
            print(f"Error: No .toml files found in '{self.experiment_dir}'.")
            return False
            
        self.main_toml_file = toml_files[0]  # Use the first TOML file found
        print(f"Using main experiment file: {self.main_toml_file.name}")
        
        return True
        
    def load_main_experiment_config(self) -> Dict[str, Any]:
        """Load the main experiment configuration file."""
        try:
            with open(self.main_toml_file, 'r') as f:
                config = toml.load(f)
            return config
        except Exception as e:
            print(f"Error loading main experiment config: {e}")
            return {}
            
    def load_model_data(self) -> List[Dict[str, Any]]:
        """Load model data from all subdirectories."""
        models_data = []

        main_config = self.load_main_experiment_config()
        search_space = main_config.get('sweep', {}).get('search_space', {})

        if not search_space:
            print("Warning: No search_space found in main experiment config.\nThis is ment to analyze a hyperparameter search. If just eval models use diffrent script.")
            return models_data
        
        # Get all subdirectories
        subdirs = [d for d in self.experiment_dir.iterdir() if d.is_dir()]
        
        for subdir in subdirs:
            try:
                # Look for TOML file in subdirectory
                toml_files = list(subdir.glob("*.toml"))
                if not toml_files:
                    continue
                    
                # Load hyperparameters from TOML
                with open(toml_files[0], 'r') as f:
                    subdir_config = toml.load(f)

                try:
                    extracted_hyperparams = self._extract_hyperparams_from_config(subdir_config, search_space)
                except Exception as e:
                    print(f"Error: in the '_extract_hyperparams_from_config' for sub dir {subdir}: {e}")
                # Load performance metrics from best_dev_metrics.json
                metrics_file = subdir / "best_dev_metrics.json"
                if not metrics_file.exists():
                    print(f"Warning: Missing best_dev_metrics.json in {subdir.name}")
                    continue
                    
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Extract dev loss and f1 score
                dev_loss = min(metrics.get('devlossi', [float('inf'), float('inf')])) #return min dev loss
                dev_f1 = max(metrics.get('devf1i', [0.0, 0.0])) #rturn max devf1 
                
                models_data.append({
                    'subdirectory': subdir.name,
                    'hyperparameters': extracted_hyperparams,
                    'dev_loss': dev_loss,
                    'dev_f1': dev_f1,
                    'performance_score': dev_f1  # Using F1 as primary performance metric
                })
                
            except Exception as e:
                print(f"Warning: Error processing {subdir.name}: {e}")
                continue
                
        return models_data
    
    def _extract_hyperparams_from_config(self, config: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hyperparameters from config based on search space definition."""
        extracted = {}
        
        for param_name, param_values in search_space.items():
            # Map common parameter names to their config paths
            param_path = self._get_config_path_for_param(param_name)
            if param_path is  None:
                raise ValueError(f"ERROR: the param 'path' is not specified for '{param_name}' you need to specify a patth for it in 'path_mappings' so we can find it in the sub dir toml")
                
            
            # Extract value from config using the path
            value = self._get_nested_value(config, param_path)
            
            if value is not None:
                extracted[param_name] = value
            else:
                raise ValueError(f"ERROR: Could not find value for parameter '{param_name}' at path '{param_path}'")
                
        return extracted
    
    def _get_config_path_for_param(self, param_name: str) -> List[str]:
        """Map parameter names to their config file paths."""
        # Common mappings for hyperparameters
        path_mappings = {
            'lr': ['optimizer', 'optimizer_params', 'lr'],
            'learning_rate': ['optimizer', 'optimizer_params', 'lr'],
            'batch_size': ['data', 'batch_size'],
            'epochs': ['epochs'],
            'hidden_dim': ['model', 'model_hyperparams', 'hidden_dim'],
            'hidden_size': ['model', 'model_hyperparams', 'hidden_size'],
            'dropout': ['model', 'model_hyperparams', 'dropout'],
            'weight_decay': ['optimizer', 'optimizer_params', 'weight_decay'],
            'in_channels': ['model', 'model_hyperparams', 'in_channels'],
        }
        
        # Return the mapped path or assume it's a top-level parameter
        return path_mappings.get(param_name, None)
    
    def _get_nested_value(self, config: Dict[str, Any], path: List[str]) -> Any:
        """Get a nested value from config using a path list."""
        current = config
        
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
                
        return current
        
    def rank_models(self) -> None:
        """Rank models by performance and select top/bottom models."""
        if not self.models_data:
            print("Error: No model data loaded.")
            return
            
        # Sort by performance score (F1 score) in descending order
        self.models_data.sort(key=lambda x: x['performance_score'], reverse=True)
        
        num_models = len(self.models_data)
        print(f"Found {num_models} models to analyze.")
        
        if num_models >= 10:
            # Select top 5 and bottom 5
            top_models_data = self.models_data[:5]
            bottom_models_data = self.models_data[-5:]
        else:
            # Split into top 50% and bottom 50%
            split_point = int(num_models * 0.5)
            top_models_data = self.models_data[:split_point]
            bottom_models_data = self.models_data[split_point:]
            
        # Create dictionaries with ranking positions
        self.top_models = {
            i + 1: {
                'hyperparameters': model['hyperparameters'],
                'subdirectory_name': model['subdirectory'],
                'performance_score': model['performance_score'],
                'dev_loss': model['dev_loss'],
                'dev_f1': model['dev_f1']
            }
            for i, model in enumerate(top_models_data)
        }
        
        self.bottom_models = {
            -(i + 1): {
                'hyperparameters': model['hyperparameters'],
                'subdirectory_name': model['subdirectory'],
                'performance_score': model['performance_score'],
                'dev_loss': model['dev_loss'],
                'dev_f1': model['dev_f1']
            }
            for i, model in enumerate(reversed(bottom_models_data))
        }
        
    def analyze_hyperparameters(self) -> None:
        """Analyze the impact of different hyperparameter values."""
        # Load main config to get search space
        main_config = self.load_main_experiment_config()
        search_space = main_config.get('sweep', {}).get('search_space', {})
        
        if not search_space:
            print("Warning: No search_space found in main experiment config")
            return
        
        # Collect all hyperparameter values and their associated performances
        hyperparam_values = defaultdict(lambda: defaultdict(list))
        
        for model in self.models_data:
            hyperparams = model['hyperparameters']
            performance = model['performance_score']
            
            # Analyze only the parameters defined in search_space
            for param_name in search_space.keys():
                if param_name in hyperparams:
                    param_value = hyperparams[param_name]
                    hyperparam_values[param_name][str(param_value)].append(performance)
        
        # Calculate averages and sort
        self.hyperparameter_analysis = {}
        for param_name, values_dict in hyperparam_values.items():
            param_analysis = []
            for value, performances in values_dict.items():
                avg_performance = statistics.mean(performances)
                count = len(performances)
                param_analysis.append({
                    'value': value,
                    'avg_performance': avg_performance,
                    'count': count
                })
            
            # Sort by average performance (descending)
            param_analysis.sort(key=lambda x: x['avg_performance'], reverse=True)
            self.hyperparameter_analysis[param_name] = param_analysis
                
    def get_best_overall_model(self) -> Optional[Dict[str, Any]]:
        """Get the best overall model."""
        if self.models_data:
            return self.models_data[0]
        return None
        
    def get_best_dev_model(self) -> Optional[Dict[str, Any]]:
        """Get the model with the best dev performance."""
        if not self.models_data:
            return None
            
        # Sort by dev_f1 score
        best_dev = max(self.models_data, key=lambda x: x['dev_f1'])
        return best_dev
        
    def format_hyperparameters(self, hyperparams: Dict[str, Any]) -> str:
        """Format hyperparameters for display."""
        formatted_params = []
        
        # Format the extracted hyperparameters
        for key, value in hyperparams.items():
            formatted_params.append(f"{key}={value}")
                    
        return ", ".join(formatted_params) if formatted_params else "N/A"
        
    def print_console_report(self) -> None:
        """Print comprehensive report to console."""
        print("\n" + "="*80)
        print("HYPERPARAMETER ANALYSIS REPORT")
        print("="*80)
        
        # Best models section
        print("\n📈 BEST MODELS")
        print("-" * 50)
        best_overall = self.get_best_overall_model()
        best_dev = self.get_best_dev_model()
        
        for rank, model_info in self.top_models.items():
            model_name = model_info['subdirectory_name']
            hyperparams = self.format_hyperparameters(model_info['hyperparameters'])
            performance = model_info['performance_score']
            
            marker = ""
            if best_overall and model_name == best_overall['subdirectory']:
                marker += " 🏆 BEST OVERALL"
            if best_dev and model_name == best_dev['subdirectory']:
                marker += " 📊 BEST DEV"
                
            print(f"{rank:2d}. {model_name:15s} | F1: {performance:.4f} | {hyperparams}{marker}")
            
        # Worst models section
        print("\n📉 WORST MODELS")
        print("-" * 50)
        for rank, model_info in self.bottom_models.items():
            model_name = model_info['subdirectory_name']
            hyperparams = self.format_hyperparameters(model_info['hyperparameters'])
            performance = model_info['performance_score']
            
            print(f"{rank:2d}. {model_name:15s} | F1: {performance:.4f} | {hyperparams}")
            
        # Hyperparameter analysis section
        print("\n🔍 HYPERPARAMETER IMPACT ANALYSIS")
        print("-" * 50)
        for param_name, analysis in self.hyperparameter_analysis.items():
            print(f"\n{param_name.upper()}:")
            for item in analysis:
                print(f"  • {item['value']:15s} (avg F1: {item['avg_performance']:.4f}, count: {item['count']:2d})")
                
    def save_markdown_report(self) -> None:
        """Save report to markdown file."""
        output_file = self.experiment_dir / "hyperparameter_analysis.md"
        
        with open(output_file, 'w') as f:
            f.write("# Hyperparameter Analysis Report\n")
            f.write(f"**{self.experiment_dir}**\n\n")
            
            # Best models section
            f.write("## Best Models\n\n")
            f.write("| Rank | Model Name | F1 Score | Hyperparameters | Notes |\n")
            f.write("|------|------------|----------|-----------------|-------|\n")
            
            best_overall = self.get_best_overall_model()
            best_dev = self.get_best_dev_model()
            
            for rank, model_info in self.top_models.items():
                model_name = model_info['subdirectory_name']
                hyperparams = self.format_hyperparameters(model_info['hyperparameters'])
                performance = model_info['performance_score']
                
                notes = []
                if best_overall and model_name == best_overall['subdirectory']:
                    notes.append("🏆 Best Overall")
                if best_dev and model_name == best_dev['subdirectory']:
                    notes.append("📊 Best Dev")
                    
                notes_str = ", ".join(notes)
                f.write(f"| {rank} | {model_name} | {performance:.4f} | {hyperparams} | {notes_str} |\n")
                
            # Worst models section
            f.write("\n## Worst Models\n\n")
            f.write("| Rank | Model Name | F1 Score | Hyperparameters |\n")
            f.write("|------|------------|----------|------------------|\n")
            
            for rank, model_info in self.bottom_models.items():
                model_name = model_info['subdirectory_name']
                hyperparams = self.format_hyperparameters(model_info['hyperparameters'])
                performance = model_info['performance_score']
                
                f.write(f"| {rank} | {model_name} | {performance:.4f} | {hyperparams} |\n")
                
            # Hyperparameter analysis section
            f.write("\n## Hyperparameter Impact Analysis\n\n")
            for param_name, analysis in self.hyperparameter_analysis.items():
                f.write(f"### {param_name.title()}\n\n")
                for item in analysis:
                    f.write(f"- **{item['value']}** (avg F1: {item['avg_performance']:.4f}, count: {item['count']})\n")
                f.write("\n")
                
        print(f"\n✅ Report saved to: {output_file}")
        
    def run_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        print("🔍 Starting hyperparameter analysis...")
        
        # Validate directory structure
        if not self.validate_directory():
            return
            
        # Load model data
        print("📊 Loading model data...")
        self.models_data = self.load_model_data()
        
        if not self.models_data:
            print("❌ No valid model data found.")
            return
            
        # Rank models
        print("🏆 Ranking models...")
        self.rank_models()
        
        # Analyze hyperparameters
        print("🔬 Analyzing hyperparameters...")
        self.analyze_hyperparameters()
        
        # Generate reports
        print("📝 Generating reports...")
        self.print_console_report()
        self.save_markdown_report()
        
        print("\n✅ Analysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter sweep results and generate comprehensive reports"
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to the experiment run directory"
    )
    
    args = parser.parse_args()
    
    # Create and run the investigator
    investigator = HyperparameterInvestigator(args.experiment_dir)
    investigator.run_analysis()


if __name__ == "__main__":
    main()