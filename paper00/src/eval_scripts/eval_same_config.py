"""
eval_same_config.py - Model Evaluation Script

Evaluates one or more models with identical hyperparameters, generates visualizations,
and makes LaTeX report compiled to PDF.
"""

import os
import sys
import json
import toml
import argparse
import importlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
import pandas as pd


class ColorTheme:
    """Color theme management for consistent visualization styling."""
    
    THEMES = {
        'default': {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'background': '#F8F9FA',
            'text': '#212529',
            'grid': '#E9ECEF',
            'success': '#28A745',
            'warning': '#FFC107',
            'error': '#DC3545'
        },
        'academic': {
            'primary': '#1F4E79',
            'secondary': '#8B4513',
            'accent': '#B22222',
            'background': '#FFFFFF',
            'text': '#000000',
            'grid': '#D3D3D3',
            'success': '#006400',
            'warning': '#FF8C00',
            'error': '#8B0000'
        },
        'modern': {
            'primary': '#6366F1',
            'secondary': '#EC4899',
            'accent': '#10B981',
            'background': '#F9FAFB',
            'text': '#111827',
            'grid': '#E5E7EB',
            'success': '#059669',
            'warning': '#F59E0B',
            'error': '#EF4444'
        }
    }
    
    def __init__(self, theme_name: str = 'default'):
        self.theme = self.THEMES.get(theme_name, self.THEMES['default'])
        self.apply_theme()
    
    def apply_theme(self):
        """Apply theme to matplotlib."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['axes.facecolor'] = self.theme['background']
        plt.rcParams['figure.facecolor'] = self.theme['background']
        plt.rcParams['text.color'] = self.theme['text']
        plt.rcParams['axes.labelcolor'] = self.theme['text']
        plt.rcParams['xtick.color'] = self.theme['text']
        plt.rcParams['ytick.color'] = self.theme['text']
        plt.rcParams['grid.color'] = self.theme['grid']
        plt.rcParams['axes.edgecolor'] = self.theme['grid']


class ModelEvaluator:
    """Main evaluation class for model analysis and reporting."""
    
    def __init__(self, model_dir: str, model_type: str, model_outputs_logits:bool = False, use_gpu: bool = True, color_theme: str = 'default'):
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.model_outputs_logits = model_outputs_logits
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.theme = ColorTheme(color_theme)
        
        # Load configuration
        self.config = self._load_config()
        self.model_subdirs = self._find_model_subdirs()
        
        # Create output directories
        self.figures_dir = self.model_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        print(f"Found {len(self.model_subdirs)} model subdirectories")
        print(f"Using device: {self.device}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.toml."""
        config_path = self.model_dir / f'{os.path.basename(self.model_dir)}.toml'
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = toml.load(f)
        
        return config
    
    def _find_model_subdirs(self) -> List[Path]:
        """Find all model subdirectories containing required files."""
        subdirs = []
        for item in self.model_dir.iterdir():
            if item.is_dir() and item.name != 'figures':
                metrics_file = item / 'best_dev_metrics.json'
                model_file = item / 'best_dev_model.pt'
                if metrics_file.exists() and model_file.exists():
                    subdirs.append(item)
                else:
                    print(f"Warning {item} does not have: 'best_dev_metrics.json' or 'best_dev_model.pt'")
        
        if not subdirs:
            raise ValueError("No valid model subdirectories found")
        
        return sorted(subdirs)
    
    def load_model(self, model_subdir: Path):
        """Load a model from the specified subdirectory."""
        # Dynamic model loading
        model_path = self.config['static']['model']['model_path']
        # model_class_name = self.config['static']['model']['model_class_name']
        model_hyperparams = (self.config['static']['model']).copy()
        del model_hyperparams['model_path']
        
        # Import model module
        module_dir = os.path.dirname(model_path)
        module_name = os.path.basename(model_path).replace('.py', '')
        sys.path.insert(0, module_dir)
        
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, 'Model')
            
            # Instantiate model
            model = model_class(**model_hyperparams)
            
            # Load state dict
            model_file = model_subdir / 'best_dev_model.pt'
            state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"Error loading model from {model_subdir}: {e}")
            return None
    
    def load_metrics(self, model_subdir: Path) -> Dict[str, List[float]]:
        """Load metrics from best_dev_metrics.json."""
        metrics_file = model_subdir / 'best_dev_metrics.json'
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def load_dev_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load development data."""
        dev_dir_path = self.config['static']['data']['dev_path']
        dev_path = Path(dev_dir_path) / "dev.pt"
        data = torch.load(dev_path, map_location=self.device, weights_only=True)
        return data  # Should be (X, y) tuple
    
    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics across all models."""
        metrics_data = []
        
        for model_subdir in self.model_subdirs:
            metrics = self.load_metrics(model_subdir)
            model_name = model_subdir.name
            
            best_dev_f1 = max(metrics['devf1i']) if metrics['devf1i'] else 0.0
            best_dev_loss = min(metrics['devlossi']) if metrics['devlossi'] else float('inf')
            
            metrics_data.append({
                'model_name': model_name,
                'best_dev_f1': best_dev_f1,
                'best_dev_loss': best_dev_loss,
                'metrics': metrics
            })
        
        return {
            'metrics_data': metrics_data,
            'is_single_model': len(metrics_data) == 1
        }
    
    def create_performance_plots(self, analysis: Dict[str, Any]):
        """Create performance visualization plots."""
        metrics_data = analysis['metrics_data']
        is_single = analysis['is_single_model']
        
        if is_single:
            # Single model: display metrics as text
            model_data = metrics_data[0]
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            ax.text(0.5, 0.7, f"Best Dev F1: {model_data['best_dev_f1']:.4f}", 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.text(0.5, 0.3, f"Best Dev Loss: {model_data['best_dev_loss']:.4f}", 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Model Performance Summary', fontsize=18, pad=20)
            
        else:
            # Multiple models: create whisker plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            f1_scores = [data['best_dev_f1'] for data in metrics_data]
            losses = [data['best_dev_loss'] for data in metrics_data]
            
            # F1 scores boxplot
            ax1.boxplot(f1_scores, patch_artist=True, 
                       boxprops=dict(facecolor=self.theme.theme['primary'], alpha=0.7),
                       medianprops=dict(color=self.theme.theme['accent'], linewidth=2))
            ax1.set_ylabel('F1 Score')
            ax1.set_title('Distribution of Best Dev F1 Scores')
            ax1.grid(True, alpha=0.3)
            
            # Loss boxplot
            ax2.boxplot(losses, patch_artist=True,
                       boxprops=dict(facecolor=self.theme.theme['secondary'], alpha=0.7),
                       medianprops=dict(color=self.theme.theme['accent'], linewidth=2))
            ax2.set_ylabel('Loss')
            ax2.set_title('Distribution of Best Dev Losses')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_training_curves(self, analysis: Dict[str, Any], data_type: str):
        """Create training curves visualization."""
        metrics_data = analysis['metrics_data']
        is_single = analysis['is_single_model']
        
        # Prepare data
        all_train = []
        all_dev = []
        
        max_epochs = 0
        
        for data in metrics_data:
            metrics = data['metrics']
            all_train.append(metrics[f'{data_type}i'])
            all_dev.append(metrics[f'dev{data_type}i'])
            
            max_epochs = max(max_epochs, len(metrics[f'{data_type}i']), len(metrics[f'dev{data_type}i']))
        
        # Pad sequences to same length
        def pad_sequence(seq, target_length):
            if len(seq) < target_length:
                return seq + [seq[-1]] * (target_length - len(seq))
            return seq[:target_length]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        epochs = list(range(1, max_epochs + 1))
        
        if is_single:
            # Single model plots
            data = metrics_data[0]
            metrics = data['metrics']
            
            ax1.plot(epochs[:len(metrics[f'{data_type}i'])], metrics[f'{data_type}i'], color=self.theme.theme['primary'], linewidth=2, label=f'Training {data_type}')
            ax2.plot(epochs[:len(metrics[f'dev{data_type}i'])], metrics[f'dev{data_type}i'], color=self.theme.theme['primary'], linewidth=2, label=f'Validation {data_type}')
            
        else:
            # Multiple models: individual curves + average
            # colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))
            
            # Plot individual curves with transparency
            color = 'grey'
            for i, data in enumerate(metrics_data):
                metrics = data['metrics']
                
                ax1.plot(epochs[:len(metrics[f'{data_type}i'])], metrics[f'{data_type}i'], color=color, alpha=0.6, linewidth=1)
                ax2.plot(epochs[:len(metrics[f'dev{data_type}i'])], metrics[f'dev{data_type}i'], color=color, alpha=0.6, linewidth=1)

            
            # Calculate and plot averages
            avg_train = np.mean([pad_sequence(seq, max_epochs) for seq in all_train], axis=0)
            avg_dev = np.mean([pad_sequence(seq, max_epochs) for seq in all_dev], axis=0)
            
            ax1.plot(epochs, avg_train, color=self.theme.theme['primary'], linewidth=1.5, label=f'Average Training {data_type}')
            ax2.plot(epochs, avg_dev, color=self.theme.theme['secondary'], linewidth=1.5, label=f'Average Validation {data_type}')
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)


        
        # Styling
        for ax, title in zip([ax1, ax2], [f'Training {data_type}', f'Validation {data_type}']):
            ax.set_xlabel('Epoch')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'training_{data_type}_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def find_best_models(self, analysis: Dict[str, Any]) -> Tuple[Path, Path]:
        """Find models with best validation loss and F1 score."""
        metrics_data = analysis['metrics_data']
        
        best_loss_model = min(metrics_data, key=lambda x: x['best_dev_loss'])
        best_f1_model = max(metrics_data, key=lambda x: x['best_dev_f1'])
        
        best_loss_path = self.model_dir / best_loss_model['model_name']
        best_f1_path = self.model_dir / best_f1_model['model_name']
        
        return best_loss_path, best_f1_path
    
    def create_confusion_matrices(self, analysis: Dict[str, Any]):
        """Create confusion matrices for best models."""
        best_loss_path, best_f1_path = self.find_best_models(analysis)
        
        # Load dev data
        X, y = self.load_dev_data()
        X, y = X.to(self.device), y.to(self.device)
        
        # Determine number of classes
        num_classes = len(torch.unique(y))
        print(f'Num classes in y is: {num_classes}')
    

        # Process each model
        for model_path, model_name in [(best_loss_path, 'best_loss'), (best_f1_path, 'best_f1')]:
            model = self.load_model(model_path)
            if model is None:
                continue
            
            y_pred = self.run_model_based_on_type(model, X, self.model_type, model_outputs_logits=self.model_outputs_logits)
            y_true = y.cpu().flatten().numpy()
            
            # Create confusion matrices
            cm = confusion_matrix(y_true, y_pred)

            # Three versions: raw, normalized on true, normalized on pred
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Raw values
            disp1 = ConfusionMatrixDisplay(cm, display_labels=range(num_classes))
            disp1.plot(ax=axes[0], cmap='Blues', colorbar=False)
            axes[0].set_title('Raw Counts')
            axes[0].grid(False)

            
            # Normalized on true (recall perspective)
            cm_norm_true = confusion_matrix(y_true, y_pred, normalize='true')
            disp2 = ConfusionMatrixDisplay(cm_norm_true, display_labels=range(num_classes))
            disp2.plot(ax=axes[1], cmap='Blues', colorbar=False)
            axes[1].set_title('Normalized on True (Recall)')
            axes[1].grid(False)
            
            # Normalized on pred (precision perspective)
            cm_norm_pred = confusion_matrix(y_true, y_pred, normalize='pred')
            disp3 = ConfusionMatrixDisplay(cm_norm_pred, display_labels=range(num_classes))
            disp3.plot(ax=axes[2], cmap='Blues', colorbar=False)
            axes[2].set_title('Normalized on Pred (Precision)')
            axes[2].grid(False)
            

            plt.suptitle(f'Confusion Matrices - {model_name.replace("_", " ").title()} Model', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_calibration_curves(self, analysis: Dict[str, Any]):
        """Create calibration curves for best models."""
        best_loss_path, best_f1_path = self.find_best_models(analysis)
        
        # Load dev data
        X, y = self.load_dev_data()
        X, y = X.to(self.device), y.to(self.device)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Process each model
        for i, (model_path, model_name) in enumerate([(best_loss_path, 'Best Loss'), (best_f1_path, 'Best F1')]):
            model = self.load_model(model_path)
            if model is None:
                continue

            prob_pos = self.run_model_based_on_type(model, X, self.model_type, model_outputs_logits=self.model_outputs_logits)
            y_binary = y.cpu().flatten().numpy()

            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, prob_pos, n_bins=10
                )
                
                axes[i].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                axes[i].plot(mean_predicted_value, fraction_of_positives, 'o-', 
                           color=self.theme.theme['primary'], label='Model Calibration')
                axes[i].set_xlabel('Mean Predicted Probability')
                axes[i].set_ylabel('Fraction of Positives')
                axes[i].set_title(f'{model_name} Model Calibration')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error creating calibration curve for {model_name}: {e}")
                axes[i].text(0.5, 0.5, f'Calibration curve\nnot available\n({str(e)})', 
                           ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_model_based_on_type(self, model, X, type, model_outputs_logits):
        # Get predictions
        with torch.no_grad():
            outputs = model(X)

        if type == 'binary_seg':
            predictions = (outputs > 0.5).long().squeeze().flatten()
            if model_outputs_logits:
                predictions = torch.nn.functional.sigmoid(predictions)
        else:
            raise ValueError(f"For type: {type} we dont know how to format outputs. Please add in run_model_based_on_type")
        #     if outputs.dim() > 1 and outputs.size(1) > 1:
        #         predictions = torsch.argmax(outputs, dim=1)
        #     else:

        y_pred = predictions.cpu().numpy()
        return y_pred
            
    
    def escape_latex(self, text):
        """Escape special LaTeX characters in text."""
        if text is None:
            return ""
        
        text = str(text)
        # Replace special LaTeX characters
        replacements = {
            '\\': r'\textbackslash{}',
            '{': r'\{',
            '}': r'\}',
            '$': r'\$',
            '&': r'\&',
            '%': r'\%',
            '#': r'\#',
            '^': r'\textasciicircum{}',
            '_': r'\_',
            '~': r'\textasciitilde{}',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def generate_latex_report(self, analysis: Dict[str, Any]) -> str:
        """Generate LaTeX report content."""
        metrics_data = sorted(analysis['metrics_data'], key=lambda item: item['best_dev_loss'] )
        is_single = analysis['is_single_model']
        
        # Get model configuration
        model_config = self.config
        
        latex_content = f"""\\documentclass{{article}}
\\usepackage{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{xcolor}}
\\usepackage{{float}}
\\usepackage{{subcaption}}
\\usepackage{{array}}

\\geometry{{margin=1in}}

\\title{{Model Evaluation Report}}
\\author{{Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Model Configuration and Hyperparameters}}

The following models were evaluated with identical hyperparameters:

\\begin{{itemize}}
"""
        for data in sorted(metrics_data, key=lambda x: x['model_name']):
            latex_content += f"\\item {data['model_name']}\n"
        
        latex_content += "\\end{itemize}\n\n"
        
        # Hyperparameters table
        latex_content += "\\subsection{Hyperparameters}\n\n"
        latex_content += "\\begin{table}[H]\n\\centering\n"
        latex_content += "\\begin{tabular}{ll}\n\\toprule\n"
        latex_content += "Parameter & Value \\\\\n\\midrule\n"
        
        for sub_dict in model_config.items():
            for param, value in sub_dict[1].items():
                param, value = self.escape_latex(param), self.escape_latex(value)
                if len(str(value)) > 80:
                    latex_content += f"{param} & \\parbox{{10cm}}{{\\texttt{{{value}}}}} \\\\\n"
                else:
                    latex_content += f"{param} & {value} \\\\\n"
        
        latex_content += "\\bottomrule\n\\end{tabular}\n"
        latex_content += "\\caption{Model Hyperparameters}\n\\end{table}\n\n"
        
        # Performance summary
        latex_content += "\\section{Performance Summary}\n\n"
        
        if is_single:
            data = metrics_data[0]
            latex_content += f"Best development F1 score: {data['best_dev_f1']:.4f}\n\n"
            latex_content += f"Best development loss: {data['best_dev_loss']:.4f}\n\n"
        else:
            latex_content += "\\begin{table}[H]\n\\centering\n"
            latex_content += "\\begin{tabular}{lcc}\n\\toprule\n"
            latex_content += "Model & Best Dev F1 & Best Dev Loss \\\\\n\\midrule\n"
            
            for data in metrics_data:
                latex_content += f"{data['model_name']} & {data['best_dev_f1']:.4f} & {data['best_dev_loss']:.4f} \\\\\n"
            
            latex_content += "\\bottomrule\n\\end{tabular}\n"
            latex_content += "\\caption{Performance Summary Across All Models}\n\\end{table}\n\n"
        
        latex_content += "\\newpage\n"

        latex_content += "\\begin{figure}[H]\n\\centering\n"
        latex_content += "\\includegraphics[width=\\textwidth]{figures/performance_summary.png}\n"
        latex_content += "\\caption{Performance Summary Visualization}\n\\end{figure}\n\n"
        
        # Training curves
        latex_content += "\\section{Training Curves}\n\n"
        latex_content += "The following plots show the training and dev curves for loss.\n\n"
        latex_content += "\\newpage\n"

        
        latex_content += "\\begin{figure}[H]\n\\centering\n"
        latex_content += "\\includegraphics[width=\\textwidth]{figures/training_loss_curves.png}\n"
        latex_content += "\\caption{Training and Dev Curves}\n\\end{figure}\n\n"

        latex_content += "\\newpage\n"
        latex_content += "The following plots show the training and dev curves for F1 score.\n\n"
        latex_content += "\\begin{figure}[H]\n\\centering\n"
        latex_content += "\\includegraphics[width=\\textwidth]{figures/training_f1_curves.png}\n"
        latex_content += "\\caption{Training and Dev Curves}\n\\end{figure}\n\n"
        
        # Confusion matrices
        latex_content += "\\newpage\n"
        latex_content += "\\section{Confusion Matrix Analysis}\n\n"
        latex_content += "Confusion matrices are shown for the models with the best validation loss and F1 score.\n\n"
        
        best_loss_path, best_f1_path = self.find_best_models(analysis)
        
        latex_content += "\\begin{figure}[H]\n\\centering\n"
        latex_content += "\\includegraphics[width=\\textwidth]{figures/confusion_matrix_best_loss.png}\n"
        latex_content += f"\\caption{{Confusion Matrix for Best Loss Model ({best_loss_path.name})}}"+"\n\\end{figure}\n\n"
        
        if best_f1_path != best_loss_path:
            latex_content += "\\begin{figure}[H]\n\\centering\n"
            latex_content += "\\includegraphics[width=\\textwidth]{figures/confusion_matrix_best_f1.png}\n"
            latex_content += f"\\caption{{Confusion Matrix for Best F1 Model ({best_f1_path.name})}}"+"\n\\end{figure}\n\n"
        else:
            latex_content += "\\textbf{Best Loss model is the same and Best F1 model so we just show one.}"
        
        # Calibration analysis
        latex_content += "\\newpage\n"
        latex_content += "\\section{Calibration Analysis}\n\n"
        latex_content += "Calibration curves show how well the predicted probabilities match the actual outcomes.\n\n"
        
        latex_content += "\\begin{figure}[H]\n\\centering\n"
        latex_content += "\\includegraphics[width=\\textwidth]{figures/calibration_curves.png}\n"
        latex_content += "\\caption{Model Calibration Curves}\n\\end{figure}\n\n"
        
        latex_content += "\\end{document}"
        
        return latex_content
    
    def compile_pdf(self, latex_content: str) -> bool:
        """Compile LaTeX to PDF."""
        tex_file = self.model_dir / 'evaluation_report.tex'
        pdf_file = self.model_dir / 'evaluation_report.pdf'
        
        # Write LaTeX content
        with open(tex_file, 'w') as f:
            f.write(latex_content)
        
        # Compile to PDF
        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', str(tex_file)],
                cwd=str(self.model_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"PDF report generated successfully: {pdf_file}")
                return True
            else:
                print(f"LaTeX compilation failed with return code: {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("pdflatex not found. Please install LaTeX to generate PDF reports.")
            print(f"LaTeX source saved to: {tex_file}")
            return False
    
    def run_evaluation(self):
        """Run the complete evaluation pipeline."""
        print("Starting model evaluation...")
        
        # Step 1: Analyze performance metrics
        print("Analyzing performance metrics...")
        analysis = self.analyze_performance_metrics()
        
        # Step 2: Create visualizations
        print("Creating performance plots...")
        self.create_performance_plots(analysis)
        
        print("Creating training curves...")
        self.create_training_curves(analysis, data_type='loss')

        print("Creating training curves...")
        self.create_training_curves(analysis, data_type='f1')
        
        print("Creating confusion matrices...")
        self.create_confusion_matrices(analysis)
        
        print("Creating calibration curves...")
        self.create_calibration_curves(analysis)
        
        # Step 3: Generate report
        print("Generating LaTeX report...")
        latex_content = self.generate_latex_report(analysis)
        
        # Step 4: Compile PDF
        print("Compiling PDF...")
        self.compile_pdf(latex_content)
        
        print("Evaluation complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate models with identical hyperparameters')
    parser.add_argument('model_directory', help='Path to directory containing model subdirectories and config.toml')
    parser.add_argument('-t', '--model_type', help='Type of model to format output properly options: {binary_seg}')
    parser.add_argument('--model_outputs_logits', action='store_true', help='If the mode outputs logits and we need to sigmoid or softmax')
    parser.add_argument('--no-gpu', action='store_true', help='Force CPU-only evaluation')
    parser.add_argument('--colors', default='default', help='Color theme for visualizations')
    
    args = parser.parse_args()
    print(args.model_type)
    
    try:
        evaluator = ModelEvaluator(
            model_dir=args.model_directory,
            model_type=args.model_type,
            model_outputs_logits=args.model_outputs_logits,
            use_gpu=not args.no_gpu,
            color_theme=args.colors
        )
        evaluator.run_evaluation()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()