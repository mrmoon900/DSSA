import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.SA_Score import sascorer
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def predict_synth_scores(model, smiles_list, device='cuda'):
    """
    Predicts synthetic accessibility scores and compares with RDKit SA scores
    
    Args:
        model: Trained model
        smiles_list: List of SMILES strings
        device: Computing device
    
    Returns:
        dict: Prediction results and analysis
    """
    model.eval()
    predictions = []
    actual_scores = []
    valid_smiles = []
    failed_smiles = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed_smiles.append((smiles, "Invalid SMILES"))
                continue
                
            # Get actual SA score
            actual_sa = sascorer.calculateScore(mol)
            
            # Get model prediction
            data = smiles_to_graph(smiles).to(device)
            with torch.no_grad():
                pred = model(data).item()
            
            predictions.append(pred)
            actual_scores.append(actual_sa)
            valid_smiles.append(smiles)
            
        except Exception as e:
            failed_smiles.append((smiles, str(e)))
    
    # Calculate metrics
    mse = mean_squared_error(actual_scores, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_scores, predictions)
    
    # Calculate correlation
    correlation = np.corrcoef(actual_scores, predictions)[0,1]
    
    # Error analysis
    errors = np.array(predictions) - np.array(actual_scores)
    
    results = {
        'predictions': predictions,
        'actual_scores': actual_scores,
        'valid_smiles': valid_smiles,
        'failed_smiles': failed_smiles,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation
        },
        'errors': errors
    }
    
    return results

def visualize_predictions(results):
    """
    Creates comprehensive visualization of prediction results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # Scatter plot of predicted vs actual
    ax1.scatter(results['actual_scores'], results['predictions'], alpha=0.5)
    ax1.plot([min(results['actual_scores']), max(results['actual_scores'])], 
             [min(results['actual_scores']), max(results['actual_scores'])], 
             'r--', label='Perfect prediction')
    ax1.set_xlabel('Actual SA Score')
    ax1.set_ylabel('Predicted Score')
    ax1.set_title('Predicted vs Actual Scores')
    ax1.legend()
    
    # Error distribution
    ax2.hist(results['errors'], bins=30)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Distribution')
    
    # Score distributions
    ax3.hist(results['actual_scores'], bins=30, alpha=0.5, label='Actual')
    ax3.hist(results['predictions'], bins=30, alpha=0.5, label='Predicted')
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Score Distributions')
    ax3.legend()
    
    # Metrics summary
    metrics = results['metrics']
    metrics_text = f"""
    MSE: {metrics['mse']:.4f}
    RMSE: {metrics['rmse']:.4f}
    R²: {metrics['r2']:.4f}
    Correlation: {metrics['correlation']:.4f}
    
    Valid molecules: {len(results['valid_smiles'])}
    Failed molecules: {len(results['failed_smiles'])}
    """
    ax4.text(0.1, 0.1, metrics_text, fontsize=12)
    ax4.axis('off')
    ax4.set_title('Performance Metrics')
    
    plt.tight_layout()
    return fig

def analyze_outliers(results, threshold=2.0):
    """
    Analyzes molecules with large prediction errors
    """
    errors = np.abs(results['errors'])
    outlier_indices = np.where(errors > threshold)[0]
    
    outliers = []
    for idx in outlier_indices:
        outliers.append({
            'smiles': results['valid_smiles'][idx],
            'actual': results['actual_scores'][idx],
            'predicted': results['predictions'][idx],
            'error': results['errors'][idx]
        })
    
    return sorted(outliers, key=lambda x: abs(x['error']), reverse=True)