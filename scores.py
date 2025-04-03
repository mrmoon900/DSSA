from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import io
from typing import Dict, Any, List, Optional, Tuple
from rdkit import Chem
import math
import uvicorn
from visualizer import ImprovedMoleculeVisualizer, EnhancedMoleculeVisualizer

class MoleculePredictor:
    """ML-based"""
    
    def __init__(self, model_path: str = "app/models/best_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.available() else 'cpu')
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        try:
            from app.src.DSSA_model import DSSA
            from app.src.data_preprocess import dataload
            (_, _, _, num_node_features, _, num_node_types, 
             num_edge_types, _, _) = dataload()
            model = DSSA(
                in_dim=num_node_features,
                hidden_dim=128,
                num_layers=4,
                num_heads=4,
                dropout=0,
                num_classes=1,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                processing_steps=3
            ).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def predict_batch(self, df: pd.DataFrame) -> List[Optional[float]]:
        try:
            from app.src.data_preprocess import RobustMoleculeDataset
            dataset = RobustMoleculeDataset(
                df,
                'smiles',
                'labels' 
            )
            predictions = []
            with torch.no_grad():
                for i in range(len(dataset)):
                    try:
                        data = dataset[i].to(self.device)
                        output = self.model(data)
                        pred = float(torch.sigmoid(output).cpu().numpy().reshape(-1)[0])
                        predictions.append(pred)
                    except Exception as e:
                        print(f"Prediction failed for molecule {i}: {str(e)}")
                        predictions.append(None)
            return predictions 
        except Exception as e:
            print(f"Batch prediction failed: {str(e)}")
            return [None] * len(df)
        
class MLPredictor:
    def __init__(self):
        self.ml_predictor = MoleculePredictor()
        self.visualizer = EnhancedMoleculeVisualizer()
        self.simple_visualizer = ImprovedMoleculeVisualizer()
    def validate_molecule(self, smiles: str) -> Tuple[bool, str]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "Invalid SMILES string"
            if mol.GetNumAtoms() > 100:
                return False, "Molecule too large (>100 atoms)"
            #atom types
            valid_atoms = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H,C,N,O,F,P,S,Cl,Br,I
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in valid_atoms:
                    return False, f"Unsupported atom type: {atom.GetSymbol()}"
            return True,
        except Exception as e:
            return False, str(e)

    async def analyze_molecules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """batch of molecules"""
        try:
            if 'labels' not in df.columns:
                df['labels'] = 0
            ml_predictions = self.ml_predictor.predict_batch(df)
            results = []
            valid_predictions = []
            for smiles, ml_pred in zip(df['smiles'].tolist(), ml_predictions):
                mol = Chem.MolFromSmiles(smiles)
                visualization = None
                advanced_viz = None
                if mol and len(results) < 5:  # limit visualizations to first 5 molecules
                    visualization = self.simple_visualizer.visualize_molecule(mol)
                    if ml_pred is not None:
                        prediction_data = {'hs': ml_pred, 'es': 1 - ml_pred}
                        advanced_viz = self.visualizer.create_feature_view(mol, prediction_data)
                result = {
                    'smiles': smiles,
                    'hs': ml_pred,
                    'es': 1 - ml_pred if ml_pred is not None else None,  # Inverted score
                    'non-binary_score': 10 * ml_pred if ml_pred is not None else None,  # 10 * ES
                    'synthesis_difficulty': 'unknown' if ml_pred is None else
                                           'easy' if ml_pred < 0.4 else
                                           'moderate' if ml_pred < 0.7 else 'hard'
                }
                if visualization or advanced_viz:
                    result['visualizations'] = {}
                    if visualization:
                        result['visualizations']['basic'] = visualization
                    if advanced_viz:
                        result['visualizations']['feature'] = advanced_viz
                if ml_pred is not None:
                    valid_predictions.append(ml_pred)
                results.append(result)
            
            #statistics calacuations
            stats = {
                'num_processed': len(results),
                'num_valid': len(valid_predictions),
                'mean_ml_prediction': float(np.mean(valid_predictions)) if valid_predictions else None,
                'synthesis_difficulty_distribution': {
                    'easy': sum(1 for r in results if r.get('synthesis_difficulty') == 'easy'),
                    'hard': sum(1 for r in results if r.get('synthesis_difficulty') == 'hard'),
                    'unknown': sum(1 for r in results if r.get('synthesis_difficulty') == 'unknown')
                }
            }
            return {
                'status': 'success',
                'statistics': stats,
                'predictions': results
            }
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

#FastAPI app
app = FastAPI(
    title="DSSA Score Prediction API",
    description="Advanced API for molecular synthetic accessibility scoring using ML",
    version="2.0.0"
)
_predictor = None


def get_predictor() -> MLPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor

@app.get("/")
async def root():
    return {"message": "Welcome to DSSA Score Prediction API!"}

@app.post("/predict")
async def predict_score(file: UploadFile = File(...)):
    try:
        print("Starting prediction request...")
        content = await file.read()
        try:
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        except UnicodeDecodeError:
            df = pd.read_csv(io.StringIO(content.decode('latin-1')))
        if 'smiles' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'smiles' column"
            )
        predictor = get_predictor()
        print("Starting molecule analysis...")
        results = await predictor.analyze_molecules(df)
        print("Analysis complete")
        return JSONResponse(content=results)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
        
