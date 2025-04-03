from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Set, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Draw, Descriptors, rdDepictor
from rdkit.Chem.Draw import MolDraw2DSVG, rdMolDraw2D, MolDraw2DCairo
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects
from io import BytesIO
import base64
import math
import traceback
import cairosvg
import gc
import os
import time
import tempfile


class ImprovedMoleculeVisualizer:
    def __init__(self):
        self.draw_opts = Draw.DrawingOptions()
        self.draw_opts.addAtomIndices = False
        self.draw_opts.bondLineWidth = 2
        self.draw_opts.atomLabelFontSize = 18
        self.draw_opts.includeAtomNumbers = False
        self.temp_dir = tempfile.mkdtemp()
        self._current_file = None
        
    def _cleanup_previous(self):
        if self._current_file and os.path.exists(self._current_file):
            try:
                os.remove(self._current_file)
            except OSError:
                pass 
    
        # clean up temporary files older than 1 hour
        current_time = time.time()
        for filename in os.listdir(self.temp_dir):
            filepath = os.path.join(self.temp_dir, filename)
            if os.path.getctime(filepath) < (current_time - 3600): 
                try:
                    os.remove(filepath)
                except OSError:
                    pass
        
    def visualize_molecule(self, mol, size=(800, 800)):
        try:
            self._cleanup_previous()
            if not mol:
                return None
            if mol.GetNumConformers() == 0:
                rdDepictor.Compute2DCoords(mol)
            drawer = Draw.MolDraw2DCairo(size[0], size[1])
            drawer.SetDrawOptions(self.draw_opts)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            png = drawer.GetDrawingText()
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.png',
                dir=self.temp_dir
            )
            temp_file.write(png)
            temp_file.close()
            self._current_file = temp_file.name
            with open(self._current_file, 'rb') as f:
                png_base64 = base64.b64encode(f.read()).decode()
            gc.collect()
            return png_base64

        except Exception as e:
            print(f"Visualization error: {str(e)}")
            traceback.print_exc()
            return None
            
    def __del__(self):
        """cleanup when the visualizer is destroyed"""
        try:
            self._cleanup_previous()
            if os.path.exists(self.temp_dir):
                for filename in os.listdir(self.temp_dir):
                    filepath = os.path.join(self.temp_dir, filename)
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                os.rmdir(self.temp_dir)
        except:
            pass  


class EnhancedMoleculeVisualizer:  
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self._current_file = None
        
        #colormaps for different properties
        self.feature_cmap = LinearSegmentedColormap.from_list('feature_colors', [
            '#313695',  
            '#4575b4',  
            '#74add1',  
            '#abd9e9',  
            '#fee090',  
            '#fdae61',
            '#f46d43',  
            '#d73027'   
        ])
        
        self.highlight_cmap = LinearSegmentedColormap.from_list('highlight_colors', [
            '#E83C3C',  # Red for Heteroatom
            '#ff8c00',  # Orange for Charged
            '#00A67E',  # Green for Ring
            '#01579B'   # Blue for Other
        ])

    def _cleanup_previous(self):
        """Clean up any previous temporary files"""
        if self._current_file and os.path.exists(self._current_file):
            try:
                os.remove(self._current_file)
            except OSError:
                pass 
        # clean up temporary files older than 1 hour
        current_time = time.time()
        for filename in os.listdir(self.temp_dir):
            filepath = os.path.join(self.temp_dir, filename)
            if os.path.getctime(filepath) < (current_time - 3600):  
                try:
                    os.remove(filepath)
                except OSError:
                    pass

    def _calculate_dynamic_weights(self, mol):
        total_atoms = mol.GetNumAtoms()
        hetero_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in ['C', 'H'])
        charged_count = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)
        ring_atoms = sum(1 for atom in mol.GetAtoms() if atom.IsInRing())
        total_degree = sum(atom.GetDegree() for atom in mol.GetAtoms())
        hetero_freq = hetero_count / total_atoms if total_atoms > 0 else 0
        charged_freq = charged_count / total_atoms if total_atoms > 0 else 0
        ring_freq = ring_atoms / total_atoms if total_atoms > 0 else 0
        avg_degree = total_degree / total_atoms if total_atoms > 0 else 0
        
        def sigmoid_norm(x): return 1 / (1 + math.exp(-x))
        
        weights = {
            'hetero': sigmoid_norm((1 - hetero_freq) * 5) * 0.5,     # heteroatoms
            'charge': sigmoid_norm((1 - charged_freq) * 5) * 0.4,    # charges
            'ring': sigmoid_norm((1 - ring_freq) * 3) * 0.3,         # non-ring atoms
            'degree': sigmoid_norm((avg_degree - 2) * 0.5) * 0.2     # average degree
        }
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()} if total > 0 else weights

    def _calculate_atom_contribution(self, atom, mol, weights):
        contribution = 0.0
        if atom.GetSymbol() not in ['C', 'H']:
            contribution += weights['hetero']
        if atom.GetFormalCharge() != 0:
            contribution += weights['charge']
        if atom.IsInRing():
            contribution += weights['ring']
        max_degree = max(a.GetDegree() for a in mol.GetAtoms())
        degree_factor = atom.GetDegree() / max_degree if max_degree > 0 else 0
        contribution += weights['degree'] * degree_factor
        return contribution

    def create_feature_view(self, mol: Chem.Mol, prediction_data: Dict[str, Any]) -> str:
        try:
            self._cleanup_previous()
            if not mol:
                return None
            mol = Chem.Mol(mol)
            if mol.GetNumConformers() == 0:
                rdDepictor.Compute2DCoords(mol)
            weights = self._calculate_dynamic_weights(mol)
            atom_contributions = []
            for atom in mol.GetAtoms():
                contribution = self._calculate_atom_contribution(atom, mol, weights)
                atom_contributions.append(contribution)
            max_contrib = max(atom_contributions)
            min_contrib = min(atom_contributions)
            if max_contrib != min_contrib:
                normalized_weights = [(c - min_contrib) / (max_contrib - min_contrib) 
                                   for c in atom_contributions]
            else:
                normalized_weights = [0.5] * len(atom_contributions)
            atom_colors = {}
            for i, weight in enumerate(normalized_weights):
                r = min(1.0, 0.8 + weight * 0.2)
                g = max(0.2, 0.8 - weight * 0.6)
                b = max(0.0, 0.5 - weight * 0.5)
                atom_colors[i] = (r, g, b)
            drawer = rdMolDraw2D.MolDraw2DSVG(400, 370)
            drawer_opts = drawer.drawOptions()
            drawer_opts.bondLineWidth = 4
            drawer_opts.fixedBondLength = 30
            drawer_opts.highlightRadius = 0.5
            drawer_opts.addAtomIndices = True
            drawer_opts.minFontSize = 14
            drawer_opts.maxFontSize = 28
            
            mol = rdMolDraw2D.PrepareMolForDrawing(mol)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(range(mol.GetNumAtoms())),
                highlightBonds=[],
                highlightAtomColors=atom_colors
            )
            drawer.FinishDrawing()
            mol_svg = drawer.GetDrawingText()
            svg = self._create_svg_with_legend(mol_svg, weights)
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.png',
                dir=self.temp_dir
            )
            png_data = cairosvg.svg2png(bytestring=svg.encode())
            temp_file.write(png_data)
            temp_file.close()
            self._current_file = temp_file.name
            with open(self._current_file, 'rb') as f:
                png_base64 = base64.b64encode(f.read()).decode()
            gc.collect()
            
            return png_base64

        except Exception as e:
            print(f"Feature view creation failed: {str(e)}")
            traceback.print_exc()
            return None

    def _create_svg_with_legend(self, mol_svg: str, weights: Dict[str, float]) -> str:
        """Create complete SVG with molecule and dynamic legend with enhanced colors and styling"""
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
        <svg width="400" height="500" viewBox="0 0 400 500" 
            xmlns="http://www.w3.org/2000/svg" version="1.1">
            <style>
                .legend-text {{ 
                    font-family: Arial, sans-serif; 
                    font-size: 12px;
                    fill: #333333;
                }}
                .legend-title {{ 
                    font-family: Arial, sans-serif; 
                    font-size: 14px; 
                    font-weight: bold;
                    fill: #1a1a1a;
                }}
                .feature-circle {{
                    stroke: #ffffff;
                    stroke-width: 1;
                }}
            </style>
            
            <!-- Molecule -->
            <g transform="translate(0,0)">
                {mol_svg[mol_svg.find("<rect"):mol_svg.find("</svg>")]}
            </g>
            
            <!-- Legend -->
            <g transform="translate(10,380)">
                <text x="0" y="0" class="legend-title">Feature Guide (Dynamic Weights)</text>
                
                <!-- Feature indicators -->
                <g transform="translate(0,20)">
                    <circle cx="10" cy="0" r="6" class="feature-circle" fill="#E83C3C" opacity="0.9"/>
                    <text x="25" y="5" class="legend-text">Heteroatom ({weights['hetero']:.3f})</text>
                    
                    <circle cx="160" cy="0" r="6" class="feature-circle" fill="#ff8c00" opacity="0.9"/>
                    <text x="175" y="5" class="legend-text">Charged ({weights['charge']:.3f})</text>
                </g>
                
                <g transform="translate(0,40)">
                    <circle cx="10" cy="0" r="6" class="feature-circle" fill="#00A67E" opacity="0.9"/>
                    <text x="25" y="5" class="legend-text">Ring ({weights['ring']:.3f})</text>
                    
                    <circle cx="160" cy="0" r="6" class="feature-circle" fill="#01579B" opacity="0.9"/>
                    <text x="175" y="5" class="legend-text">Degree ({weights['degree']:.3f})</text>
                </g>
                
                <!-- Contribution scale -->
                <g transform="translate(0,70)">
                    <text x="0" y="5" class="legend-text">Contribution Scale:</text>
                    <defs>
                        <linearGradient id="contribScale" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" style="stop-color:#BDE3FF"/>
                            <stop offset="50%" style="stop-color:#2E5BFF"/>
                            <stop offset="100%" style="stop-color:#003C8F"/>
                        </linearGradient>
                    </defs>
                    <rect x="100" y="0" width="120" height="10" rx="2" fill="url(#contribScale)" 
                        stroke="#ffffff" stroke-width="0.5"/>
                    <text x="100" y="25" class="legend-text">Low</text>
                    <text x="200" y="25" class="legend-text">High</text>
                </g>
            </g>
        </svg>'''
        return svg

    def create_advanced_view(self, mol: Chem.Mol, prediction_data: Dict[str, Any], size=(800, 800)) -> str:
        try:
            self._cleanup_previous()
            
            if not mol:
                return None
            mol = Chem.Mol(mol)
            if mol.GetNumConformers() == 0:
                rdDepictor.Compute2DCoords(mol)
            weights = self._calculate_dynamic_weights(mol)
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            opts = drawer.drawOptions()
            self._configure_drawing_options(opts)

            #colors highlight 
            highlight_atoms, highlight_atom_colors = self._create_highlight_colors(mol, weights)
            highlight_bonds, highlight_bond_colors = self._create_bond_highlights(mol)
            mol = rdMolDraw2D.PrepareMolForDrawing(mol)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(highlight_atoms.keys()),
                highlightBonds=list(highlight_bonds.keys()),
                highlightAtomColors=highlight_atom_colors,
                highlightBondColors=highlight_bond_colors
            )
            drawer.FinishDrawing()
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.png',
                dir=self.temp_dir
            )
            temp_file.write(drawer.GetDrawingText())
            temp_file.close()
            self._current_file = temp_file.name
            with open(self._current_file, 'rb') as f:
                png_base64 = base64.b64encode(f.read()).decode()
            gc.collect()
            return png_base64
        except Exception as e:
            print(f"Advanced visualization failed: {str(e)}")
            traceback.print_exc()
            return None

    def _configure_drawing_options(self, opts):
        opts.bondLineWidth = 2.5
        opts.multipleBondOffset = 0.18
        opts.rotating = True
        opts.addStereoAnnotation = True
        opts.additionalAtomLabelPadding = 0.15
        opts.includeMetadata = True
        opts.prepareMolsBeforeDrawing = True
        opts.setBackgroundColour((1, 1, 1, 1))
        opts.symbolColour = (0.2, 0.2, 0.2, 1)
        opts.legendFontSize = 16
        opts.annotationFontScale = 0.7
        opts.atomLabelFontSize = 14

    def _create_highlight_colors(self, mol, weights):
        """Create dynamic highlight colors for atoms"""
        highlight_atoms = {}
        highlight_atom_colors = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            contribution = self._calculate_atom_contribution(atom, mol, weights)
            if contribution > 0:
                highlight_atoms[idx] = 1

                intensity = min(1.0, contribution * 2) 
                if atom.GetSymbol() not in ['C', 'H']:
                    highlight_atom_colors[idx] = (0.9 * intensity, 0.2, 0.2)  # red for heteroatoms
                elif atom.GetIsAromatic():
                    highlight_atom_colors[idx] = (0.2, 0.6 * intensity, 0.9)  # blue for aromatic
                elif atom.GetFormalCharge() != 0:
                    highlight_atom_colors[idx] = (0.9, 0.6 * intensity, 0.2)  # orange for charged
        return highlight_atoms, highlight_atom_colors

    def _create_bond_highlights(self, mol):
        highlight_bonds = {}
        highlight_bond_colors = {}
        
        for bond in mol.GetBonds():
            idx = bond.GetIdx()
            if bond.GetIsConjugated():
                highlight_bonds[idx] = 1
                highlight_bond_colors[idx] = (0.4, 0.6, 0.9)  # blue for conjugated
            elif not bond.IsInRing():
                highlight_bonds[idx] = 1
                highlight_bond_colors[idx] = (0.6, 0.8, 0.3)  # green for rotatable
                
        return highlight_bonds, highlight_bond_colors

    def __del__(self):
        """Cleanup """
        try:
            self._cleanup_previous()
            if os.path.exists(self.temp_dir):
                for filename in os.listdir(self.temp_dir):
                    filepath = os.path.join(self.temp_dir, filename)
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                os.rmdir(self.temp_dir)
        except:
            pass  
