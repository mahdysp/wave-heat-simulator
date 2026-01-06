"""Export utilities for data and figures"""

import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def export_to_csv(sim, L: float, T: float, filename: str = None, 
                 num_x_points: int = 100, num_t_points: int = 50) -> bool:
    """
    Export simulation data to CSV file
    
    Args:
        sim: Simulation object (wave or heat)
        L: Domain length
        T: Total simulation time
        filename: Output filename (optional, will prompt if None)
        num_x_points: Number of spatial points
        num_t_points: Number of time points
        
    Returns:
        True if successful
    """
    try:
        if not filename:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"simulation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        
        if not filename:
            return False
        
        # Generate data grid
        x = np.linspace(0, L, num_x_points)
        t_array = np.linspace(0, T, num_t_points)
        
        # Write CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write metadata as comments
            writer.writerow([f"# Simulation Type: {sim.__class__.__name__}"])
            writer.writerow([f"# Length: {L}, Time: {T}"])
            writer.writerow([f"# Generated: {datetime.now().isoformat()}"])
            writer.writerow([])
            
            # Write header
            header = ["time\\x"] + [f"{xi:.4f}" for xi in x]
            writer.writerow(header)
            
            # Write data
            for t in t_array:
                u = sim.solution(x, t)
                row = [f"{t:.4f}"] + [f"{ui:.6f}" for ui in u]
                writer.writerow(row)
        
        messagebox.showinfo("Success", f"Data exported to {os.path.basename(filename)}")
        return True
        
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export CSV: {str(e)}")
        return False


def export_to_json(data: Dict[str, Any], filename: str = None,
                  include_metadata: bool = True) -> bool:
    """
    Export parameters and settings to JSON file
    
    Args:
        data: Dictionary of data to export
        filename: Output filename (optional)
        include_metadata: Whether to include metadata
        
    Returns:
        True if successful
    """
    try:
        if not filename:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        if not filename:
            return False
        
        # Add metadata if requested
        if include_metadata:
            export_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "generator": "PDE Fourier Simulator"
                },
                "data": data
            }
        else:
            export_data = data
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = _convert_numpy_to_list(export_data)
        
        # Write JSON with pretty formatting
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, sort_keys=True)
        
        messagebox.showinfo("Success", f"Parameters exported to {os.path.basename(filename)}")
        return True
        
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export JSON: {str(e)}")
        return False


def save_figure(fig: Figure, filename: str = None, dpi: int = 150,
               quality: int = 95, transparent: bool = False) -> bool:
    """
    Save matplotlib figure to file
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename (optional)
        dpi: Resolution in dots per inch
        quality: JPEG quality (1-100)
        transparent: Whether to use transparent background
        
    Returns:
        True if successful
    """
    try:
        if not filename:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ],
                initialfile=f"figure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
        
        if not filename:
            return False
        
        # Determine format from extension
        ext = os.path.splitext(filename)[1].lower()
        
        # Set appropriate parameters based on format
        save_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'transparent': transparent
        }
        
        if ext in ['.jpg', '.jpeg']:
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True
        
        # Save figure
        fig.savefig(filename, **save_kwargs)
        
        # Check file size
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        size_info = f" ({file_size_mb:.2f} MB)" if file_size_mb > 1 else ""
        
        messagebox.showinfo("Success", 
                          f"Figure saved to {os.path.basename(filename)}{size_info}")
        return True
        
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to save figure: {str(e)}")
        return False


def export_data_matrix(data: np.ndarray, filename: str = None,
                      format: str = 'npz') -> bool:
    """
    Export numpy array data in various formats
    
    Args:
        data: Numpy array to export
        filename: Output filename
        format: Export format ('npz', 'npy', 'mat', 'h5')
        
    Returns:
        True if successful
    """
    try:
        if not filename:
            extensions = {
                'npz': ("NumPy compressed", "*.npz"),
                'npy': ("NumPy array", "*.npy"),
                'mat': ("MATLAB file", "*.mat"),
                'h5': ("HDF5 file", "*.h5")
            }
            
            file_type = extensions.get(format, ("Data file", f"*.{format}"))
            
            filename = filedialog.asksaveasfilename(
                defaultextension=f".{format}",
                filetypes=[file_type, ("All files", "*.*")],
                initialfile=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            )
        
        if not filename:
            return False
        
        # Export based on format
        if format == 'npz':
            np.savez_compressed(filename, data=data)
        elif format == 'npy':
            np.save(filename, data)
        elif format == 'mat':
            import scipy.io
            scipy.io.savemat(filename, {'data': data})
        elif format == 'h5':
            import h5py
            with h5py.File(filename, 'w') as f:
                f.create_dataset('data', data=data, compression='gzip')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        messagebox.showinfo("Success", f"Data exported to {os.path.basename(filename)}")
        return True
        
    except ImportError as e:
        messagebox.showerror("Import Error", 
                           f"Required library not installed for {format} format: {str(e)}")
        return False
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
        return False


def export_coefficients(sim, filename: str = None, format: str = 'csv') -> bool:
    """
    Export Fourier coefficients
    
    Args:
        sim: Simulation object with coefficients
        filename: Output filename
        format: Export format ('csv' or 'json')
        
    Returns:
        True if successful
    """
    try:
        if not filename:
            ext = 'csv' if format == 'csv' else 'json'
            filename = filedialog.asksaveasfilename(
                defaultextension=f".{ext}",
                filetypes=[(f"{ext.upper()} files", f"*.{ext}"), ("All files", "*.*")],
                initialfile=f"coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
            )
        
        if not filename:
            return False
        
        # Prepare coefficient data
        if hasattr(sim, 'An') and hasattr(sim, 'Bn'):
            # Wave equation coefficients
            coeffs = {
                'An': sim.An.tolist() if isinstance(sim.An, np.ndarray) else sim.An,
                'Bn': sim.Bn.tolist() if isinstance(sim.Bn, np.ndarray) else sim.Bn
            }
        elif hasattr(sim, 'Bn'):
            # Heat equation coefficients
            coeffs = {
                'Bn': sim.Bn.tolist() if isinstance(sim.Bn, np.ndarray) else sim.Bn
            }
        else:
            raise ValueError("No coefficients found in simulation object")
        
        if format == 'csv':
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['n'] + list(coeffs.keys()))
                max_len = max(len(v) for v in coeffs.values())
                for i in range(max_len):
                    row = [i + 1]
                    for key in coeffs.keys():
                        if i < len(coeffs[key]):
                            row.append(f"{coeffs[key][i]:.6e}")
                        else:
                            row.append("")
                    writer.writerow(row)
        else:  # JSON
            with open(filename, 'w') as f:
                json.dump(coeffs, f, indent=2)
        
        messagebox.showinfo("Success", f"Coefficients exported to {os.path.basename(filename)}")
        return True
        
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export coefficients: {str(e)}")
        return False


def export_animation_frames(sim, L: float, T: float, filename_pattern: str = None,
                          num_frames: int = 100, dpi: int = 100) -> bool:
    """
    Export animation as individual frames
    
    Args:
        sim: Simulation object
        L: Domain length
        T: Total time
        filename_pattern: Pattern for filenames (e.g., 'frame_{:04d}.png')
        num_frames: Number of frames to export
        dpi: Resolution
        
    Returns:
        True if successful
    """
    try:
        if not filename_pattern:
            directory = filedialog.askdirectory(title="Select directory for frames")
            if not directory:
                return False
            filename_pattern = os.path.join(directory, "frame_{:04d}.png")
        
        # Generate frames
        x = np.linspace(0, L, 500)
        times = np.linspace(0, T, num_frames)
        
        # Create figure for frames
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, t in enumerate(times):
            ax.clear()
            u = sim.solution(x, t)
            ax.plot(x, u, 'b-', linewidth=2)
            ax.set_xlabel('Position x')
            ax.set_ylabel('u(x,t)')
            ax.set_title(f'Time: {t:.3f} s')
            ax.grid(True, alpha=0.3)
            
            # Save frame
            frame_filename = filename_pattern.format(i)
            fig.savefig(frame_filename, dpi=dpi, bbox_inches='tight')
            
            # Update progress (could add progress bar here)
            if (i + 1) % 10 == 0:
                print(f"Exported {i + 1}/{num_frames} frames")
        
        plt.close(fig)
        
        messagebox.showinfo("Success", f"Exported {num_frames} frames")
        return True
        
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export frames: {str(e)}")
        return False


def create_report(sim, params: Dict[str, Any], filename: str = None,
                 include_plots: bool = True) -> bool:
    """
    Create a comprehensive HTML report
    
    Args:
        sim: Simulation object
        params: Dictionary of parameters
        filename: Output filename
        include_plots: Whether to include plots in report
        
    Returns:
        True if successful
    """
    try:
        if not filename:
            filename = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
                initialfile=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
        
        if not filename:
            return False
        
        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PDE Simulation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .plot {{ max-width: 800px; margin: 20px auto; }}
        .metadata {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>PDE Simulation Report</h1>
    <div class="metadata">
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Simulation Type:</strong> {sim.__class__.__name__}</p>
    </div>
    
    <h2>Parameters</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
"""
        
        # Add parameters to table
        for key, value in params.items():
            if isinstance(value, (int, float)):
                html_content += f"        <tr><td>{key}</td><td>{value:.6g}</td></tr>\n"
            else:
                html_content += f"        <tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html_content += """
    </table>
    
    <h2>Results Summary</h2>
"""
        
        # Add coefficient information if available
        if hasattr(sim, 'An'):
            html_content += f"""
    <p><strong>Number of Fourier coefficients:</strong> {len(sim.An)}</p>
    <p><strong>Max |An|:</strong> {np.max(np.abs(sim.An)):.6e}</p>
"""
        
        if hasattr(sim, 'Bn'):
            html_content += f"""
    <p><strong>Max |Bn|:</strong> {np.max(np.abs(sim.Bn)):.6e}</p>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Write HTML file
        with open(filename, 'w') as f:
            f.write(html_content)
        
        messagebox.showinfo("Success", f"Report saved to {os.path.basename(filename)}")
        return True  #by-mahdysp
        
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to create report: {str(e)}")
        return False


def _convert_numpy_to_list(obj: Any) -> Any:
    """
    Recursively convert numpy arrays to lists for JSON serialization
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_to_list(item) for item in obj)
    else:
        return obj


# Additional utility functions

def estimate_file_size(data: Union[np.ndarray, Dict], format: str) -> float:
    """
    Estimate file size in MB before exporting
    
    Args:
        data: Data to export
        format: Export format
        
    Returns:
        Estimated size in MB
    """
    if isinstance(data, np.ndarray):
        # Rough estimation based on array size and format
        bytes_per_element = data.dtype.itemsize
        total_bytes = data.size * bytes_per_element
        
        if format == 'csv':
            # CSV typically 2-3x larger due to text representation
            total_bytes *= 2.5
        elif format == 'npz':
            # Compressed, typically 30-70% of original
            total_bytes *= 0.5
        
        return total_bytes / (1024 * 1024)
    
    elif isinstance(data, dict):
        # For JSON, rough estimate
        json_str = json.dumps(data, default=str)
        return len(json_str) / (1024 * 1024)
    
    return 0


def validate_export_size(data: Any, format: str, max_size_mb: float = 100) -> bool:
    """
    Check if export size is within limits
    
    Args:
        data: Data to export
        format: Export format
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if within limits
    """
    estimated_size = estimate_file_size(data, format)
    
    if estimated_size > max_size_mb:
        response = messagebox.askyesno(
            "Large File Warning",
            f"Estimated file size is {estimated_size:.2f} MB. "
            f"This exceeds the recommended maximum of {max_size_mb} MB. "
            "Do you want to continue?"
        )
        return response
    
    return True
