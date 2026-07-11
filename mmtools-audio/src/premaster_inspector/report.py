"""Report generation: JSON and HTML."""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import loudness, stereo, spectrum, metrics
from .issues import Issue


class AnalysisReport:
    """Container for full analysis results."""
    
    def __init__(self, audio_path: str, duration: float, sample_rate: int,
                 channel_count: int, all_metrics: dict, issues: list[Issue]):
        self.audio_path = str(audio_path)
        self.duration = duration
        self.sample_rate = sample_rate
        self.channel_count = channel_count
        self.timestamp = datetime.now().isoformat()
        self.all_metrics = all_metrics
        self.issues = issues
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Convert Issue objects to dicts
        issues_list = []
        for issue in self.issues:
            issue_dict = asdict(issue)
            issues_list.append(issue_dict)
        
        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        issues_list.sort(key=lambda x: severity_order.get(x["severity"], 3))
        
        return {
            "metadata": {
                "audio_file": self.audio_path,
                "duration_seconds": self.duration,
                "sample_rate": self.sample_rate,
                "channels": self.channel_count,
                "analysis_timestamp": self.timestamp,
            },
            "metrics": self.all_metrics,
            "issues": issues_list,
            "summary": self._generate_summary(),
        }
    
    def _generate_summary(self) -> dict:
        """Generate summary statistics."""
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        categories = {}
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
            cat = issue.category
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_issues": len(self.issues),
            "severity_counts": severity_counts,
            "category_counts": categories,
        }
    
    def save_json(self, output_path: str | Path) -> Path:
        """Save analysis as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        return output_path
    
    def save_html(self, output_path: str | Path, audio_data=None, 
                  sample_rate: int = None) -> Path:
        """Save analysis as HTML with inline CSS and optional plots."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create plots directory
        plots_dir = output_path.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plots_paths = {}
        if audio_data is not None and sample_rate is not None:
            plots_paths = self._generate_plots(audio_data, sample_rate, plots_dir)
        
        html_content = self._generate_html(plots_paths)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_plots(self, audio: np.ndarray, sample_rate: int, 
                       output_dir: Path) -> dict[str, str]:
        """Generate diagnostic plots."""
        plots = {}
        
        # Waveform
        try:
            fig = self._plot_waveform(audio, sample_rate)
            path = output_dir / "waveform.png"
            fig.savefig(path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            plots['waveform'] = str(path.relative_to(output_dir.parent))
        except Exception:
            pass
        
        # Spectrum
        try:
            fig = self._plot_spectrum(audio, sample_rate)
            path = output_dir / "spectrum.png"
            fig.savefig(path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            plots['spectrum'] = str(path.relative_to(output_dir.parent))
        except Exception:
            pass
        
        # Mid/Side
        try:
            fig = self._plot_mid_side(audio, sample_rate)
            path = output_dir / "mid_side.png"
            fig.savefig(path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            plots['mid_side'] = str(path.relative_to(output_dir.parent))
        except Exception:
            pass
        
        return plots
    
    def _plot_waveform(self, audio: np.ndarray, sample_rate: int) -> Figure:
        """Plot waveform (decimated for large files)."""
        fig, axes = plt.subplots(audio.shape[1], 1, figsize=(12, 4 * audio.shape[1]))
        if audio.shape[1] == 1:
            axes = [axes]
        
        time = np.arange(audio.shape[0]) / sample_rate
        
        # Decimate for display
        decimation = max(1, audio.shape[0] // 10000)
        time_dec = time[::decimation]
        
        for ch in range(audio.shape[1]):
            audio_dec = audio[::decimation, ch]
            axes[ch].plot(time_dec, audio_dec, linewidth=0.5)
            axes[ch].set_ylabel(f'Ch{ch + 1}')
            axes[ch].set_ylim([-1, 1])
            axes[ch].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('Waveform Overview')
        
        return fig
    
    def _plot_spectrum(self, audio: np.ndarray, sample_rate: int) -> Figure:
        """Plot spectrum."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for ch in range(audio.shape[1]):
            freqs, power = spectrum.compute_spectrum(audio[:, ch:ch+1], sample_rate)
            ax.semilogy(freqs, 10 ** (power / 10), label=f'Ch{ch + 1}', alpha=0.7)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_xlim([20, sample_rate // 2])
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        fig.suptitle('Power Spectrum')
        
        return fig
    
    def _plot_mid_side(self, audio: np.ndarray, sample_rate: int) -> Figure:
        """Plot Mid/Side energy by band."""
        if audio.shape[1] < 2:
            fig = plt.figure()
            return fig
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        mid_energies, side_energies = stereo.analyze_mid_side_by_band(audio, sample_rate)
        
        bands = list(mid_energies.keys())
        mid_vals = [mid_energies[b] for b in bands]
        side_vals = [side_energies[b] for b in bands]
        
        x = np.arange(len(bands))
        width = 0.35
        
        ax.bar(x - width / 2, mid_vals, width, label='Mid', alpha=0.8)
        ax.bar(x + width / 2, side_vals, width, label='Side', alpha=0.8)
        
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Energy (dBFS)')
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.suptitle('Mid/Side Energy by Band')
        fig.tight_layout()
        
        return fig
    
    def _generate_html(self, plots_paths: dict) -> str:
        """Generate HTML report."""
        data = self.to_dict()
        
        # Group issues by severity
        issues_by_severity = {"high": [], "medium": [], "low": []}
        for issue in self.issues:
            issues_by_severity[issue.severity].append(issue)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MMTools Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .summary-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #3498db;
            border-radius: 5px;
        }}
        .summary-card h4 {{
            margin: 0 0 10px 0;
            color: #34495e;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .issues-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .issues-table th {{
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        .issues-table td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        .issues-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .severity-high {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .severity-medium {{
            color: #f39c12;
            font-weight: bold;
        }}
        .severity-low {{
            color: #27ae60;
            font-weight: bold;
        }}
        .confidence {{
            display: inline-block;
            background-color: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .evidence {{
            font-family: monospace;
            background-color: #f9f9f9;
            padding: 5px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .suggestions {{
            background-color: #ecf0f1;
            padding: 10px;
            border-left: 3px solid #3498db;
            border-radius: 3px;
            margin-top: 10px;
            font-size: 0.9em;
        }}
        .suggestions ul {{
            margin: 5px 0;
            padding-left: 20px;
        }}
        .suggestions li {{
            margin: 5px 0;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .note {{
            background-color: #fffacd;
            padding: 15px;
            border-left: 4px solid #f0ad4e;
            border-radius: 5px;
            margin: 20px 0;
        }}
        footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MMTools by Alexandre Lack Report</h1>
        
        <div class="metadata">
            <p><strong>File:</strong> {data['metadata']['audio_file']}</p>
            <p><strong>Duration:</strong> {data['metadata']['duration_seconds']:.2f} seconds</p>
            <p><strong>Sample Rate:</strong> {data['metadata']['sample_rate']} Hz</p>
            <p><strong>Channels:</strong> {data['metadata']['channels']}</p>
            <p><strong>Analysis Time:</strong> {data['metadata']['analysis_timestamp']}</p>
        </div>
        
        <h2>Summary</h2>
        <div class="summary">
            <div class="summary-card">
                <h4>Total Issues</h4>
                <div class="value">{data['summary']['total_issues']}</div>
            </div>
            <div class="summary-card">
                <h4>🔴 High Severity</h4>
                <div class="value" style="color: #e74c3c;">{data['summary']['severity_counts']['high']}</div>
            </div>
            <div class="summary-card">
                <h4>🟠 Medium Severity</h4>
                <div class="value" style="color: #f39c12;">{data['summary']['severity_counts']['medium']}</div>
            </div>
            <div class="summary-card">
                <h4>🟢 Low Severity</h4>
                <div class="value" style="color: #27ae60;">{data['summary']['severity_counts']['low']}</div>
            </div>
        </div>
        
        <div class="note">
            <strong>ℹ️ Note:</strong> This tool is <strong>assistive</strong> and does not substitute for professional 
            mastering judgment. All findings should be verified by a qualified audio engineer. Thresholds and 
            confidence levels are conservative estimates based on audio analysis heuristics.
        </div>
"""
        
        # Add plots if available
        if plots_paths:
            html += "<h2>Diagnostic Plots</h2>\n"
            for plot_name, plot_path in plots_paths.items():
                title = plot_name.replace('_', ' ').title()
                html += f"""        <div class="plot">
            <h3>{title}</h3>
            <img src="{plot_path}" alt="{title}">
        </div>
"""
        
        # Add issues by severity
        for severity in ["high", "medium", "low"]:
            if not issues_by_severity[severity]:
                continue
            
            severity_emoji = {"high": "🔴", "medium": "🟠", "low": "🟢"}[severity]
            severity_title = severity.upper()
            
            html += f"""
        <h2>{severity_emoji} {severity_title} SEVERITY ISSUES</h2>
        <table class="issues-table">
            <thead>
                <tr>
                    <th>Issue</th>
                    <th>Category</th>
                    <th>Confidence</th>
                    <th>Evidence</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for issue in issues_by_severity[severity]:
                severity_class = f"severity-{severity}"
                html += f"""                <tr>
                    <td>
                        <div><strong>{issue.title}</strong></div>
                        <div class="suggestions">
                            <strong>Impact:</strong> {issue.impact or 'N/A'}<br>
"""
                if issue.suggestions:
                    html += "<strong>Suggestions:</strong><ul>\n"
                    for suggestion in issue.suggestions:
                        html += f"<li>{suggestion}</li>\n"
                    html += "</ul>"
                html += f"""                        </div>
                    </td>
                    <td>{issue.category}</td>
                    <td><span class="confidence">{issue.confidence:.0%}</span></td>
                    <td><span class="evidence">{issue.evidence or 'N/A'}</span></td>
                </tr>
"""
            
            html += """            </tbody>
        </table>
"""
        
        if not self.issues:
            html += """
        <div class="note">
            <strong>✓ No issues detected!</strong> This audio appears to be technically well-prepared for mastering.
        </div>
"""
        
        html += """
        <footer>
            <p>Generated by MMTools by Alexandre Lack v0.2.0</p>
            <p>For mix review and master validation workflows.</p>
        </footer>
    </div>
</body>
</html>
"""
        
        return html
