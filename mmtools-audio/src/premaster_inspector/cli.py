"""Command-line interface using Click and Rich."""

from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import audio_io, metrics, loudness, stereo, spectrum, transients, issues, report


console = Console()


@click.group()
def cli():
    """MMTools - Technical analysis for mix review and master validation."""
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--out', '-o', default='./report', 
              help='Output directory for reports (default: ./report)')
def analyze(input_file: str, out: str):
    """
    Analyze a WAV/AIFF audio file for mix and master issues.
    
    Generates terminal report, JSON, and HTML with diagnostic plots.
    """
    
    input_path = Path(input_file)
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold cyan]MMTools by Alexandre Lack[/bold cyan] v0.2.0")
    console.print(f"[dim]Analyzing: {input_path.name}[/dim]\n")
    
    # 1. Load audio
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("Loading audio...", total=None)
        try:
            audio_data = audio_io.load_audio(input_path)
        except Exception as e:
            console.print(f"[red]Error loading audio:[/red] {e}")
            return
    
    # 2. Analyze
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("Analyzing audio...", total=None)
        
        # Check if stereo
        is_stereo = audio_io.validate_stereo(audio_data)
        if not is_stereo:
            console.print(f"[yellow]Warning: Mono audio detected. Stereo metrics will not apply.[/yellow]")
        
        # Compute all metrics
        metrics_l = metrics.analyze_channel_metrics(audio_data.samples, 0)
        metrics_r = (metrics.analyze_channel_metrics(audio_data.samples, 1) 
                    if audio_data.channel_count > 1 else metrics_l)
        
        clipping = metrics.analyze_clipping(audio_data.samples)
        loud = loudness.analyze_loudness(audio_data.samples, audio_data.sample_rate)
        stereo_metrics = stereo.analyze_stereo(audio_data.samples, audio_data.sample_rate)
        spec = spectrum.analyze_spectrum(audio_data.samples, audio_data.sample_rate)
        trans = transients.analyze_transients(
            np.mean(audio_data.samples, axis=1), audio_data.sample_rate
        )
        
        # Detect issues
        detected_issues = issues.detect_issues(
            audio_data.samples, audio_data.sample_rate
        )
    
    # 3. Display summary
    _display_summary(audio_data, metrics_l, metrics_r, clipping, loud, stereo_metrics, spec)
    
    # 4. Display issues
    _display_issues(detected_issues)
    
    # 5. Generate reports
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("Generating reports...", total=None)
        
        # Prepare all metrics dict
        all_metrics = {
            "channels": {
                "L": {
                    "peak_dbfs": metrics_l.peak_dbfs,
                    "rms_dbfs": metrics_l.rms_dbfs,
                    "dc_offset": metrics_l.dc_offset,
                    "crest_factor": metrics_l.crest_factor,
                },
                "R": {
                    "peak_dbfs": metrics_r.peak_dbfs,
                    "rms_dbfs": metrics_r.rms_dbfs,
                    "dc_offset": metrics_r.dc_offset,
                    "crest_factor": metrics_r.crest_factor,
                } if audio_data.channel_count > 1 else None,
            },
            "clipping": {
                "left_count": clipping.clipped_samples_l,
                "right_count": clipping.clipped_samples_r,
                "left_ratio": clipping.clipping_ratio_l,
                "right_ratio": clipping.clipping_ratio_r,
            },
            "loudness": {
                "lufs_integrated": loud.lufs_integrated if not np.isinf(loud.lufs_integrated) else None,
                "true_peak_dbtp": loud.true_peak_dbtp,
                "peak_dbfs": loud.peak_dbfs,
            },
            "stereo": {
                "correlation_global": stereo_metrics.correlation_global,
                "delay_ms": stereo_metrics.delay_ms,
                "mid_rms_dbfs": stereo_metrics.mid_rms_dbfs,
                "side_rms_dbfs": stereo_metrics.side_rms_dbfs,
                "correlation_by_band": stereo_metrics.correlation_by_band,
            },
            "spectrum": {
                "infrasonic_dbfs": spec.infrasonic_energy_dbfs,
                "ultrasonic_dbfs": spec.ultrasonic_energy_dbfs,
                "hum_50hz_dbfs": spec.hum_50hz_dbfs if not np.isinf(spec.hum_50hz_dbfs) else None,
                "hum_60hz_dbfs": spec.hum_60hz_dbfs if not np.isinf(spec.hum_60hz_dbfs) else None,
            },
        }
        
        # Create report object
        rep = report.AnalysisReport(
            audio_path=str(input_path),
            duration=audio_data.duration,
            sample_rate=audio_data.sample_rate,
            channel_count=audio_data.channel_count,
            all_metrics=all_metrics,
            issues=detected_issues
        )
        
        # Save JSON
        json_path = rep.save_json(output_dir / "report.json")
        
        # Save HTML with plots
        html_path = rep.save_html(
            output_dir / "report.html",
            audio_data=audio_data.samples,
            sample_rate=audio_data.sample_rate
        )
    
    # 6. Print summary
    console.print(f"\n[bold green]✓ Analysis complete![/bold green]\n")
    
    report_table = Table(title="Generated Reports", show_header=True)
    report_table.add_column("Format", style="cyan")
    report_table.add_column("Path", style="green")
    
    report_table.add_row("JSON", str(json_path))
    report_table.add_row("HTML", str(html_path))
    
    console.print(report_table)
    console.print(f"\n[dim]Reports saved to: {output_dir.absolute()}[/dim]\n")


def _display_summary(audio_data, metrics_l, metrics_r, clipping, loud, stereo_metrics, spec):
    """Display audio metrics summary."""
    summary_table = Table(title="Audio Metrics Summary", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Left", style="yellow")
    summary_table.add_column("Right" if audio_data.channel_count > 1 else "", style="yellow")
    
    summary_table.add_row(
        "Peak Level",
        f"{metrics_l.peak_dbfs:.2f} dBFS",
        f"{metrics_r.peak_dbfs:.2f} dBFS" if audio_data.channel_count > 1 else ""
    )
    
    summary_table.add_row(
        "RMS Level",
        f"{metrics_l.rms_dbfs:.2f} dBFS",
        f"{metrics_r.rms_dbfs:.2f} dBFS" if audio_data.channel_count > 1 else ""
    )
    
    summary_table.add_row(
        "DC Offset",
        f"{metrics_l.dc_offset:.6f}",
        f"{metrics_r.dc_offset:.6f}" if audio_data.channel_count > 1 else ""
    )
    
    summary_table.add_row(
        "Crest Factor",
        f"{metrics_l.crest_factor:.2f}",
        f"{metrics_r.crest_factor:.2f}" if audio_data.channel_count > 1 else ""
    )
    
    if not np.isinf(loud.lufs_integrated):
        summary_table.add_row(
            "Loudness (LUFS)",
            f"{loud.lufs_integrated:.2f} LUFS",
            ""
        )
    
    summary_table.add_row(
        "True Peak",
        f"{loud.true_peak_dbtp:.2f} dBTP",
        ""
    )
    
    if audio_data.channel_count > 1:
        summary_table.add_row(
            "L/R Correlation",
            f"{stereo_metrics.correlation_global:.3f}",
            ""
        )
        
        if abs(stereo_metrics.delay_ms) > 0.1:
            summary_table.add_row(
                "L/R Delay",
                f"{stereo_metrics.delay_ms:.2f} ms",
                ""
            )
    
    summary_table.add_row(
        "Clipping Samples",
        f"{clipping.clipped_samples_l}",
        f"{clipping.clipped_samples_r}" if audio_data.channel_count > 1 else ""
    )
    
    console.print(summary_table)


def _display_issues(detected_issues):
    """Display issues grouped by severity."""
    if not detected_issues:
        console.print(Panel(
            "[bold green]✓ No issues detected![/bold green]",
            title="Issues Report",
            border_style="green"
        ))
        return
    
    # Group by severity
    by_severity = {"high": [], "medium": [], "low": []}
    for issue in detected_issues:
        by_severity[issue.severity].append(issue)
    
    severity_order = ["high", "medium", "low"]
    severity_colors = {"high": "red", "medium": "yellow", "low": "green"}
    
    for severity in severity_order:
        if not by_severity[severity]:
            continue
        
        color = severity_colors[severity]
        count = len(by_severity[severity])
        title_text = f"[{color}]{severity.upper()} SEVERITY ({count})[/{color}]"
        
        issues_text = ""
        for issue in by_severity[severity]:
            issues_text += f"\n[bold]{issue.title}[/bold]\n"
            issues_text += f"  Category: {issue.category}\n"
            issues_text += f"  Confidence: {issue.confidence:.0%}\n"
            if issue.evidence:
                issues_text += f"  Evidence: {issue.evidence}\n"
            if issue.impact:
                issues_text += f"  Impact: {issue.impact}\n"
            if issue.suggestions:
                issues_text += "  Suggestions:\n"
                for sug in issue.suggestions:
                    issues_text += f"    • {sug}\n"
        
        console.print(Panel(issues_text.strip(), title=title_text, border_style=color))


def main():
    """Entry point."""
    cli()


if __name__ == '__main__':
    main()
