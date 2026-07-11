"""
MMTools Web Interface.

Integrated mix review and mastering validation app with PT/EN UI.
"""

import html
import io
import json
import os
import sys
import tempfile
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "mmtools-audio" / "src"))

from premaster_inspector import audio_io, issues, loudness, mastering, metrics, player
from premaster_inspector import report, spectrum, stereo, transients


APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
LOGO_CANDIDATES = (
    ASSETS_DIR / "logo.png",
    ASSETS_DIR / "logo.svg",
    ASSETS_DIR / "logo.jpg",
    ASSETS_DIR / "logo.jpeg",
    ASSETS_DIR / "logo.webp",
)


TEXT = {
    "en": {
        "language": "Language",
        "mode": "Analysis mode",
        "mix_mode": "Mix Review",
        "master_mode": "Master Validation",
        "title": "MMTools by Alexandre Lack",
        "subtitle": "Mix review and master validation for WAV/AIFF files",
        "about_title": "About",
        "about": """**MMTools** analyzes mixes before mastering and validates final masters.

**Mix Review**
- Loudness and headroom
- Stereo correlation and delay
- Frequency-domain checks
- Issue detection and recommendations

**Master Validation**
- LUFS, true peak, RMS
- DC offset and aliasing checks
- Mono compatibility
- Stereo/mono playback

**Reports:** JSON and HTML""",
        "upload": "Upload Your Audio File",
        "upload_help": "Choose a WAV or AIFF file for analysis",
        "analyzing": "Analyzing audio... Please wait.",
        "computing": "Computing metrics...",
        "duration": "Duration",
        "sample_rate": "Sample Rate",
        "channels": "Channels",
        "loudness": "Loudness & Headroom",
        "per_channel": "Per-Channel Metrics",
        "stereo_analysis": "Stereo Analysis",
        "issues": "Issues & Recommendations",
        "plots": "Diagnostic Plots",
        "downloads": "Download Reports",
        "no_issues": "No significant issues detected.",
        "complete": "Analysis complete.",
        "category": "Category",
        "frequency_range": "Frequency Range",
        "evidence": "Evidence",
        "impact": "Impact",
        "confidence": "Confidence",
        "suggestions": "Suggestions",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "ideal": "Ideal",
        "waveform": "Waveform",
        "spectrum": "Spectrum",
        "mid_side": "Mid/Side",
        "loudness_timeline": "Loudness Timeline",
        "audio_player": "Interactive Audio Player",
        "playback_mode": "Playback Mode",
        "stereo": "Stereo",
        "mono": "Mono",
        "mono_loss": "Mono Energy Loss",
        "phase_safe": "Phase Safe",
        "mastering_metrics": "Mastering Metrics",
        "loudness_details": "Loudness Details",
        "dc_offset": "DC Offset Analysis",
        "aliasing": "Aliasing Detection",
        "mono_check": "Mono Compatibility Check",
        "status": "Status",
        "severity": "Severity",
        "detected": "Detected",
        "none": "None",
        "issue": "Issue",
        "ok": "OK",
        "safe_for_mono": "Safe for Mono",
        "phase_issue": "Phase Issue",
        "yes": "YES",
        "no": "NO",
        "json_report": "Download JSON Report",
        "html_report": "Download HTML Report",
        "technical_details": "Technical details",
        "error": "Error during analysis",
        "error_checks": "Please check that the file is a valid WAV/AIFF and is not corrupted.",
        "footer": "Free technical analysis tool by Alexandre Lack for mix review and master validation. Not a substitute for professional mastering engineering.",
    },
    "pt": {
        "language": "Idioma",
        "mode": "Modo de análise",
        "mix_mode": "Análise de Mix",
        "master_mode": "Validação de Master",
        "title": "MMTools by Alexandre Lack",
        "subtitle": "Análise de mix e validação de master para arquivos WAV/AIFF",
        "about_title": "Sobre",
        "about": """**MMTools** analisa mixes antes da masterização e valida masters finais.

**Análise de Mix**
- Loudness e headroom
- Correlação estéreo e delay
- Checagens de frequência
- Problemas e recomendações

**Validação de Master**
- LUFS, true peak, RMS
- DC offset e aliasing
- Compatibilidade mono
- Player estéreo/mono

**Relatórios:** JSON e HTML""",
        "upload": "Enviar Arquivo de Áudio",
        "upload_help": "Escolha um arquivo WAV ou AIFF para análise",
        "analyzing": "Analisando áudio... Aguarde.",
        "computing": "Calculando métricas...",
        "duration": "Duração",
        "sample_rate": "Sample Rate",
        "channels": "Canais",
        "loudness": "Loudness & Headroom",
        "per_channel": "Métricas por Canal",
        "stereo_analysis": "Análise Estéreo",
        "issues": "Problemas & Recomendações",
        "plots": "Gráficos Diagnósticos",
        "downloads": "Baixar Relatórios",
        "no_issues": "Nenhum problema significativo detectado.",
        "complete": "Análise concluída.",
        "category": "Categoria",
        "frequency_range": "Faixa de Frequência",
        "evidence": "Evidência",
        "impact": "Impacto",
        "confidence": "Confiança",
        "suggestions": "Sugestões",
        "high": "Alta",
        "medium": "Média",
        "low": "Baixa",
        "ideal": "Ideal",
        "waveform": "Forma de Onda",
        "spectrum": "Espectro",
        "mid_side": "Mid/Side",
        "loudness_timeline": "Linha de Loudness",
        "audio_player": "Player Interativo",
        "playback_mode": "Modo de Reprodução",
        "stereo": "Estéreo",
        "mono": "Mono",
        "mono_loss": "Perda de Energia em Mono",
        "phase_safe": "Fase Segura",
        "mastering_metrics": "Métricas de Masterização",
        "loudness_details": "Detalhes de Loudness",
        "dc_offset": "Análise de DC Offset",
        "aliasing": "Detecção de Aliasing",
        "mono_check": "Compatibilidade Mono",
        "status": "Status",
        "severity": "Severidade",
        "detected": "Detectado",
        "none": "Nenhum",
        "issue": "Problema",
        "ok": "OK",
        "safe_for_mono": "Seguro em Mono",
        "phase_issue": "Problema de Fase",
        "yes": "SIM",
        "no": "NÃO",
        "json_report": "Baixar Relatório JSON",
        "html_report": "Baixar Relatório HTML",
        "technical_details": "Detalhes técnicos",
        "error": "Erro durante a análise",
        "error_checks": "Verifique se o arquivo é WAV/AIFF válido e não está corrompido.",
        "footer": "Ferramenta gratuita de análise técnica by Alexandre Lack para análise de mix e validação de master. Não substitui engenharia de masterização profissional.",
    },
}

ISSUE_PT = {
    "dc_offset_l": "DC offset detectado no canal esquerdo",
    "dc_offset_r": "DC offset detectado no canal direito",
    "clipping_l": "Clipping digital detectado no canal esquerdo",
    "clipping_r": "Clipping digital detectado no canal direito",
    "true_peak_high": "True Peak acima do limite de -1 dBTP",
    "low_headroom": "Headroom limitado para masterização",
    "low_correlation_global": "Baixa correlação global L/R",
    "channel_delay": "Possível delay relativo entre canais",
    "infrasonic_content": "Conteúdo infrassônico elevado (<20 Hz)",
    "ultrasonic_content": "Conteúdo ultrassônico elevado próximo ao Nyquist",
    "hum_50hz": "Possível hum de 50 Hz e harmônicos",
    "hum_60hz": "Possível hum de 60 Hz e harmônicos",
    "transient_outliers": "Picos transientes isolados detectados",
    "pre_ringing": "Possível pre-ringing antes dos transientes",
}

CATEGORY_PT = {
    "amplitude": "amplitude",
    "loudness": "loudness",
    "stereo": "estéreo",
    "frequency": "frequência",
    "transients": "transientes",
}


def ui_text(key):
    return TEXT[st.session_state.get("language", "pt")][key]


def find_logo():
    for logo_path in LOGO_CANDIDATES:
        if logo_path.exists():
            return logo_path
    return None


def issue_title(issue):
    if st.session_state.get("language") == "pt":
        return ISSUE_PT.get(issue.id, issue.title)
    return issue.title


def issue_category(issue):
    if st.session_state.get("language") == "pt":
        return CATEGORY_PT.get(issue.category, issue.category)
    return issue.category


def channel_name(index):
    names = {
        "en": ["Left", "Right"],
        "pt": ["Esquerdo", "Direito"],
    }[st.session_state.get("language", "pt")]
    return names[index] if index < len(names) else f"Ch{index + 1}"


def save_upload(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def load_and_analyze(tmp_path):
    loaded = audio_io.load_audio(tmp_path)
    audio_data = loaded.samples
    sample_rate = loaded.sample_rate
    channels = loaded.channel_count

    ch_metrics = []
    for ch in range(channels):
        channel_metrics = metrics.analyze_channel_metrics(audio_data, ch)
        ch_metrics.append(
            {
                "peak": channel_metrics.peak_dbfs,
                "rms": channel_metrics.rms_dbfs,
                "dc": channel_metrics.dc_offset,
                "cf": channel_metrics.crest_factor,
            }
        )

    loud_metrics = loudness.analyze_loudness(audio_data, sample_rate)
    stereo_metrics = stereo.analyze_stereo(audio_data, sample_rate) if channels > 1 else None
    spectrum_metrics = spectrum.analyze_spectrum(audio_data, sample_rate)
    freqs, psd_db = spectrum.compute_spectrum(audio_data, sample_rate)
    mono_audio = np.mean(audio_data, axis=1) if channels > 1 else audio_data[:, 0]
    transient_metrics = transients.analyze_transients(mono_audio, sample_rate)
    detected_issues = issues.detect_issues(audio_data, sample_rate)
    mastering_report = mastering.generate_mastering_report(
        audio_data, sample_rate, ch_metrics, loud_metrics
    )

    return {
        "loaded": loaded,
        "audio": audio_data,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": loaded.duration,
        "ch_metrics": ch_metrics,
        "loud_metrics": loud_metrics,
        "stereo_metrics": stereo_metrics,
        "spectrum_metrics": spectrum_metrics,
        "freqs": freqs,
        "psd_db": psd_db,
        "transient_metrics": transient_metrics,
        "issues": detected_issues,
        "mastering_report": mastering_report,
    }


def show_audio_info(result):
    col1, col2, col3 = st.columns(3)
    col1.metric(ui_text("duration"), f"{result['duration']:.2f}s")
    col2.metric(ui_text("sample_rate"), f"{result['sample_rate']} Hz")
    col3.metric(ui_text("channels"), f"{result['channels']}")


def show_player(result):
    audio_data = result["audio"]
    sample_rate = result["sample_rate"]
    channels = result["channels"]

    st.subheader(ui_text("audio_player"))
    player_col1, player_col2, player_col3 = st.columns(3)
    with player_col1:
        playback_mode = st.radio(
            ui_text("playback_mode"),
            [ui_text("stereo"), ui_text("mono")],
            horizontal=True,
        )

    if playback_mode == ui_text("mono"):
        playback_audio = player.convert_to_mono(audio_data)
    else:
        playback_audio = audio_data if audio_data.ndim > 1 else audio_data.reshape(-1, 1)

    max_val = np.max(np.abs(playback_audio))
    if max_val > 1.0:
        playback_audio = playback_audio / max_val * 0.95

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, playback_audio.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    st.audio(wav_buffer, format="audio/wav")

    if channels > 1:
        mono_info = player.analyze_stereo_to_mono(audio_data)
        player_col2.metric(ui_text("mono_loss"), f"{mono_info['energy_loss']:.1f}%")
        player_col3.metric(ui_text("phase_safe"), "OK" if mono_info["phase_safe"] else ui_text("issue"))


def show_mix_metrics(result):
    loud_metrics = result["loud_metrics"]
    loud = {
        "lufs": loud_metrics.lufs_integrated,
        "true_peak": loud_metrics.true_peak_dbtp,
        "headroom": -1.0 - loud_metrics.true_peak_dbtp,
        "peak_dbfs": loud_metrics.peak_dbfs,
    }

    st.subheader(ui_text("loudness"))
    col1, col2, col3 = st.columns(3)
    col1.metric("LUFS", f"{loud['lufs']:.1f}", delta=f"{loud['lufs'] - (-23):.1f} (target -23)")
    col2.metric("True Peak", f"{loud['true_peak']:.1f} dBTP")
    col3.metric("Headroom", f"{loud['headroom']:.1f} dB", delta=f"{ui_text('ideal')}: >1.0 dB")

    st.markdown("---")
    st.subheader(ui_text("per_channel"))
    for ch_idx, metric in enumerate(result["ch_metrics"]):
        col1, col2, col3, col4 = st.columns(4)
        name = channel_name(ch_idx)
        col1.metric(f"{name} - Peak", f"{metric['peak']:.2f} dBFS")
        col2.metric(f"{name} - RMS", f"{metric['rms']:.2f} dBFS")
        col3.metric(f"{name} - DC Offset", f"{metric['dc']:+.4f}")
        col4.metric(f"{name} - Crest Factor", f"{metric['cf']:.2f} dB")

    stereo_metrics = result["stereo_metrics"]
    if result["channels"] > 1 and stereo_metrics:
        st.markdown("---")
        st.subheader(ui_text("stereo_analysis"))
        col1, col2 = st.columns(2)
        col1.metric("L/R Correlation", f"{stereo_metrics.correlation_global:.3f}", delta=f"{ui_text('ideal')}: >0.8")
        col2.metric("L/R Delay", f"{stereo_metrics.delay_ms:.2f} ms", delta=f"{ui_text('ideal')}: <5 ms")


def show_mastering_metrics(result):
    mastering_report = result["mastering_report"]
    st.subheader(ui_text("mastering_metrics"))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LUFS", f"{mastering_report.lufs:.1f}", delta=f"{mastering_report.lufs - (-14):.1f} (target -14)")
    col2.metric("True Peak", f"{mastering_report.true_peak:.1f} dBTP")
    col3.metric("RMS", f"{mastering_report.rms:.1f} dBFS")
    col4.metric("Headroom", f"{mastering_report.headroom_db:.1f} dB")

    col1, col2, col3 = st.columns(3)
    col1.metric("Short-Term LUFS", f"{mastering_report.lufs_short_term:.1f}")
    col2.metric("Peak/Loudness", f"{mastering_report.peak_to_loudness_ratio:.1f} dB")
    col3.metric("Dynamic Range", f"{mastering_report.dynamic_range:.2f} LU")

    st.markdown("---")
    st.subheader(ui_text("dc_offset"))
    col1, col2, col3 = st.columns(3)
    col1.metric("DC Offset L", f"{mastering_report.dc_offset_l:+.4f}")
    col2.metric("DC Offset R", f"{mastering_report.dc_offset_r:+.4f}")
    col3.metric(ui_text("status"), ui_text("issue") if mastering_report.dc_offset_issue else ui_text("ok"))

    st.markdown("---")
    st.subheader(ui_text("aliasing"))
    col1, col2, col3 = st.columns(3)
    col1.metric(ui_text("status"), ui_text("detected") if mastering_report.aliasing_detected else ui_text("none"))
    col2.metric(ui_text("severity"), mastering_report.aliasing_severity.upper())
    col3.metric(ui_text("confidence"), f"{mastering_report.aliasing_confidence * 100:.0f}%")

    st.markdown("---")
    st.subheader(ui_text("mono_check"))
    col1, col2, col3 = st.columns(3)
    col1.metric(ui_text("safe_for_mono"), ui_text("yes") if mastering_report.mono_compatible else ui_text("no"))
    col2.metric(ui_text("phase_issue"), ui_text("yes") if mastering_report.mono_phase_issue else ui_text("no"))
    col3.metric(ui_text("mono_loss"), f"{mastering_report.mono_energy_loss:.1f}%")


def show_issues(detected_issues):
    st.subheader(ui_text("issues"))
    high_count = sum(1 for issue in detected_issues if issue.severity == "high")
    med_count = sum(1 for issue in detected_issues if issue.severity == "medium")
    low_count = sum(1 for issue in detected_issues if issue.severity == "low")

    col1, col2, col3 = st.columns(3)
    col1.metric(f"High / {ui_text('high')}", high_count)
    col2.metric(f"Medium / {ui_text('medium')}", med_count)
    col3.metric(f"Low / {ui_text('low')}", low_count)

    if not detected_issues:
        st.success(ui_text("no_issues"))
        return

    for severity, marker in [("high", "High"), ("medium", "Medium"), ("low", "Low")]:
        severity_issues = [issue for issue in detected_issues if issue.severity == severity]
        if not severity_issues:
            continue
        st.markdown(f"### {marker} / {ui_text(severity)}")
        for issue in severity_issues:
            with st.expander(f"**{issue_title(issue)}**"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**{ui_text('category')}:** {issue_category(issue)}")
                    if issue.frequency_range:
                        st.write(f"**{ui_text('frequency_range')}:** {issue.frequency_range}")
                    st.write(f"**{ui_text('evidence')}:** {issue.evidence}")
                    st.write(f"**{ui_text('impact')}:** {issue.impact}")
                with col2:
                    st.metric(ui_text("confidence"), f"{issue.confidence * 100:.0f}%")
                st.markdown(f"**{ui_text('suggestions')}:**")
                for suggestion in issue.suggestions:
                    st.write(f"- {suggestion}")


def show_plots(result, include_loudness_timeline=False):
    audio_data = result["audio"]
    sample_rate = result["sample_rate"]
    channels = result["channels"]

    st.subheader(ui_text("plots"))
    tab_names = [ui_text("waveform"), ui_text("spectrum"), ui_text("mid_side")]
    if include_loudness_timeline:
        tab_names.append(ui_text("loudness_timeline"))
    tabs = st.tabs(tab_names)

    with tabs[0]:
        fig, ax = plt.subplots(figsize=(12, 4))
        decimate_factor = max(1, len(audio_data) // 10000)
        x_axis = np.arange(len(audio_data)) / sample_rate
        for ch in range(channels):
            ch_data = audio_data[:, ch] if channels > 1 else audio_data[:, 0]
            ax.plot(
                x_axis[::decimate_factor],
                ch_data[::decimate_factor],
                label=channel_name(ch),
                alpha=0.7,
                linewidth=0.5,
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(ui_text("waveform"))
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.semilogy(result["freqs"], 10 ** (result["psd_db"] / 10), label="Average", alpha=0.8, linewidth=1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.set_xscale("log")
        ax.set_title(ui_text("spectrum"))
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim([20, sample_rate / 2])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[2]:
        stereo_metrics = result["stereo_metrics"]
        if channels > 1 and stereo_metrics:
            fig, ax = plt.subplots(figsize=(12, 5))
            bands = list(stereo_metrics.energy_mid_by_band.keys())
            x = np.arange(len(bands))
            width = 0.35
            ax.bar(x - width / 2, list(stereo_metrics.energy_mid_by_band.values()), width, label="Mid (L+R)", alpha=0.8)
            ax.bar(x + width / 2, list(stereo_metrics.energy_side_by_band.values()), width, label="Side (L-R)", alpha=0.8)
            ax.set_ylabel("Energy (dB)")
            ax.set_title(ui_text("mid_side"))
            ax.set_xticks(x)
            ax.set_xticklabels(bands, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Mid/Side analysis requires stereo audio.")

    if include_loudness_timeline:
        with tabs[3]:
            fig, ax = plt.subplots(figsize=(12, 5))
            hop_length = max(1, sample_rate // 10)
            timeline = []
            for start in range(0, len(audio_data), hop_length):
                chunk = audio_data[start : start + hop_length]
                rms = np.sqrt(np.mean(chunk**2))
                timeline.append(metrics.dbfs(rms))
            times = np.arange(len(timeline)) / 10
            ax.plot(times, timeline, linewidth=1)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("dBFS")
            ax.set_title(ui_text("loudness_timeline"))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def build_report_dict(result, mode_label):
    mastering_report = result["mastering_report"]
    return {
        "timestamp": datetime.now().isoformat(),
        "mode": mode_label,
        "audio_info": {
            "duration": result["duration"],
            "sample_rate": result["sample_rate"],
            "channels": result["channels"],
        },
        "channels": {
            f"Ch{idx + 1}": {
                "peak_dbfs": channel["peak"],
                "rms_dbfs": channel["rms"],
                "dc_offset": channel["dc"],
                "crest_factor": channel["cf"],
            }
            for idx, channel in enumerate(result["ch_metrics"])
        },
        "loudness": {
            "lufs_integrated": result["loud_metrics"].lufs_integrated,
            "true_peak_dbtp": result["loud_metrics"].true_peak_dbtp,
            "peak_dbfs": result["loud_metrics"].peak_dbfs,
        },
        "mastering": asdict(mastering_report),
        "issues": [issue.__dict__ for issue in result["issues"]],
    }


def build_html_report(result, mode_label):
    report_dict = build_report_dict(result, mode_label)
    mastering_report = result["mastering_report"]
    issue_items = "".join(
        f"<li><strong>{html.escape(issue.severity.upper())}: {html.escape(issue_title(issue))}</strong><br>"
        f"{html.escape(issue_category(issue))} - {html.escape(str(issue.evidence))}</li>"
        for issue in result["issues"]
    ) or f"<li>{html.escape(ui_text('no_issues'))}</li>"

    return f"""<!doctype html>
<html lang="{st.session_state.get('language', 'pt')}">
<head>
  <meta charset="utf-8">
  <title>MMTools Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; background: #f8fafc; }}
    .label {{ color: #627d98; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    li {{ margin-bottom: 12px; }}
  </style>
</head>
<body>
  <h1>MMTools by Alexandre Lack - {html.escape(mode_label)}</h1>
  <p>{html.escape(report_dict["timestamp"])}</p>
  <h2>{html.escape(ui_text("duration"))} / {html.escape(ui_text("sample_rate"))}</h2>
  <div class="grid">
    <div class="card"><div class="label">{html.escape(ui_text("duration"))}</div><div class="value">{result["duration"]:.2f}s</div></div>
    <div class="card"><div class="label">{html.escape(ui_text("sample_rate"))}</div><div class="value">{result["sample_rate"]} Hz</div></div>
    <div class="card"><div class="label">{html.escape(ui_text("channels"))}</div><div class="value">{result["channels"]}</div></div>
  </div>
  <h2>{html.escape(ui_text("mastering_metrics"))}</h2>
  <div class="grid">
    <div class="card"><div class="label">LUFS</div><div class="value">{mastering_report.lufs:.1f}</div></div>
    <div class="card"><div class="label">True Peak</div><div class="value">{mastering_report.true_peak:.1f} dBTP</div></div>
    <div class="card"><div class="label">RMS</div><div class="value">{mastering_report.rms:.1f} dBFS</div></div>
    <div class="card"><div class="label">Headroom</div><div class="value">{mastering_report.headroom_db:.1f} dB</div></div>
  </div>
  <h2>{html.escape(ui_text("issues"))}</h2>
  <ul>{issue_items}</ul>
</body>
</html>
"""


def show_downloads(result, uploaded_name, mode_label):
    st.subheader(ui_text("downloads"))
    report_dict = build_report_dict(result, mode_label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(uploaded_name).stem
    st.download_button(
        label=ui_text("json_report"),
        data=json.dumps(report_dict, indent=2, default=str),
        file_name=f"{stem}_{mode_label.lower().replace(' ', '_')}_{timestamp}.json",
        mime="application/json",
    )
    st.download_button(
        label=ui_text("html_report"),
        data=build_html_report(result, mode_label),
        file_name=f"{stem}_{mode_label.lower().replace(' ', '_')}_{timestamp}.html",
        mime="text/html",
    )


def render_app(default_mode="mix"):
    st.set_page_config(
        page_title="MMTools by Alexandre Lack",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
<style>
    .high-severity { color: #ff4444; font-weight: bold; }
    .medium-severity { color: #ff9933; font-weight: bold; }
    .low-severity { color: #44aa44; font-weight: bold; }
</style>
""",
        unsafe_allow_html=True,
    )

    language_label = st.sidebar.selectbox(
        "Idioma / Language",
        ["Português", "English"],
        index=0,
    )
    st.session_state["language"] = "pt" if language_label == "Português" else "en"
    labels = [ui_text("mix_mode"), ui_text("master_mode")]
    default_mode_index = 1 if default_mode == "master" else 0
    mode_label = st.sidebar.radio(ui_text("mode"), labels, index=default_mode_index)
    is_master_mode = mode_label == ui_text("master_mode")

    logo_path = find_logo()
    if logo_path:
        st.sidebar.image(str(logo_path), use_container_width=True)

    if logo_path:
        st.image(str(logo_path), width=800)
        st.title(ui_text("title"))
        st.markdown(f"**{ui_text('subtitle')}**")
    else:
        st.title(f"🎵 {ui_text('title')}")
        st.markdown(f"**{ui_text('subtitle')}**")
    st.markdown("---")

    st.sidebar.markdown(f"## {ui_text('about_title')}")
    st.sidebar.markdown(ui_text("about"))

    uploaded_file = st.file_uploader(
        ui_text("upload"),
        type=["wav", "aiff", "aif"],
        help=ui_text("upload_help"),
    )

    if uploaded_file is None:
        st.info(ui_text("upload_help"))
        return

    tmp_path = None
    try:
        tmp_path = save_upload(uploaded_file)
        st.info(ui_text("analyzing"))
        with st.spinner(ui_text("computing")):
            result = load_and_analyze(tmp_path)

        show_audio_info(result)
        st.markdown("---")

        if is_master_mode:
            show_player(result)
            st.markdown("---")
            show_mastering_metrics(result)
            st.markdown("---")
            show_plots(result, include_loudness_timeline=True)
        else:
            show_mix_metrics(result)
            st.markdown("---")
            show_plots(result, include_loudness_timeline=False)

        st.markdown("---")
        show_issues(result["issues"])
        st.markdown("---")
        show_downloads(result, uploaded_file.name, mode_label)
        st.success(ui_text("complete"))

    except Exception as error:
        st.error(f"{ui_text('error')}: {error}")
        with st.expander(ui_text("technical_details")):
            st.code(traceback.format_exc())
        st.write(ui_text("error_checks"))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    st.markdown("---")
    st.markdown(
        f"""
<div style='text-align: center; color: #888; font-size: 0.85em;'>
    <p><strong>MMTools by Alexandre Lack v0.2.0</strong></p>
    <p>{html.escape(ui_text("footer"))}</p>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_app()
