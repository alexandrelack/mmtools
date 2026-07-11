# MMTools Audio Engine

Pacote Python usado pelo app **MMTools by Alexandre Lack**.

Ele concentra o motor de análise:

- carregamento WAV/AIFF;
- peak, RMS, DC offset e clipping;
- LUFS e true peak;
- correlação estéreo, Mid/Side e delay L/R;
- espectro, hum, conteúdo infrassônico/ultrassônico;
- validações de master como aliasing e compatibilidade mono;
- geração de relatório JSON/HTML via CLI.

## Instalação Local

```bash
pip install -e .
```

## CLI

```bash
mmtools analyze input.wav --out reports/minha_analise
```

## Testes

```bash
python3 -m pytest tests -q
```

## Nota

O nome interno do módulo ainda é `premaster_inspector` para evitar quebrar imports existentes. A marca pública do app é **MMTools by Alexandre Lack**.
