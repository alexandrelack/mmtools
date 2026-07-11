# MMTools by Alexandre Lack

Ferramenta gratuita para **análise de mix** e **validação de master** em arquivos WAV/AIFF.

O app reúne dois fluxos em uma interface só:

- **Análise de Mix:** loudness, headroom, clipping, DC offset, estéreo, espectro e recomendações.
- **Validação de Master:** LUFS, true peak, RMS, DC offset, aliasing, compatibilidade mono e player estéreo/mono.
- **Relatórios:** download em JSON e HTML.

## Rodar Localmente

```bash
cd caminho/para/mmtools
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Depois abra o endereço mostrado no terminal, geralmente `http://localhost:8501`.


## Atalho no Mac

Depois de instalar as dependências, você também pode abrir pelo arquivo:

```text
RUN_MMTOOLS.command
```

## Estrutura

```text
.
├── app.py                  # App Streamlit principal
├── mastering_app.py        # Atalho legado para abrir direto em Master
├── requirements.txt        # Dependências do app
├── assets/                 # Logo e arquivos visuais
├── docs/                   # Guia curto para publicar no GitHub
└── mmtools-audio/          # Pacote Python com o motor de análise e testes
```

## Testes

```bash
cd mmtools-audio
python3 -m pytest tests -q
```

## Aviso

MMTools ajuda a identificar sinais técnicos em mixes e masters, mas não substitui escuta crítica nem engenharia profissional de masterização.
