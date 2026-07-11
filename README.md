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

## Logo

Coloque sua logo em `assets/` com um destes nomes:

- `logo.png`
- `logo.svg`
- `logo.jpg`
- `logo.jpeg`
- `logo.webp`

O app detecta automaticamente e mostra a logo no topo e na lateral.

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

## Publicar no GitHub

Veja o passo a passo curto em [`docs/GITHUB_STEPS.md`](docs/GITHUB_STEPS.md).

## Subir no GitHub

```bash
git init
git add .
git commit -m "Initial MMTools release"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/mmtools.git
git push -u origin main
```

Antes do `git add .`, confirme que seus áudios pessoais e relatórios locais não entraram. O `.gitignore` já ignora `.wav`, `.aif`, `.aiff` e pastas de relatório.

## Aviso

MMTools ajuda a identificar sinais técnicos em mixes e masters, mas não substitui escuta crítica nem engenharia profissional de masterização.
