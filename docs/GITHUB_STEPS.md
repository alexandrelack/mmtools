# Publicar no GitHub

## 1. Criar o repositório no GitHub

No site do GitHub:

1. Clique em **New repository**.
2. Nome sugerido: `mmtools`.
3. Deixe como **Public** se quiser disponibilizar para o público.
4. Não marque README, `.gitignore` ou license, porque o projeto já tem esses arquivos.
5. Clique em **Create repository**.

## 2. Preparar a pasta local

No Terminal:

```bash
cd "/Volumes/ALE SSD 1T/LABORATORIO DE APPS/premaster-inspector-web"
git init
git add .
git status
```

Veja se aparecem principalmente arquivos como `app.py`, `README.md`, `requirements.txt`, `assets/logo.png`, `docs/GITHUB_STEPS.md` e `mmtools-audio/`.

## 3. Primeiro commit

```bash
git commit -m "Initial MMTools release"
git branch -M main
```

## 4. Conectar ao GitHub

Troque `SEU-USUARIO` pelo seu usuário do GitHub:

```bash
git remote add origin https://github.com/SEU-USUARIO/mmtools.git
git push -u origin main
```

## 5. Depois das próximas mudanças

```bash
git status
git add .
git commit -m "Update MMTools"
git push
```

## Checklist Antes de Publicar

- `README.md` aparece na raiz.
- `assets/logo.png` está presente.
- Áudios pessoais não aparecem no `git status`.
- Pastas de relatório não aparecem no `git status`.
- Testes passam com `python3 -m pytest tests -q` dentro de `mmtools-audio`.
