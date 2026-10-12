# Análise Multimodal de EEG e Eye Tracking na Classificação do Transtorno do Espectro do Austimo (TEA)

Este repositório contém o pipeline computacional e os dados de validação do projeto de mestrado desenvolvido no Instituto Santos Dumont (ISD) pelo Programa de Pós-Graduação em Neuroengenharia (PPgN). 

A pesquisa investiga o processamento socioemocional em crianças com TEA através da análise topográfica de sinais de EEG (16 canais) frente a estímulos faciais de diferentes valências.

### Principais Achados
A aplicação de algoritmos de Aprendizado de Máquina aos dados de Potência Relativa revelou assinaturas neurobiológicas distintas no processamento da Face Feliz:
* Desempenho Preditivo: O modelo de aprendizado de máquina alcançou ~86.0% de acurácia (Validação 5-Fold) na classificação entre TEA e Controle.
* Desequilíbrio Excitatório/Inibitório: A extração de características (Feature Importance e SHAP Values) associada aos mapas topográficos (d de Cohen) confirmou uma hipoativação global da banda Alpha (déficit inibitório) e uma hiperativação compensatória da banda Gamma nas regiões temporais e frontais no grupo TEA.

---
### 📁 Estrutura do Projeto

```text
eeg-asd-classification/
├── .devcontainer/        # Configurações do ambiente de desenvolvimento (VS Code)
├── data/                 # Armazenamento de dados do EEG
│   ├── interim/          # Dados intermediários durante o processamento
│   └── processed/        # Épocas finais prontas para extração
├── notebooks/            # Pipeline computacional sequencial
│   ├── 01_Structuring_eeg.ipynb
│   ├── 02_Preprocessing_eeg.ipynb
│   ├── 03_Features_eeg.ipynb
│   ├── 04_ML_eeg.ipynb
│   └── 05_Plots_eeg.ipynb
│   ├── 06_Structuring_et.ipynb
│   ├── 07_Preprocessing_et.ipynb
│   ├── 08_Features_et.ipynb
│   ├── 09_ML_et.ipynb
│   └── 10_Plots_et.ipynb
│   └── 11_Multimodal_fusion.ipynb
├── reports/              # Saída automática de gráficos e mapas topográficos
├── results/              # Tabelas e métricas brutas exportadas pelo modelo
├── Dockerfile            # Receita para construção do contêiner isolado
├── docker-compose.yml    # Orquestrador para inicialização do ambiente reprodutível
├── requirements.txt      # Dependências com versionamento matemático estrito
├── README.md             # Visão geral e documentação do projeto
└── .gitignore            # Omissão de arquivos temporários do controle de versão
```
---
### Reproduzir este Estudo (Ambiente Docker)
Para garantir a integridade matemática e evitar conflitos de versão entre bibliotecas, este projeto foi encapsulado em um contêiner Docker. Nenhuma instalação local de bibliotecas Python é necessária.

1. Certifique-se de ter o Docker e o Docker Compose instalados em sua máquina.
2. Clone este repositório e navegue até a pasta raiz no seu terminal.
3. Execute o comando de construção e inicialização:
   ```bash
   docker-compose up
4. O processo fará o download do sistema operacional isolado e instalará as dependências exatas. Ao final, acesse o link do localhost fornecido no terminal.
5. Quando o Jupyter Lab solicitar a senha de acesso para os notebooks, digite: **isd2026**

Autor: Bruno Oswando

Mestrando em Neuroengenharia - Instituto Santos Dumont (ISD / PPgN)
