# Comparativo de Performance: Apriori vs. FP-Growth 🛒

Este repositório contém um estudo prático de Machine Learning focado em **Regras de Associação**. O projeto compara a eficiência do algoritmo clássico **Apriori** contra o moderno **FP-Growth**, utilizando o dataset *Market Basket Optimization*.

## 🚀 Tecnologias Utilizadas
* **Python 3.12**
* **Pandas**: Manipulação de dados.
* **Mlxtend**: Implementação dos algoritmos de associação.
* **Kagglehub**: Integração direta com datasets do Kaggle.
* **Matplotlib**: Visualização de resultados.

## 🧠 Contexto e Objetivo
Este projeto compara dois dos algoritmos mais famosos de mineração de regras de associação: Apriori e FP-Growth. O objetivo foi medir a performance (tempo de execução) e validar a integridade dos resultados (consistência) utilizando um dataset de transações de supermercado


## 🛠️ Desafios e Aprendizados

Durante o desenvolvimento, foram resolvidos problemas críticos de engenharia de dados:

|Dificuldade|Solução Aplicada|
|-----------|----------------|
|Ambiente Virtual|Erro de diretório ao tentar ativar o `.venv.` Resolvido com a criação correta do ambiente via `python -m venv` e uso de caminhos relativos.|
|Caminhos Dinâmicos|O `kagglehub` baixa arquivos em pastas com hashes aleatórios. Foi implementada uma varredura de diretório com `os.listdir()` para encontrar o CSV automaticamente.|
|Dados "Sujos"|Itens vindo como strings únicas separadas por vírgula em vez de listas. Utilizado `.split(",")` e `.strip()` para separar e limpar os nomes dos produtos.|
|Regras Inexistentes|Suporte muito alto (5%) resultava em DataFrames vazios. Aplicado ajuste fino do ``min_support`` para 0.5% e 0.1%, permitindo capturar associações em datasets esparsos.|
|Erro de Consistência|Algoritmos retornavam os mesmos itens, mas em ordens diferentes. Implementada a normalização dos itemsets (conversão de frozensets para listas ordenadas e strings) antes da comparação.|
|Erro de Hash (Categorical)|Erro ao tentar ordenar listas no Pandas (``unhashable type: 'list'``). Criada uma ``itemset_key`` (string) para permitir a ordenação e comparação segura dos DataFrames.|

## 📊 Resultados Obtidos

Nos testes realizados, o **FP-Growth** demonstrou uma superioridade clara:
* **Apriori**: ~0.0358s
* **FP-Growth**: ~0.0041s
* **Veredito**: O FP-Growth foi aproximadamente **9x mais rápido**, mantendo a integridade total dos resultados (Consistência: OK).



## 📋 Como Executar

1. Clone o repositório:
   ```bash
   git clone [https://github.com/seu-usuario/associacao-apriori-fpgrowth](https://github.com/seu-usuario/associacao-apriori-fpgrowth)
   ```

2. Crie o ambiente virtual
    ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4. Rode o script principal:
    ```bash
    python3 main.py
    ```