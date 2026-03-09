import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import kagglehub
import os
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("shazadudwadia/supermarket")
arquivos = [f for f in os.listdir(path) if f.endswith('.csv')]
file_path = os.path.join(path, arquivos[0])

df_raw = pd.read_csv(file_path, header=None)

transactions = []
for i in range(len(df_raw)):
    linha_completa = ",".join([str(x) for x in df_raw.values[i] if str(x) != "nan"])
    itens = [item.strip().upper() for item in linha_completa.split(",") if item.strip()]
    if itens:
        transactions.append(itens)
    
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

def run_apriori(dados, suporte):
    inicio = time.time()
    resultado = apriori(dados, min_support=suporte, use_colnames=True)
    fim = time.time()
    return resultado, fim - inicio

def run_fpgrowth(dados, suporte):
    inicio = time.time()
    resultado = fpgrowth(dados, min_support=suporte, use_colnames=True)
    fim = time.time()
    return resultado, fim - inicio

suporte_minimo = 0.05 
resultado_apriori, tempo_ap = run_apriori(df, suporte_minimo)
resultado_fpgrowth, tempo_fp = run_fpgrowth(df, suporte_minimo)

print("-" * 30)
print(f"Benchmark (20 transacoes):")
print(f"Apriori:   {tempo_ap:.4f}s")
print(f"FP-Growth: {tempo_fp:.4f}s")

def normalize_df(df_patterns):
    df_copy = df_patterns.copy()
    df_copy['itemsets_key'] = df_copy['itemsets'].apply(lambda x: ', '.join(sorted(list(x))))
    return df_copy.sort_values(by=['support', 'itemsets_key']).reset_index(drop=True)

res_ap_norm = normalize_df(resultado_apriori)
res_fp_norm = normalize_df(resultado_fpgrowth)

if res_ap_norm[['support', 'itemsets_key']].equals(res_fp_norm[['support', 'itemsets_key']]):
    print("Consistencia: OK")
else:
    print("Consistencia: Erro de ordenacao")

regras = association_rules(resultado_fpgrowth, metric="confidence", min_threshold=0.1)

print("-" * 30)
if not regras.empty:
    print(f"Sucesso! {len(regras)} regras encontradas:")
    regras["antecedents"] = regras["antecedents"].apply(lambda x: ', '.join(list(x)))
    regras["consequents"] = regras["consequents"].apply(lambda x: ', '.join(list(x)))
    print(regras.sort_values(by="lift", ascending=False)[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
else:
    print("Itens mais frequentes (sem regras de associacao):")
    print(resultado_fpgrowth.sort_values(by='support', ascending=False).head(5))
    
# Gerar gráfico de comparação de tempos
algoritmos = ['Apriori', 'FP-Growth']
tempos = [tempo_ap, tempo_fp]

plt.figure(figsize=(8, 5))
cores = ['skyblue', 'salmon']
bars = plt.bar(algoritmos, tempos, color=cores)

plt.ylabel('Tempo de Execução (segundos)')
plt.title('Comparação de Performance: Apriori vs FP-Growth')
plt.bar_label(bars, fmt='%.4fs')

plt.savefig('comparacao_performance.png')
print("\n[Gráfico 'comparacao_performance.png' gerado com sucesso!]")
plt.show()