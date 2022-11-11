import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression #Regressão linear
from sklearn import metrics #Cálculo do erro
from sklearn import tree

st.title('Tópicos Especiais em Informática - Trabalho')

#preparando base
database_path = 'base-teste.csv' #caminho da base de dados
database_path2 = 'base-teste2.csv' #caminho da base de dados
df = pd.read_csv(database_path, sep=",")
df_gols = pd.read_csv(database_path2, sep=",")

#limpando base
df_full = df.dropna(subset=df.select_dtypes(int).columns, how='all')
df_full = df_full[df_full["chutes"] != 0]
df_full = df_full[df_full["precisao_passes"] != "None"]

#criando arvore
quali = ['clube'] #Variáveis qualitativas
quant = ['chutes', 'chutes_no_alvo'] #Variáveis quantitativas
dfQualiDummies = pd.get_dummies(df_full[quali]) #Dataframe com qualitativas dummy
dfQuant = df_full[quant] #Dataframe com quantitativas
dfWork = pd.concat([dfQualiDummies, dfQuant ], axis=1 ) #Dataframe com quali dummy e quant
target = df_full['posse_de_bola']
arv = tree.DecisionTreeClassifier() #árvore de classificação
arv.fit(dfWork, target)
clubes_df = ["America-MG","Athletico-PR","Atletico-GO","Atletico-MG","Avai","Bahia","Botafogo-RJ","Bragantino","CSA","Ceara","Chapecoense","Corinthians","Coritiba","Cruzeiro","Cuiaba",
          "Flamengo","Fluminense","Fortaleza","Goias","Gremio","Internacional","Juventude","Palmeiras","Parana","Santos","Sao Paulo","Sport","Vasco","Vitoria"]

#coletando informações do usuario
#Clube
st.write('Escolha um time para a análise')
clube_user = st.selectbox('opções: ', clubes_df)
st.write(f'- CLUBE escolhido: {clube_user}')
vals = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(0,29):
    clube_lower = clubes_df[i].lower()
    clube_user_lower = clube_user.lower()
    if clube_user_lower == clube_lower:
        clube_user2 = clubes_df[i]
        vals[i] = 1
        st.write('Ok')
df_user = df_full[df_full["clube"] == clube_user2]
#Chutes
st.write('Insira um número de chutes')
chutes_min = int(df_user["chutes"].min())
chutes_max = int(df_user["chutes"].max())
chutes_user = st.slider("Selecione: ",chutes_min,chutes_max)
st.write(f'Chutes: {chutes_user}')
#Chutes no alvo
st.write('Insira um número de chutes no alvo')
chutes_alvo_min = int(df_user["chutes_no_alvo"].min())
chutes_alvo_max = int(df_user["chutes_no_alvo"].max())
chutes_alvo_user = st.slider("Selecione: ",chutes_alvo_min,chutes_alvo_max)
st.write(f'Chutes no alvo: {chutes_alvo_user}')
#upando resultados
vals[29] = chutes_user
vals[30] = chutes_alvo_user
st.write('Ok2')

#predição
res = arv.predict([vals])
resul = str(res[0])
st.write('De acordo com a analise, seu time nesses parametros possui uma porcentagem de: ',resul," em posse de bola")

#avaliando partidas
df_user = df_full[df_full["clube"] == clube_user2]
df_full_resul = df_user[df_user["posse_de_bola"]==resul]
id_partida = df_full_resul["partida_id"].values
st.write(df_full_resul)

#analise final
empate = 0 
vitoria = 0
derrota = 0
for i in range(0,len(id_partida)):
#   print("partida:", i)
  contador_gols_time = 0
  contador_gols_rival = 0 
  partida = df_gols[df_gols["partida_id"] == id_partida[i]]
#   print(partida)
  gols = partida["clube"].values
  for p in range(0,len(gols)):
    if gols[p] == clube_user2:
      contador_gols_time +=1 
    else:
      contador_gols_rival +=1
  if contador_gols_time == contador_gols_rival:
    empate += 1
    #print("partida", i ,"resultado empate: ", empate)
  elif contador_gols_time > contador_gols_rival:
    vitoria += 1
    #print("partida", i ,"resultado vitoria: ", vitoria)
  elif contador_gols_time < contador_gols_rival:
    derrota += 1
    #print("partida", i ,"resultado derrota: ", derrota)
  else:
    st.write('Erro2')
st.write('vitorias: ',vitoria)
st.write('empate: ',empate)
st.write('derrota: ',derrota)
if vitoria == 0 and derrota == 0 and empate == 0:
    st.write("Nesse cenário, nao podemos prever (sem experiencias com essa porcentagem de posse de bola)")
elif vitoria > derrota and vitoria == empate:
    st.write("Nesse cenário, seu time provavelmente ganharia ou empataria")
elif derrota > vitoria and derrota == empate:
    st.write("Nesse cenário, seu time provavelmente perderia ou empataria")
elif empate > vitoria and empate > derrota:
    st.write("Nesse cenário, seu time provavelmente empataria")
elif vitoria > empate and vitoria > derrota:
    st.write("Nesse cenário, seu time provavelmente ganharia")
elif derrota > empate and derrota > vitoria:
    st.write("Nesse cenário, seu time provavelmente perderia")
elif derrota == empate and derrota == vitoria:
    st.write("Nesse cenário, seu time pode tanto ganhar, empatar ou perder")
else:
    st.write('Erro3')




