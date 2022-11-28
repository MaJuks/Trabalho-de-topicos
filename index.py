import pandas as pd
import streamlit as st
from sklearn import tree
# INTRODUÇÃO:
# - TECNOLOGIAS (PANDAS, STREAMLIT, SKLEARN (tree))


# A base de dados escolhida reúne as principais estatísticas sobre as partidas realizadas no campeonato Braliseiro de futebol (Brasileirão), 
# possui um total de 8025 partidas dos anos de 2003 a 2022, sendo continuado até o momento no repositório no github:
# https://github.com/adaoduque/Brasileirao_Dataset
# https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol

# Principais colunas usadas: 
# ID ou partida_ID - ID da partida 
# Data : Data que ocorreu a partida
# Mandante : Clube mandante                      
# Visitante : Clube Visitante
# Vencedor : Clube vencedor da partida. Quando tiver "-", é um empate
# Mandante Placar : Gols que o clube mandante fez na partida                      
# Visitante Placar : Gols que o clube visitante fez na partida                      
# Clube - Nome do clube
# Chutes - Finalizações
# Chutes a gol - Finalizações na direção do gol
# Posse de bola - Percentual da posse de bola


st.title('Tópicos Especiais em Informática - Trabalho')

# Carregando a base
database1= 'base-teste.csv' # Caminho da base de dados
database3 = 'base-teste3.csv' # Caminho da base de dados
df_estatistica = pd.read_csv(database1, sep=",") # Dataset estatistico do clube na partida (encanteio, cartão, ...)
df_partidas = pd.read_csv(database3, sep=",") # Dataset estatistico da partida (arena, mandante, placar, ...)


# Limpando base estatística (partidas que não aconteceram ou são muito antigas (poucas informações))  2992 rows × 13 columns
df_estatistica= df_estatistica.dropna(subset=df_estatistica.select_dtypes(int).columns, how='all')
df_estatistica= df_estatistica[df_estatistica["chutes"] != 0]
df_estatistica= df_estatistica[df_estatistica["precisao_passes"] != "None"]

# Criando arvore
quali = ['clube'] # Variáveis qualitativas
quant = ['chutes', 'chutes_no_alvo'] # Variáveis quantitativas
dfQualiDummies = pd.get_dummies(df_estatistica[quali]) # Dataframe com qualitativas dummy
dfQuant = df_estatistica[quant] # Dataframe com quantitativas
dfWork = pd.concat([dfQualiDummies, dfQuant ], axis=1 ) # Dataframe com quali dummy e quant
target = df_estatistica['posse_de_bola'] # Variavel alvo da predição (string)
arv = tree.DecisionTreeClassifier() # Arvore de classificação 
arv.fit(dfWork, target)

clubes_df = ["America-MG","Athletico-PR","Atletico-GO","Atletico-MG","Avai","Bahia","Botafogo-RJ","Bragantino","CSA","Ceara","Chapecoense","Corinthians","Coritiba","Cruzeiro","Cuiaba",
          "Flamengo","Fluminense","Fortaleza","Goias","Gremio","Internacional","Juventude","Palmeiras","Parana","Santos","Sao Paulo","Sport","Vasco","Vitoria"]

# Coletando informações do usuario
# - Clube
st.header('Faremos uma análise de partidas de futebol com base nos dados do usuário!')
st.write('- ESCOLHA UM TIME PARA A ANÁLISE')
clube_user = st.selectbox('', clubes_df) # Time escolhido do usuário
st.text(f'clube escolhido: {clube_user}')
# Variaveis para a predição zeradas
vals = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #31
tester = 0
for i in range(0,29): # 29 times
    if clube_user == clubes_df[i]:
        vals[i] = 1 # Colocando resultado (time) nas variaveis de predição
        tester = 1
if tester == 0:
    st.write("erro1")
df_user = df_estatistica[df_estatistica["clube"] == clube_user] # Filtrando base de dados com o clube escolhido
# - Chutes - Finalizações
st.write('- INSIRA UM NÚMERO DE CHUTES')
chutes_min = int(df_user["chutes"].min())
chutes_max = int(df_user["chutes"].max())
chutes_user = st.slider("",chutes_min,chutes_max) # Número de chutes escolhidos pelo usuário
st.text(f'quantidade de chutes selecionados: {chutes_user}')
# - Chutes no alvo - Finalizações na direção do gol
st.write('- INSIRA UM NÚMERO DE CHUTES NO ALVO')
chutes_alvo_min = int(df_user["chutes_no_alvo"].min())
chutes_alvo_max = int(df_user["chutes_no_alvo"].max())
chutes_alvo_user = st.slider("",chutes_alvo_min,chutes_alvo_max)
st.text(f'quantidade de chutes no alvo selecionado: {chutes_alvo_user}')  # Número de chutes no alvo escolhidos pelo usuário
# Colocando resultados (chutes e chutes no alvo) nas variaveis de predição
vals[29] = chutes_user
vals[30] = chutes_alvo_user

# Predição
res = arv.predict([vals]) # Predição
resul = str(res[0]) # Coletando resultado em string
st.subheader(f'De acordo com a análise, seu time nesses parâmetros possui uma porcentagem de: {resul} em posse de bola')
st.write("Quantidade de partidas em que seu time possuiu essa porcentagem de posse de bola e seus dados estatísticos")

# Recolhendo ID das partidas
df_estatistica_resul = df_user[df_user["posse_de_bola"]==resul] # Coletando partidas do time escolhido com posse de bola predita
id_partida = df_estatistica_resul["partida_id"].values # ID das partidas
st.write(df_estatistica_resul)
st.subheader("Agora, faremos uma análise dessas partidas, indicando se dentro desses parâmetros calculados ele, clube escolhido, venceria, perderia ou ganharia o jogo!")

# Variáveis da análise final zeradas
st.write('- PARTIDAS: (data, placar)')
contador = 1 
empate = 0 
vitoria = 0
derrota = 0
for i in range (0,len(id_partida)): # Detalhar cada partida
    clube_rival = ""
    pontos_rival = 0
    partida = df_partidas[df_partidas["ID"]== id_partida[i]] # Filtrando partidas em outra base (ID)
    dados = ['data','mandante',"visitante","vencedor","mandante_Placar","visitante_Placar"] # Colunas específicas para a análise
    partida = partida[dados] # Filtrando
    partida_data = partida["data"].values # Datando (recolhendo datas)
    if partida["mandante"].values == clube_user: # Identificando mandante e visitante (recolhendo placar)
        clube_rival = partida["visitante"].values
        pontos_rival = int(partida["visitante_Placar"].values)
        pontos_user = int(partida["mandante_Placar"].values)
    else:
        clube_rival = partida["mandante"].values
        pontos_rival = int(partida["mandante_Placar"].values)
        pontos_user = int(partida["visitante_Placar"].values)
    if partida["vencedor"].values == clube_user: # Identificando vencedor
        vitoria += 1
    elif partida["vencedor"].values == "-":
        empate += 1 
    else:
        derrota += 1 
    st.write(f"{contador} - {partida_data[0]} --- {clube_user} {pontos_user} x {pontos_rival} {clube_rival[0]}") #apresentando dados
    contador +=1

# Apresentando dados
st.write('- DADOS DAS PARTIDAS:')
st.write('vitorias: ',vitoria)
st.write('empate: ',empate)
st.write('derrota: ',derrota)

# Plotando Gráfico
data = pd.DataFrame({
    
     'x': ['VITÓRIA', 'DERROTA', 'EMPATE'],
          'Resultado': [vitoria, derrota, empate],
 }).set_index('x')
st.bar_chart(data)

# Aprensentando resultado final
if vitoria == 0 and derrota == 0 and empate == 0: #invalido
    st.subheader("Nesse cenário, nao podemos prever (sem experiencias com essa porcentagem de posse de bola)")
elif vitoria > derrota and vitoria == empate: # vitoria == empate > derrota
    st.subheader("Nesse cenário, seu time provavelmente ganharia ou empataria")
elif derrota > vitoria and derrota == empate: #  derrota == empate > vitoria
    st.subheader("Nesse cenário, seu time provavelmente perderia ou empataria")
elif derrota == vitoria and derrota > empate: # vitoria == derrota > empate 
    st.subheader("Nesse cenário, seu time pode possuir ambos resultados (vitória ou derrota)")
elif empate > vitoria and empate > derrota: # empate > vitoria e derrota
    st.subheader("Nesse cenário, seu time provavelmente empataria")
elif vitoria > empate and vitoria > derrota: # vitoria > empate e derrota
    st.subheader("Nesse cenário, seu time provavelmente ganharia")
elif derrota > empate and derrota > vitoria: # derrota > empate e vitoria
    st.subheader("Nesse cenário, seu time provavelmente perderia")
elif derrota == empate and derrota == vitoria: # derrota == empate == vitoria
    st.subheader("Nesse cenário, seu time pode tanto ganhar, empatar ou perder")
else:
    st.write('Erro3')
