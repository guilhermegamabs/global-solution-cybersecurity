import streamlit as st
import pandas as pd
import joblib
import hashlib
import os

MODELO_PATH = "gs_cyber_20251115-1756.pkl" 
HASH_MODELO_CONHECIDO = "815f75b67b73302e91f57c88853c86a1688a33ff15ddcfc7fcaa93034bd57707" 


def gerar_hash_sha256(filepath):
    """Calcula o hash SHA-256 de um arquivo."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        st.error(f"Erro ao calcular hash: {e}")
        return None

def carregar_modelo_seguro(path, hash_conhecido):
    """Valida a integridade e carrega o modelo e o label encoder."""
    if not os.path.exists(path):
        st.error(f"Arquivo do modelo não encontrado em: {path}")
        return None, None

    # 1. Validar o Hash
    hash_atual = gerar_hash_sha256(path)
    
    if hash_atual is None:
        return None, None
        
    if hash_atual == hash_conhecido:
        st.success(f"VERIFICAÇÃO DE INTEGRIDADE: Hash validado.")
        # 2. Carregar o modelo e o encoder
        try:
            data = joblib.load(path)
            modelo = data["model"]
            le = data["label_encoder"]
            st.write("Modelo e Label Encoder carregados com sucesso.")
            return modelo, le
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo .pkl (joblib): {e}")
            return None, None
    else:
        st.error("ALERTA: O ARQUIVO DO MODELO FOI ADULTERADO!")
        st.code(f"Hash Esperado: {hash_conhecido}\nHash Encontrado: {hash_atual}")
        return None, None

st.set_page_config(layout="wide")
st.title("Classificador de Acidentes")

model, le = carregar_modelo_seguro(MODELO_PATH, HASH_MODELO_CONHECIDO)

if model and le:
    st.header("Preencha os dados para prever a gravidade do acidente:")

    features = ['dia_semana', 'br', 'km', 'fase_dia', 'sentido_via',
                'condicao_metereologica', 'tipo_pista', 'tracado_via', 'latitude',
                'longitude', 'delegacia', 'postos_policiais_PRF']
    
    col1, col2 = st.columns(2)

    with col1:
        dia_semana = st.selectbox("Dia da Semana", 
            ['domingo', 'segunda-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado', 'terça-feira'])
        
        fase_dia = st.selectbox("Fase do Dia", 
            ['Pleno dia', 'Plena Noite', 'Amanhecer', 'Anoitecer'])
        
        condicao_metereologica = st.selectbox("Condição Meteorológica", 
            ['Céu Claro', 'Nublado', 'Chuva', 'Garoa/Chuvisco', 'Sol', 'Nevoeiro/Neblina', 'Ignorado', 'Vento', 'Granizo'])
        
        tracado_via = st.selectbox("Traçado da Via", 
            ['Curva', 'Reta', 'Retorno Regulamentado', 'Não Informado', 'Ponte', 'Interseção de vias', 'Rotatória', 'Desvio Temporário', 'Viaduto', 'Túnel'])
        
        br = st.selectbox("BR", [
            'BR-365', 'BR-101', 'BR-116', 'BR-277', 'BR-290', 'BR-40', 'BR-153', 'BR-230',
            'BR-376', 'BR-356', 'BR-251', 'BR-324', 'BR-402', 'BR-282', 'BR-424', 'BR-343',
            'BR-364', 'BR-285', 'BR-163', 'BR-158', 'BR-381', 'BR-386', 'BR-110', 'BR-60',
            'BR-222', 'BR-470', 'BR-232', 'BR-448', 'BR-287', 'BR-135', 'BR-70', 'BR-262',
            'BR-280', 'BR-316', 'BR-415', 'BR-414', 'BR-174', 'BR-435', 'BR-405', 'BR-393',
            'BR-267', 'BR-20', 'BR-392', 'BR-407', 'BR-354', 'BR-401', 'BR-425', 'BR-369',
            'BR-420', 'BR-242', 'BR-493', 'BR-235', 'BR-476', 'BR-80', 'BR-412', 'BR-50',
            'BR-447', 'BR-423', 'BR-319', 'BR-104', 'BR-488', 'BR-418', 'BR-467', 'BR-427',
            'BR-373', 'BR-30', 'BR-463', 'BR-226', 'BR-304', 'BR-293', 'BR-330', 'BR-428',
            'BR-472', 'BR-10', 'BR-480', 'BR-259', 'BR-465', 'BR-155', 'BR-367', 'BR-308',
            'BR-471', 'BR-361', 'BR-406', 'BR-421', 'BR-452', 'BR-459', 'BR-495', 'BR-146',
            'BR-408', 'BR-429', 'BR-156', 'BR-416', 'BR-210', 'BR-359', 'BR-468', 'BR-482',
            'BR-317', 'BR-469', 'BR-349', 'BR-272', 'BR-487', 'BR-419', 'BR-432', 'BR-377',
            'BR-410', 'BR-426', 'BR-122', 'BR-403', 'BR-404', 'BR-436', 'BR-484', 'BR-473',
            'BR-430', 'BR-498', 'BR-422'
        ])
        
        delegacia = st.selectbox("Delegacia", [
            'DEL10-MG', 'DEL09-BA', 'DEL04-SP', 'DEL01-PR', 'DEL03-RS', 'DEL03-PR',
            'DEL01-DF', 'DEL02-TO', 'DEL01-PE', 'DEL03-PI', 'DEL04-SC', 'DEL01-SC',
            'DEL07-PR', 'DEL08-RJ', 'DEL12-MG', 'DEL01-RJ', 'DEL01-BA', 'DEL05-PI',
            'DEL01-RS', 'DEL03-PE', 'DEL01-PI', 'DEL02-GO', 'DEL02-SC', 'DEL01-ES',
            'DEL01-RN', 'DEL03-RO', 'DEL10-RS', 'DEL02-MS', 'DEL02-PR', 'DEL03-SP',
            'DEL11-MG', 'DEL04-RS', 'DEL05-BA', 'DEL05-GO', 'DEL01-GO', 'DEL05-SP',
            'DEL01-MG', 'DEL02-MG', 'DEL09-SP', 'DEL04-MG', 'DEL01-AC', 'DEL04-MA',
            'DEL06-MG', 'DEL06-RS', 'DEL02-MT', 'DEL05-MG', 'DEL04-RJ', 'DEL04-PR',
            'DEL06-SC', 'DEL16-MG', 'DEL03-PA', 'DEL01-PB', 'DEL04-PE', 'DEL02-ES',
            'DEL06-MT', 'DEL07-SC', 'DEL02-RN', 'DEL09-PR', 'DEL08-SP', 'DEL08-MS',
            'DEL12-RS', 'DEL08-RS', 'DEL03-MG', 'DEL03-MA', 'DEL02-RS', 'DEL01-RO',
            'DEL02-BA', 'DEL01-CE', 'DEL05-MT', 'DEL04-MS', 'DEL08-MG', 'DEL03-SC',
            'DEL03-AL', 'DEL02-SP', 'DEL01-SP', 'DEL08-BA', 'DEL11-RS', 'DEL06-SP',
            'DEL07-RJ', 'DELS05-RS', 'DEL07-RS', 'DEL01-RR', 'DEL05-PE', 'DEL07-MT',
            'DEL13-MG', 'DEL03-RJ', 'DEL07-MG', 'DEL04-RO', 'DEL01-PA', 'DEL05-CE',
            'DEL01-MA', 'DEL02-RO', 'DEL07-SP', 'DEL04-RN', 'DEL05-RJ', 'DEL03-MS',
            'DEL02-DF', 'DEL13-RS', 'DEL09-RS', 'DEL04-BA', 'DEL02-PE', 'DEL17-MG',
            'DEL01-MT', 'DEL04-ES', 'DEL05-SC', 'DEL03-BA', 'DEL10-BA', 'DEL02-SE',
            'DEL09-MS', 'DEL01-SE', 'DEL02-RJ', 'DEL02-PB', 'DEL15-MG', 'DEL01-MS',
            'DEL06-RJ', 'DEL06-GO', 'DEL03-MT', 'DEL05-PR', 'DEL04-MT', 'DEL07-BA',
            'DEL07-GO', 'DEL05-PA', 'DEL01-AL', 'DEL03-ES', 'DEL03-PB', 'DEL05-MS',
            'DEL03-RN', 'DEL06-BA', 'DEL14-MG', 'DEL06-PE', 'DEL08-PR', 'DEL04-GO',
            'DEL02-MA', 'DEL04-PI', 'DEL01-TO', 'DEL06-MS', 'DEL03-GO', 'DEL01-AM',
            'DEL02-AL', 'DEL06-PR', 'DEL03-CE', 'DEL05-MA', 'DEL09-MG', 'DEL02-CE',
            'DEL07-MS', 'DEL01-AP', 'DEL02-PI', 'DEL02-PA', 'DEL04-CE', 'DEL04-PA'
        ])

    with col2:
        sentido_via = st.selectbox("Sentido da Via", ['Crescente', 'Decrescente'])
        
        tipo_pista = st.selectbox("Tipo de Pista", ['Simples', 'Dupla', 'Múltipla'])
        
        km = st.number_input("KM (com casas decimais)", format="%.1f")
        
        latitude = st.number_input("Latitude (ex: -15.7801)", format="%.6f")
        
        longitude = st.number_input("Longitude (ex: -47.9292)", format="%.6f")

        postos_policiais_PRF = st.selectbox("Posto Policial", [
            'Média', 'Alta', 'Baixa'
        ])

    if st.button("Classificar Acidente"):
        dados_input = {
            'dia_semana': dia_semana,
            'br': br,
            'km': km,
            'fase_dia': fase_dia,
            'sentido_via': sentido_via,
            'condicao_metereologica': condicao_metereologica,
            'tipo_pista': tipo_pista,
            'tracado_via': tracado_via,
            'latitude': latitude,
            'longitude': longitude,
            'delegacia': delegacia,
            'postos_policiais_PRF': postos_policiais_PRF
        }
        
        input_df = pd.DataFrame([dados_input], columns=features)
        
        predicao_enc = model.predict(input_df)
        
        predicao_legivel = le.inverse_transform(predicao_enc)
        
        st.success(f"Classificação do Acidente: **{predicao_legivel[0]}**")
        
        st.subheader("Dados de Entrada (DataFrame):")
        st.dataframe(input_df)

else:
    st.error("APLICAÇÃO PARADA. O modelo não pôde ser carregado com segurança.")
    st.warning("Verifique se o nome do arquivo .pkl e o HASH estão corretos no `app.py`.")