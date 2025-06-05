import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from google.cloud import storage
import json
import shutil # ファイル/ディレクトリ操作用

# --- 1. Gemini APIキーの設定 ---
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Gemini APIキーが設定されていません。'.streamlit/secrets.toml' ファイルに GOOGLE_API_KEY を設定してください。")
    st.stop()

# --- 2. GCS設定と認証 ---
try:
    GCS_BUCKET_NAME = st.secrets["GCS_BUCKET_NAME"]
    GCS_INDEX_PREFIX = st.secrets["GCS_INDEX_PREFIX"] # 例: "my_rag_index/"
    
    # GCSサービスアカウントJSONをファイルに書き出し、環境変数に設定
    gcs_service_account_json_str = st.secrets["GCS_SERVICE_ACCOUNT_JSON"]
    # 改行コードのエスケープを解除
    gcs_service_account_json_str = gcs_service_account_json_str.replace('\\n', '\n')
    
    # Streamlitのテンポラリディレクトリにサービスアカウントキーを保存
    # これはデプロイ環境でのみ必要で、ローカル実行時はgcloud CLI認証や環境変数で対応可能
    temp_gcs_key_path = os.path.join("/tmp", "gcs_key.json")
    with open(temp_gcs_key_path, "w") as f:
        f.write(gcs_service_account_json_str)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_gcs_key_path

except KeyError as e:
    st.error(f"GCS設定が不足しています。'.streamlit/secrets.toml' ファイルに {e} を設定してください。")
    st.stop()


# ローカルにインデックスをダウンロードする一時ディレクトリ
LOCAL_INDEX_DIR = "downloaded_storage" # GitHubには上がらないのでこのままでOK

# ★カスタムQAプロンプトの定義（デフォルト値）
DEFAULT_QA_PROMPT = """
あなたは、提供された「参照情報」に基づいて、ユーザーの「質問」に明確かつ簡潔に回答するAIアシスタントです。
以下の指示に従って回答を生成してください:

1.  「参照情報」からユーザーの「質問」に「最終回答」を1～2文で作ってください
2.  「最終回答」と「質問」を結ぶ説明を「参照情報」を参考にしつつ、オリジナルで作成してください

参照情報:
---------------------
{context_str}
---------------------

質問:
{query_str}

回答:
"""

@st.cache_resource
def load_llama_index_from_gcs():
    """
    GCSからインデックスファイルをダウンロードし、それをロードします。
    """
    # 既存のローカルディレクトリをクリア (初回ロード時のみ必要)
    if os.path.exists(LOCAL_INDEX_DIR):
        st.write(f"既存のローカルインデックスディレクトリ '{LOCAL_INDEX_DIR}' をクリアします...")
        shutil.rmtree(LOCAL_INDEX_DIR) 
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True) # ディレクトリを作成

    st.info(f"GCSバケット '{GCS_BUCKET_NAME}' からインデックスファイルをダウンロード中... (パス: '{GCS_INDEX_PREFIX}')")
    
    try:
        # GCSクライアントの初期化 (認証は環境変数 GOOGLE_APPLICATION_CREDENTIALS 経由)
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        # 指定されたプレフィックス内のすべてのブロブをリストアップ
        blobs = bucket.list_blobs(prefix=GCS_INDEX_PREFIX)
        
        download_count = 0
        for blob in blobs:
            # GCSのパスからローカルファイルパスを構築
            # 例: my_rag_index/docstore.json -> downloaded_storage/docstore.json
            relative_path = os.path.relpath(blob.name, GCS_INDEX_PREFIX)
            local_file_path = os.path.join(LOCAL_INDEX_DIR, relative_path)
            
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # ファイルをダウンロード
            blob.download_to_filename(local_file_path)
            # st.write(f"ダウンロード済み: {blob.name} -> {local_file_path}") # デバッグ用
            download_count += 1
        
        if download_count == 0:
            st.warning(f"GCSバケット '{GCS_BUCKET_NAME}' の '{GCS_INDEX_PREFIX}' パスにファイルが見つかりませんでした。パスが正しいか確認してください。")
            return None

        st.success(f"インデックスファイル {download_count} 個がGCSから正常にダウンロードされました。")

        # ダウンロードしたファイルからインデックスをロード
        storage_context = StorageContext.from_defaults(persist_dir=LOCAL_INDEX_DIR)
        index = load_index_from_storage(storage_context)
        st.success("LlamaIndexがインデックスを正常にロードしました。")
        return index
    except Exception as e:
        st.error(f"GCSからのインデックスロード中にエラーが発生しました: {e}")
        st.info("GCSバケット名、パス、または認証情報（Streamlit Secretsの GCS_SERVICE_ACCOUNT_JSON）を確認してください。")
        return None

def get_response_from_llm(index, query: str, n_value: int, custom_qa_template_str: str):
    """
    指定されたインデックス、クエリ、N値、カスタムプロンプトを使用してLLMから応答を取得します。
    """
    llm = GoogleGenAI(model="gemini-2.5-flash-preview-05-20") 
    qa_template = PromptTemplate(custom_qa_template_str)

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=n_value,
        text_qa_template=qa_template
    ) 

    st.info(f"クエリを実行中: '{query}'")
    with st.spinner('AIが回答を生成中です...'):
        response = query_engine.query(query)
    return response

# --- Streamlit UIの構築 ---
st.set_page_config(page_title="RAGベースQAウェブアプリ (GCS対応)", layout="wide")
st.title("📚 ドキュメントQAボット (GCS連携)")

st.markdown("""
このアプリは、Google Cloud Storage (GCS) に保存されたドキュメントインデックスに基づいて質問に回答します。
左側のサイドバーでプロンプトや関連度 (`N` 値) を調整して、AIの応答を試すことができます。
---
""")

# インデックスのロード
# GCSからロードする関数を呼び出す
llama_index = load_llama_index_from_gcs()

if llama_index:
    # サイドバーでの設定
    st.sidebar.header("⚙️ 設定")

    # N値の調整スライダー
    n_value = st.sidebar.slider(
        "類似度トップK (N値)", 
        min_value=1, 
        max_value=10, 
        value=5, 
        step=1,
        help="回答生成のために取得する最も関連性の高いドキュメントチャンクの数を設定します。"
    )
    st.sidebar.info(f"上位 **{n_value}** 個の関連性の高いドキュメントチャンクを使用して回答を生成します。")

    # プロンプトの調整テキストエリア
    st.sidebar.subheader("📝 カスタムQAプロンプト")
    custom_prompt_text = st.sidebar.text_area(
        "プロンプトを編集してください (context_str と query_str は必須のプレースホルダーです)",
        DEFAULT_QA_PROMPT,
        height=400,
        help="AIに指示を与えるプロンプトをカスタマイズできます。`{context_str}` と `{query_str}` は必ず含めてください。"
    )

    st.sidebar.markdown("---")
    st.sidebar.write("© 2024 LlamaIndex Streamlit Demo")

    # メインコンテンツエリア
    st.header("質問をしてください")
    user_query = st.text_input("ここに質問を入力してください", placeholder="例: このドキュメントで説明されている主要なコンセプトは何ですか？")

    if user_query:
        if "context_str" not in custom_prompt_text or "query_str" not in custom_prompt_text:
            st.warning("カスタムプロンプトには `{context_str}` と `{query_str}` が含まれている必要があります。サイドバーを確認してください。")
        else:
            st.info(f"質問: **{user_query}**")
            response = get_response_from_llm(llama_index, user_query, n_value, custom_prompt_text)
            
            st.subheader("🤖 回答")
            st.write(str(response))

            # 詳細情報（オプション）
            with st.expander("詳細情報を見る"):
                st.write("**使用されたプロンプト:**")
                st.code(custom_prompt_text, language='text')
                st.write(f"**使用されたN値:** {n_value}")
                if hasattr(response, 'source_nodes'):
                    st.write("**参照されたドキュメントのチャンク:**")
                    for i, node in enumerate(response.source_nodes):
                        st.write(f"--- チャンク {i+1} ---")
                        st.text(node.text)
                        if node.metadata:
                            st.json(node.metadata)
                else:
                    st.info("ソースドキュメント情報は利用できません。")

else:
    st.warning("インデックスがロードされていないため、質問に回答できません。上記のエラーメッセージを確認してください。")