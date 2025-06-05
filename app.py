import os
import shutil
import streamlit as st
import json
import tempfile
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    Settings,
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.cloud import storage

# --- ページ設定 (最初に一度だけ呼び出す) ---
st.set_page_config(page_title="RAGベースQAウェブアプリ (GCS対応)", layout="wide")

# --- 定数定義 ---
LOCAL_INDEX_DIR = "downloaded_storage"
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

# --- 1. APIキーとGCS認証情報の設定 ---
temp_gcs_key_path = None
try:
    # Streamlit secretsから設定を読み込み
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    GCS_BUCKET_NAME = st.secrets["GCS_BUCKET_NAME"]
    GCS_INDEX_PREFIX = st.secrets["GCS_INDEX_PREFIX"]
    gcs_service_account_json_str = st.secrets["GCS_SERVICE_ACCOUNT_JSON"]

    # GCSサービスアカウントJSONをパースして一時ファイルに保存
    try:
        parsed_json = json.loads(gcs_service_account_json_str)
        clean_json_str = json.dumps(parsed_json)
    except json.JSONDecodeError as e:
        st.error(f"Streamlit secretsの'GCS_SERVICE_ACCOUNT_JSON'が不正なJSON形式です: {e}")
        st.info(
            "secrets.tomlのGCPサービスアカウントキーが正しいJSON形式か確認してください。"
            "特に、三重引用符(`\"\"\"`)で囲むと改行やエスケープの問題が起きにくくなります。"
        )
        st.stop()

    # 一時ファイルに認証情報を書き出し、環境変数にパスを設定
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8") as tmp_file:
        tmp_file.write(clean_json_str)
        temp_gcs_key_path = tmp_file.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_gcs_key_path

except KeyError as e:
    st.error(
        f"必要な設定が secrets.toml に見つかりません: {e}。"
        "GOOGLE_API_KEY, GCS_BUCKET_NAME, GCS_INDEX_PREFIX, GCS_SERVICE_ACCOUNT_JSON を設定してください。"
    )
    st.stop()
except Exception as e:
    st.error(f"設定の読み込み中に予期せぬエラーが発生しました: {e}")
    st.exception(e)
    st.stop()


# --- 2. LlamaIndex関連の関数 ---
@st.cache_resource
def load_llama_index_from_gcs():
    """
    Google Cloud Storage (GCS) からLlamaIndexのインデックスをダウンロードし、ロードします。
    インデックスはローカルディレクトリにキャッシュされます。
    """
    if os.path.exists(LOCAL_INDEX_DIR):
        shutil.rmtree(LOCAL_INDEX_DIR)
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

    st.info(f"GCSバケット '{GCS_BUCKET_NAME}' からインデックスをダウンロード中...")

    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=GCS_INDEX_PREFIX))

        if not blobs or all(blob.name == GCS_INDEX_PREFIX and blob.size == 0 for blob in blobs):
            st.warning(f"GCSバケット '{GCS_BUCKET_NAME}' の '{GCS_INDEX_PREFIX}' にインデックスファイルが見つかりません。")
            return None

        download_count = 0
        for blob in blobs:
            if blob.name == GCS_INDEX_PREFIX or blob.name.endswith('/'):
                continue
            relative_path = os.path.relpath(blob.name, GCS_INDEX_PREFIX)
            local_file_path = os.path.join(LOCAL_INDEX_DIR, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
            download_count += 1

        if download_count == 0:
            st.warning(f"GCSの '{GCS_INDEX_PREFIX}' パスにダウンロード可能なファイルが見つかりませんでした。")
            return None

        st.success(f"{download_count} 個のインデックスファイルをGCSからダウンロードしました。")

        # 埋め込みモデルを設定
        embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")

        # ローカルのインデックスをロード
        storage_context = StorageContext.from_defaults(persist_dir=LOCAL_INDEX_DIR)
        index = load_index_from_storage(storage_context)
        st.success("インデックスのロードが完了しました。質問を入力できます。")
        return index

    except Exception as e:
        st.error(f"GCSからのインデックスロード中にエラーが発生しました: {e}")
        st.exception(e)
        st.info(
            "以下の点を確認してください:\n"
            f"- GCSバケット名 ('{GCS_BUCKET_NAME}') とプレフィックス ('{GCS_INDEX_PREFIX}') が正しいか。\n"
            "- 'GCS_SERVICE_ACCOUNT_JSON' が正しく、適切な権限を持っているか。\n"
            "- インターネット接続が安定しているか。"
        )
        return None

def get_response_from_llm(index: VectorStoreIndex, query: str, n_value: int, custom_qa_template_str: str):
    """
    LLMを使用して、LlamaIndexのインデックスから回答を生成します。
    """
    try:
        llm = GoogleGenAI(model="gemini-2.5-flash-preview-05-20")
        qa_template = PromptTemplate(custom_qa_template_str)
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=n_value,
            text_qa_template=qa_template
        )
        with st.spinner('AIが回答を生成中です...'):
            response = query_engine.query(query)
        return response
    except Exception as e:
        st.error(f"LLMからの応答取得中にエラーが発生しました: {e}")
        st.exception(e)
        st.info("Gemini APIキーが有効か、選択したモデルが利用可能か確認してください。")
        return None

# --- 3. Streamlit UIの構築 ---
def main():
    """
    StreamlitアプリケーションのメインUIを構築します。
    """
    st.title("📚 ドキュメントQAボット (GCS連携)")
    st.markdown("""
    このアプリは、GCSに保存されたドキュメントのインデックスを使い、内容に関する質問に回答します。
    左のサイドバーで、検索設定やAIへの指示（プロンプト）を調整できます。
    """)
    st.markdown("---")

    llama_index = load_llama_index_from_gcs()

    if llama_index:
        st.sidebar.header("⚙️ 高度な設定")
        n_value = st.sidebar.slider(
            "類似ドキュメント検索数 (N値)", 1, 10, 3, 1,
            help="回答生成の際に参照する、関連性の高いドキュメントの数を指定します。"
        )
        st.sidebar.info(f"現在、上位 **{n_value}** 個の関連チャンクを使用して回答を生成します。")

        st.sidebar.subheader("📝 カスタムQAプロンプト")
        custom_prompt_text = st.sidebar.text_area(
            "プロンプトを編集 ({context_str}と{query_str}は必須):",
            DEFAULT_QA_PROMPT, height=350,
            help="AIへの指示です。`{context_str}`(参照情報)と`{query_str}`(質問)を含めてください。"
        )
        st.sidebar.markdown("---")
        st.sidebar.caption("© 2024 RAG Demo")

        st.header("💬 質問を入力してください")
        user_query = st.text_input(
            "ドキュメントに関する質問をここに入力してください:",
            placeholder="例: このドキュメントの主要なテーマは何ですか？"
        )

        if user_query:
            if "{context_str}" not in custom_prompt_text or "{query_str}" not in custom_prompt_text:
                st.warning("プロンプトには`{context_str}`と`{query_str}`の両方を含めてください。")
            else:
                response = get_response_from_llm(llama_index, user_query, n_value, custom_prompt_text)
                if response:
                    st.subheader("🤖 AIからの回答")
                    st.write(str(response))
                    
                    # 参照ソースの表示（デバッグ用ではなく、ユーザーへの情報提供として）
                    if response.source_nodes:
                        with st.expander("参照されたソースを確認"):
                            for i, node in enumerate(response.source_nodes):
                                st.markdown(f"--- **ソース {i+1} (関連度: {node.score:.2f})** ---")
                                st.text_area(
                                    label=f"ソース {i+1} の内容",
                                    value=node.text, 
                                    height=150, 
                                    disabled=True,
                                    key=f"chunk_{i}"
                                )


    else:
        st.error(
            "インデックスのロードに失敗したため、QAシステムを起動できませんでした。"
            "ページ上部のエラーメッセージを確認し、設定やGCSの状態を見直してください。"
        )

if __name__ == "__main__":
    main()
