import os
import shutil
import streamlit as st
import json
import logging
import uuid  # ★ ユニークIDを生成するためにインポート

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    Settings,
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from google.cloud import storage
from google.cloud import logging as google_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.oauth2 import service_account

st.set_page_config(page_title="フランケンAIプロンプトテスト", layout="wide")
logger = logging.getLogger(__name__)

LOCAL_INDEX_DIR = "downloaded_storage_openai_embed"
DEFAULT_QA_PROMPT = """
あなたは、提供された「参照情報」に基づいて、ユーザーの「質問」に回答するAIアシスタントです。
以下の指示に従って回答を生成してください:
1.  参照情報の話者であるフランケンとして答えてください
2.  率直な意見をください。回答者を刺すくらいの勢いで。
3.  結論を先に決めて、その結論を見た目上エキセントリックな表現にして、最終的には平易な解釈で質問者に飲み込ませるような表現で。
参照情報:
---------------------
{context_str}
---------------------
質問:
{query_str}
回答:
"""

# --- 1. 設定と初期化処理 ---
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'last_response' not in st.session_state:
    st.session_state.last_response = ""
if 'source_nodes' not in st.session_state:
    st.session_state.source_nodes = []
# ★ リクエストIDをセッションステートで管理
if 'request_id' not in st.session_state:
    st.session_state.request_id = None

# ★ コンソール出力でもリクエストIDを見やすくするためのカスタムフォーマッター
class RequestIdFormatter(logging.Formatter):
    def format(self, record):
        # extraで渡された辞書をログメッセージに含める
        log_message = super().format(record)
        if hasattr(record, 'json_fields'):
            # json_fieldsの内容を追記
            log_message += f" {record.json_fields}"
        return log_message

@st.cache_resource
def setup_gcp_services():
    st.info("GCPサービスとの接続を初期化中...")
    try:
        gcs_service_account_json_str = st.secrets["GCS_SERVICE_ACCOUNT_JSON"]
        parsed_json = json.loads(gcs_service_account_json_str)
        project_id = parsed_json.get("project_id")
        if not project_id:
            raise ValueError("サービスアカウントJSONに 'project_id' が見つかりません。")
        credentials = service_account.Credentials.from_service_account_info(parsed_json)
        st.success(f"GCPプロジェクト '{project_id}' の認証情報を準備しました。")
    except Exception as e:
        st.error(f"GCS_SERVICE_ACCOUNT_JSON の設定読み込み中にエラーが発生しました: {e}")
        st.stop()

    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    
    # ストリームハンドラー (コンソール出力用)
    sh = logging.StreamHandler()
    # ★ カスタムフォーマッターを設定
    sh.setFormatter(RequestIdFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

    # Cloud Logging ハンドラー
    try:
        client = google_logging.Client(credentials=credentials, project=project_id)
        handler = CloudLoggingHandler(client, name="franken-ai-prompt-test")
        logger.addHandler(handler)
        logger.info("Google Cloud Loggingに接続しました。")
    except Exception as e:
        logger.warning(f"Google Cloud Loggingとの連携に失敗しました: {e}", exc_info=True)

    gcs_client = storage.Client(credentials=credentials, project=project_id)
    return gcs_client

# (load_llama_index_from_gcs, get_response_from_llm は変更なし)
@st.cache_resource
def load_llama_index_from_gcs(_gcs_client: storage.Client, bucket_name: str, index_prefix: str):
    if os.path.exists(LOCAL_INDEX_DIR):
        shutil.rmtree(LOCAL_INDEX_DIR)
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)
    with st.spinner("初回インデックス読み込み中... (約1分かかります)"):
        try:
            bucket = _gcs_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=index_prefix))
            if not blobs: return None
            for blob in blobs:
                if blob.name == index_prefix or blob.name.endswith('/'): continue
                relative_path = os.path.relpath(blob.name, index_prefix)
                local_file_path = os.path.join(LOCAL_INDEX_DIR, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                blob.download_to_filename(local_file_path)
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", dimensions=3072)
            storage_context = StorageContext.from_defaults(persist_dir=LOCAL_INDEX_DIR)
            index = load_index_from_storage(storage_context)
            st.success("インデックスのロードが完了しました。")
            return index
        except Exception as e:
            st.error(f"GCSからのインデックスロード中にエラーが発生しました: {e}")
            return None

def get_response_from_llm(index: VectorStoreIndex, query: str, n_value: int, custom_qa_template_str: str):
    try:
        llm = GoogleGenAI(model="gemini-1.5-flash-latest")
        qa_template = PromptTemplate(custom_qa_template_str)
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=n_value, text_qa_template=qa_template)
        with st.spinner('AIが回答を生成中です...'):
            response = query_engine.query(query)
        return response
    except Exception as e:
        st.error(f"LLMからの応答取得中にエラーが発生しました: {e}")
        return None

def main():
    st.title("🤖 プロンプトテスト")
    st.markdown("このアプリは、フランケンラジオAIのプロンプトテストが出来ます。")
    st.markdown("---")
    
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        GCS_BUCKET_NAME = st.secrets["GCS_BUCKET_NAME"]
        GCS_INDEX_PREFIX = st.secrets["GCS_INDEX_PREFIX"]
    except KeyError as e:
        st.error(f"必要な設定が secrets.toml に見つかりません: {e}。")
        st.stop()

    gcs_client = setup_gcp_services()
    if not gcs_client:
        st.stop()
        
    llama_index = load_llama_index_from_gcs(gcs_client, GCS_BUCKET_NAME, GCS_INDEX_PREFIX)

    if llama_index:
        st.sidebar.header("⚙️ 高度な設定")
        n_value = st.sidebar.slider("類似ドキュメント検索数 (N値)", 1, 10, 3, 1)
        custom_prompt_text = st.sidebar.text_area("カスタムQAプロンプト:", DEFAULT_QA_PROMPT, height=350)

        st.header("💬 質問を入力してください")
        user_query = st.text_input("フランケンAIに聞きたいことを入力:", key="user_query_input")

        if user_query and user_query != st.session_state.last_query:
            st.session_state.last_query = user_query
            st.session_state.feedback_submitted = False
            
            # ★ 新しい質問が来たので、新しいリクエストIDを生成
            st.session_state.request_id = str(uuid.uuid4())
            
            # ★ extra に辞書を渡すことで、構造化ログとしてIDを記録
            log_extra = {'json_fields': {'request_id': st.session_state.request_id, 'query': user_query}}
            logger.info("新しいクエリの処理を開始します。", extra=log_extra)

            response = get_response_from_llm(llama_index, user_query, n_value, custom_prompt_text)

            if response:
                st.session_state.last_response = str(response)
                st.session_state.source_nodes = response.source_nodes
                # ★ ログにリクエストIDを付与
                logger.info(f"LLMからの回答を記録しました。", extra={'json_fields': {'request_id': st.session_state.request_id, 'response': str(response)}})
            else:
                st.session_state.last_response = ""
                st.session_state.source_nodes = []
                logger.error("LLMからの応答がありませんでした。", extra={'json_fields': {'request_id': st.session_state.request_id}})

        if st.session_state.last_response:
            st.subheader("🤖 AIからの回答")
            st.write(st.session_state.last_response)

            with st.expander("参照されたソースを確認"):
                for i, node in enumerate(st.session_state.source_nodes):
                    st.markdown(f"--- **ソース {i+1} (関連度: {node.score:.4f})** ---")
                    st.text_area("", value=node.text, height=150, disabled=True, key=f"chunk_{i}")

            st.markdown("---")
            st.subheader("📝 この回答についての感想")

            if st.session_state.feedback_submitted:
                st.success("フィードバックを記録しました。ありがとうございます！")
            else:
                with st.form(key='feedback_form'):
                    ratings = {
                        "busso_doai": st.slider("1. 物騒度合い", 1, 5, 3),
                        "datousei": st.slider("2. 妥当性", 1, 5, 3),
                        "igaisei": st.slider("3. 意外性", 1, 5, 3),
                        "humor": st.slider("4. ユーモア", 1, 5, 3),
                    }
                    feedback_comment = st.text_area("その他コメント:")
                    submit_button = st.form_submit_button(label='フィードバックを送信')

                    if submit_button:
                        # ★ 感想ログにもリクエストIDを付与
                        feedback_log_extra = {
                            'json_fields': {
                                'request_id': st.session_state.request_id,
                                'query': st.session_state.last_query,
                                'response': st.session_state.last_response,
                                'ratings': ratings,
                                'comment': feedback_comment
                            }
                        }
                        logger.info("ユーザーからの感想を記録しました。", extra=feedback_log_extra)
                        st.session_state.feedback_submitted = True
                        st.rerun()

if __name__ == "__main__":
    main()
