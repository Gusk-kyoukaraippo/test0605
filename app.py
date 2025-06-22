import os
import shutil
import streamlit as st
import json
import logging
import uuid
import streamlit.components.v1 as components

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    Settings,
)
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from google.cloud import storage
from google.cloud import logging as google_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.oauth2 import service_account

st.set_page_config(page_title="フランケンAIプロンプトテスト(チャット版)", layout="wide")
logger = logging.getLogger(__name__)

LOCAL_INDEX_DIR = "downloaded_storage_openai_embed"

# [変更] ペルソナ設定を「システムプロンプト」として分離
SYSTEM_PROMPT = """
あなたは、参照情報の話者である「フランケン」というキャラクターです。常にフランケンとして、以下の厳格なルールに従って応答してください。
1.  率直かつ物議を醸すような、刺々しい意見を述べなさい。
2.  まず最初に、常識から外れたエキセントリックな結論を断定的に提示しなさい。
3.  その後、その結論が実は的を射たものであることを、平易な言葉で論理的に説明し、質問者を納得させなさい。
4.  参照情報がある場合は、その内容を自身の知識であるかのように自然に会話に織り交ぜなさい。
"""

# [変更] コンテキストプロンプトは参照情報のフォーマットに特化させる
CONTEXT_PROMPT_TEMPLATE = """
参照情報:
---------------------
{context_str}
---------------------
上記の参照情報を踏まえて、ユーザーの質問に答えなさい。
"""

# --- 1. 設定と初期化処理 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if 'request_id' not in st.session_state:
    st.session_state.request_id = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'last_response_obj' not in st.session_state:
    st.session_state.last_response_obj = None
if 'scroll_to_bottom' not in st.session_state:
    st.session_state.scroll_to_bottom = False


class RequestIdFormatter(logging.Formatter):
    def format(self, record):
        log_message = super().format(record)
        if hasattr(record, 'json_fields'):
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
    
    sh = logging.StreamHandler()
    sh.setFormatter(RequestIdFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

    try:
        client = google_logging.Client(credentials=credentials, project=project_id)
        handler = CloudLoggingHandler(client, name="franken-ai-prompt-test-chat")
        logger.addHandler(handler)
        logger.info("Google Cloud Loggingに接続しました。")
    except Exception as e:
        logger.warning(f"Google Cloud Loggingとの連携に失敗しました: {e}", exc_info=True)

    gcs_client = storage.Client(credentials=credentials, project=project_id)
    return gcs_client

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

# [変更] get_chat_responseに関数を渡すように変更
def get_chat_response(index: VectorStoreIndex, query: str, chat_history: list, n_value: int, system_prompt: str, context_prompt_template: str):
    try:
        llm = GoogleGenAI(model="gemini-1.5-flash-latest")
        context_template = PromptTemplate(context_prompt_template)
        
        llama_chat_history = [ChatMessage(role=m["role"], content=m["content"]) for m in chat_history]

        chat_engine = index.as_chat_engine(
            llm=llm,
            similarity_top_k=n_value,
            chat_mode="context",
            system_prompt=system_prompt,  # [変更] system_prompt を指定
            context_prompt=context_template, # [変更] context_prompt を指定
        )
        with st.spinner('AIが回答を生成中です...'):
            response = chat_engine.chat(query, chat_history=llama_chat_history)
        return response
    except Exception as e:
        st.error(f"LLMからの応答取得中にエラーが発生しました: {e}")
        return None

def main():
    st.title("🤖 プロンプトテスト (チャット版)")
    st.markdown("このアプリは、フランケンラジオAIとチャット形式で会話ができます。")
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
        # [変更] サイドバーでシステムプロンプトを編集可能にする
        custom_system_prompt = st.sidebar.text_area("システムプロンプト (ペルソナ設定):", SYSTEM_PROMPT, height=250)
        n_value = st.sidebar.slider("類似ドキュメント検索数 (N値)", 1, 10, 3, 1)

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("フランケンAIに聞きたいことを入力:"):
            if not st.session_state.conversation_id:
                st.session_state.conversation_id = str(uuid.uuid4())
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            turn_id = str(uuid.uuid4())
            st.session_state.request_id = turn_id
            
            log_extra_user = { 'json_fields': { 'conversation_id': st.session_state.conversation_id, 'request_id': turn_id, 'query': prompt } }
            logger.info("新しいクエリの処理を開始します。", extra=log_extra_user)

            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    # [変更] システムプロンプトを渡すように変更
                    response_obj = get_chat_response(
                        index=llama_index,
                        query=prompt,
                        chat_history=st.session_state.messages[:-1],
                        n_value=n_value,
                        system_prompt=custom_system_prompt,
                        context_prompt_template=CONTEXT_PROMPT_TEMPLATE
                    )

                    if response_obj:
                        full_response = str(response_obj)
                        message_placeholder.markdown(full_response)
                        st.session_state.last_response_obj = response_obj
                        log_extra_assistant = { 'json_fields': { 'conversation_id': st.session_state.conversation_id, 'request_id': turn_id, 'response': full_response } }
                        logger.info("LLMからの回答を記録しました。", extra=log_extra_assistant)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        error_message = "エラーにより回答を生成できませんでした。"
                        message_placeholder.error(error_message)
                        log_extra_error = { 'json_fields': { 'conversation_id': st.session_state.conversation_id, 'request_id': turn_id } }
                        logger.error("LLMからの応答がありませんでした。", extra=log_extra_error)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

            st.session_state.feedback_submitted = False
            st.session_state.scroll_to_bottom = True
            st.rerun()

        if st.session_state.last_response_obj:
            with st.expander("参照されたソースを確認"):
                for i, node in enumerate(st.session_state.last_response_obj.source_nodes):
                    st.markdown(f"--- **ソース {i+1} (関連度: {node.score:.4f})** ---")
                    st.text_area("", value=node.text, height=150, disabled=True, key=f"chunk_{i}")

            st.markdown("---")
            st.subheader("📝 この回答についての感想")

            if st.session_state.feedback_submitted:
                st.success("フィードバックを記録しました。ありがとうございます！")
            else:
                with st.form(key='feedback_form'):
                    ratings = { "busso_doai": st.slider("1. 物騒度合い", 1, 5, 3), "datousei": st.slider("2. 妥当性", 1, 5, 3), "igaisei": st.slider("3. 意外性", 1, 5, 3), "humor": st.slider("4. ユーモア", 1, 5, 3) }
                    feedback_comment = st.text_area("その他コメント:")
                    submit_button = st.form_submit_button(label='フィードバックを送信')

                    if submit_button:
                        last_user_message = st.session_state.messages[-2] if len(st.session_state.messages) > 1 else {}
                        last_assistant_message = st.session_state.messages[-1] if st.session_state.messages else {}
                        feedback_log_extra = { 'json_fields': { 'conversation_id': st.session_state.conversation_id, 'request_id': st.session_state.request_id, 'query': last_user_message.get('content', ''), 'response': last_assistant_message.get('content', ''), 'ratings': ratings, 'comment': feedback_comment } }
                        logger.info("ユーザーからの感想を記録しました。", extra=feedback_log_extra)
                        st.session_state.feedback_submitted = True
                        st.session_state.last_response_obj = None
                        st.rerun()

    if st.session_state.get('scroll_to_bottom', False):
        components.html(
            """
            <script>
                window.scrollTo(0, document.body.scrollHeight);
            </script>
            """,
            height=0,
            width=0,
        )
        st.session_state.scroll_to_bottom = False

if __name__ == "__main__":
    main()
