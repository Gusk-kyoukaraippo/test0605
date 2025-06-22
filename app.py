import os
import shutil
import streamlit as st
import json
import logging
import uuid

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    Settings,
)
from llama_index.core.llms import ChatMessage  # ★ チャット履歴のためにインポート
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from google.cloud import storage
from google.cloud import logging as google_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.oauth2 import service_account

st.set_page_config(page_title="フランケンAIプロンプトテスト(チャット版)", layout="wide")
logger = logging.getLogger(__name__)

LOCAL_INDEX_DIR = "downloaded_storage_openai_embed"
# 既存のプロンプトをそのまま context_prompt として利用
DEFAULT_CONTEXT_PROMPT = """
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

# --- 1. 設定と初期化処理 (チャット対応) ---
# [変更] 会話履歴(messages)と会話ID(conversation_id)をセッションステートに追加 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if 'request_id' not in st.session_state: # 各ターンのIDとして利用
    st.session_state.request_id = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
# [変更] 最新のLLM応答オブジェクトを保持するために追加
if 'last_response_obj' not in st.session_state:
    st.session_state.last_response_obj = None

# ★ ロギングフォーマッター (変更なし)
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

# [変更] LLM応答取得関数をクエリエンジンからチャットエンジンに変更 
def get_chat_response(index: VectorStoreIndex, query: str, chat_history: list, n_value: int, custom_context_prompt_str: str):
    try:
        llm = GoogleGenAI(model="gemini-1.5-flash-latest")
        context_template = PromptTemplate(custom_context_prompt_str)
        
        # LlamaIndexが要求するChatMessage形式に履歴を変換
        llama_chat_history = [ChatMessage(role=m["role"], content=m["content"]) for m in chat_history]

        # as_chat_engineに変更し、会話の文脈を考慮 
        chat_engine = index.as_chat_engine(
            llm=llm,
            similarity_top_k=n_value,
            chat_mode="context",
            context_prompt=context_template, # カスタムプロンプトを適用
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
        n_value = st.sidebar.slider("類似ドキュメント検索数 (N値)", 1, 10, 3, 1)
        custom_prompt_text = st.sidebar.text_area("カスタムコンテキストプロンプト:", DEFAULT_CONTEXT_PROMPT, height=350)

        st.header("💬 フランケンAIとチャット")

        # [変更] 会話履歴をループで表示 
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # [変更] 入力ウィジェットをst.chat_inputに変更 
        if prompt := st.chat_input("フランケンAIに聞きたいことを入力:"):
            # [追加] 会話IDがなければ新規作成 
            if not st.session_state.conversation_id:
                st.session_state.conversation_id = str(uuid.uuid4())
            
            # [追加] ユーザーのメッセージを履歴に追加して表示
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # [変更] ターンごとのIDを生成 
            turn_id = str(uuid.uuid4())
            st.session_state.request_id = turn_id
            
            # [変更] ユーザー質問ログに会話IDとターンIDを追加 
            log_extra_user = {
                'json_fields': {
                    'conversation_id': st.session_state.conversation_id,
                    'request_id': turn_id,
                    'query': prompt
                }
            }
            logger.info("新しいクエリの処理を開始します。", extra=log_extra_user)

            # [追加] AIの応答を待つ間のプレースホルダー
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # LLMから応答を取得
                response_obj = get_chat_response(
                    index=llama_index,
                    query=prompt,
                    chat_history=st.session_state.messages[:-1], # 最新の質問は除く
                    n_value=n_value,
                    custom_context_prompt_str=custom_prompt_text
                )

                if response_obj:
                    full_response = str(response_obj)
                    message_placeholder.markdown(full_response)
                    st.session_state.last_response_obj = response_obj
                    
                    # [変更] AI回答ログに会話IDとターンIDを追加 
                    log_extra_assistant = {
                        'json_fields': {
                            'conversation_id': st.session_state.conversation_id,
                            'request_id': turn_id,
                            'response': full_response
                        }
                    }
                    logger.info("LLMからの回答を記録しました。", extra=log_extra_assistant)
                    # [追加] AIの応答を履歴に追加
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    error_message = "エラーにより回答を生成できませんでした。"
                    message_placeholder.error(error_message)
                    log_extra_error = {
                        'json_fields': {
                            'conversation_id': st.session_state.conversation_id,
                            'request_id': turn_id,
                        }
                    }
                    logger.error("LLMからの応答がありませんでした。", extra=log_extra_error)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

            st.session_state.feedback_submitted = False
            st.rerun()

        # [変更] 最新のAI応答に対してのみソースとフィードバックフォームを表示
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
                    ratings = {
                        "busso_doai": st.slider("1. 物騒度合い", 1, 5, 3),
                        "datousei": st.slider("2. 妥当性", 1, 5, 3),
                        "igaisei": st.slider("3. 意外性", 1, 5, 3),
                        "humor": st.slider("4. ユーモア", 1, 5, 3),
                    }
                    feedback_comment = st.text_area("その他コメント:")
                    submit_button = st.form_submit_button(label='フィードバックを送信')

                    if submit_button:
                        # [変更] 感想ログに会話IDとターンIDを追加 
                        last_user_message = st.session_state.messages[-2] if len(st.session_state.messages) > 1 else {}
                        last_assistant_message = st.session_state.messages[-1] if st.session_state.messages else {}
                        
                        feedback_log_extra = {
                            'json_fields': {
                                'conversation_id': st.session_state.conversation_id,
                                'request_id': st.session_state.request_id,
                                'query': last_user_message.get('content', ''),
                                'response': last_assistant_message.get('content', ''),
                                'ratings': ratings,
                                'comment': feedback_comment
                            }
                        }
                        logger.info("ユーザーからの感想を記録しました。", extra=feedback_log_extra)
                        st.session_state.feedback_submitted = True
                        st.session_state.last_response_obj = None # フィードバック送信後は一旦クリア
                        st.rerun()

if __name__ == "__main__":
    main()
