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

# [変更] ペルソナ設定とJSON出力形式を指示するシステムプロンプト
SYSTEM_PROMPT = """
LLMへの指示文（フランケン氏の意見活用・修正案）
あなたは、ユーザーからの質問と、それに関連する【参照情報】を元に対話するAIアシスタントです。以下のキャラクター設定とルールを厳格に守って、ユーザーと対話してください。

# キャラクター設定
名前: フランケン
役割: 夜の街の片隅でカウンターに立つ、人生経験豊富なオネエのママ。
性格: 気さくで面倒見が良く、どんな話にも優しく耳を傾ける。人生の酸いも甘いも噛み分けた鋭い洞察力を持ち、時に相談者の甘えや見栄を、愛のある厳しさでピシャリと指摘する。
口調:
一人称は「アタシ」、二人称は「アンタ」。
タメ口を基本とした、親しみやすいオネエ口調（「〜よ」「〜だわ」「〜なのよ」「〜かしら」など）で話す。
敬体（です・ます調）は絶対に使わない。
# 思考プロセス（回答を考えるときの手順）
共感と受け止め:まず、ユーザーの言葉を包み込むように優しく受け止め、労いや共感の気持ちを表現する。

本質の喝破:次に、あなたに与えられた**【参照情報】を深く読み解く。** これには客観的な事実に加え、「フランケン」という人物の意見が多く含まれていることを理解する。
あなたはその意見を吟味し、あたかもそれが元々自分自身の持論であったかのように、完全に自分の言葉・自分の視点として再構築する。 その鋭い洞察を根拠に、相談の核心を突く指摘を行う。

愛のあるお説教と応援:その事実や持論を踏まえた上で、ユーザーが次に取るべき行動を、具体的かつ少し強引なくらいハッキリと示す。最後は応援の言葉で締めくくる。

# 厳守すべきルール
回答は、いかなる時も必ずペルソナ（オネエのママ・フランケン）として記述してください。
【最重要】 回答の中に「参照情報によると」「フランケン氏の意見では」といった情報源を示す言葉や、「AI」「アシスタント」など舞台裏を匂わせる言葉は絶対に使わないでください。あなたはあくまで、参照情報の内容を完全に自分のものとして、自身の人生経験から語っているという設定を貫いてください。
最終的な回答の文字数は、全体で200字から300字程度にまとめてください。
# 出力形式
あなたの応答は、必ず以下のJSON形式に従ってください。他の形式は一切認められません。

```json
{
  "response": "ここに、上記のキャラクター設定に従って生成したユーザー向けの回答を記述します。",
  "reasoning": "ここには、なぜその回答を生成したのか、あなたの思考プロセスを開発者向けに簡潔に記述します。例えば、どの参照情報を重視したか、どのような論理で結論に至ったかなどを含めてください。参照情報の引用は一部のみとしてください。"
}
```
"""

# 検索した情報をLLMに渡す際のプロンプト
CONTEXT_PROMPT_TEMPLATE = """
参照情報:
---------------------
{context_str}
---------------------
上記の参照情報を踏まえて、ユーザーの質問に答えなさい。
"""

# 会話履歴を元に検索クエリを再生成するためのプロンプト
CONDENSE_QUESTION_PROMPT_TEMPLATE = """
チャット履歴と最後の質問が与えられます。チャット履歴の文脈を使って、関連する会話を盛り込んだスタンドアロンの質問に変換してください。

チャット履歴:
---------------------
{chat_history}
---------------------
最後の質問: {question}

スタンドアロンの質問:
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

def get_chat_response(index: VectorStoreIndex, query: str, chat_history: list, n_value: int, system_prompt: str, context_prompt_template: str, condense_prompt_template: str):
    try:
        llm = GoogleGenAI(model="gemini-2.5-flash-lite-preview-06-17")
        context_template = PromptTemplate(context_prompt_template)
        condense_template = PromptTemplate(condense_prompt_template)
        
        llama_chat_history = [ChatMessage(role=m["role"], content=m["content"]) for m in chat_history]

        chat_engine = index.as_chat_engine(
            llm=llm,
            similarity_top_k=n_value,
            chat_mode="condense_plus_context",
            system_prompt=system_prompt,
            context_prompt=context_template,
            condense_prompt=condense_template
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
        custom_system_prompt = st.sidebar.text_area("システムプロンプト (ペルソナ設定):", SYSTEM_PROMPT, height=400)
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
                    
                    response_obj = get_chat_response(
                        index=llama_index,
                        query=prompt,
                        chat_history=st.session_state.messages[:-1],
                        n_value=n_value,
                        system_prompt=custom_system_prompt,
                        context_prompt_template=CONTEXT_PROMPT_TEMPLATE,
                        condense_prompt_template=CONDENSE_QUESTION_PROMPT_TEMPLATE
                    )

                    if response_obj:
                        full_response_text = ""
                        reasoning_text = ""
                        
                        # [変更] LLMからの応答をJSONとしてパース
                        try:
                            # ```json ... ``` のようなマークダウンブロックを削除
                            raw_json = response_obj.response.strip().replace("```json", "").replace("```", "")
                            parsed_response = json.loads(raw_json)
                            full_response_text = parsed_response.get("response", "回答の取得に失敗しました。")
                            reasoning_text = parsed_response.get("reasoning", "理由の取得に失敗しました。")
                        except (json.JSONDecodeError, AttributeError) as e:
                            logger.warning(f"LLMの応答のJSONパースに失敗しました: {e}", extra={'json_fields': {'response_text': response_obj.response}})
                            full_response_text = str(response_obj.response) # パース失敗時はそのまま表示
                            reasoning_text = "JSONパース失敗"

                        message_placeholder.markdown(full_response_text)
                        st.session_state.last_response_obj = response_obj
                        
                        source_nodes_for_log = [
                            {"text": node.text, "score": node.score} for node in response_obj.source_nodes
                        ]
                        chat_history_for_log = st.session_state.messages[:-1]

                        # [変更] ログにreasoningを追加
                        log_extra_assistant = {
                            'json_fields': {
                                'conversation_id': st.session_state.conversation_id,
                                'request_id': turn_id,
                                'query': prompt,
                                'response': full_response_text,
                                'reasoning': reasoning_text, # 生成理由を記録
                                'source_nodes': source_nodes_for_log,
                                'chat_history': chat_history_for_log,
                            }
                        }
                        logger.info("LLMからの回答と関連情報を記録しました。", extra=log_extra_assistant)
                        
                        st.session_state.messages.append({"role": "assistant", "content": full_response_text})
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
