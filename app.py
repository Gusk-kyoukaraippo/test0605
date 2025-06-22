import os
import shutil
import streamlit as st
import json
import tempfile
import logging
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
# --- Google Cloud Logging連携のためにライブラリをインポート ---
from google.cloud import logging as google_logging
from google.cloud.logging.handlers import CloudLoggingHandler

# --- ページ設定 (最初に一度だけ呼び出す) ---
st.set_page_config(page_title="フランケンAIプロンプトテスト", layout="wide")

# --- ロガーの準備 ---
# ハンドラーは後ほど設定を追加していく
logger = logging.getLogger(__name__)


# --- 定数定義 ---
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

# --- 1. APIキーとGCS認証情報の設定 & ロギング設定 ---
# セッションステートの初期化
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'last_response' not in st.session_state:
    st.session_state.last_response = ""


temp_gcs_key_path = None
try:
    # APIキーの設定
    if "GOOGLE_API_KEY" not in st.secrets:
        raise KeyError("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    if "OPENAI_API_KEY" not in st.secrets:
        raise KeyError("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    # GCS設定
    if "GCS_BUCKET_NAME" not in st.secrets:
        raise KeyError("GCS_BUCKET_NAME")
    GCS_BUCKET_NAME = st.secrets["GCS_BUCKET_NAME"]

    if "GCS_INDEX_PREFIX" not in st.secrets:
        raise KeyError("GCS_INDEX_PREFIX")
    GCS_INDEX_PREFIX = st.secrets["GCS_INDEX_PREFIX"]

    # サービスアカウントJSONのパースとプロジェクトIDの取得
    if "GCS_SERVICE_ACCOUNT_JSON" not in st.secrets:
        raise KeyError("GCS_SERVICE_ACCOUNT_JSON")
    gcs_service_account_json_str = st.secrets["GCS_SERVICE_ACCOUNT_JSON"]

    try:
        parsed_json = json.loads(gcs_service_account_json_str)
        project_id = parsed_json.get("project_id")
        if not project_id:
            raise ValueError("サービスアカウントJSONに 'project_id' が見つかりません。")
        clean_json_str = json.dumps(parsed_json)
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Streamlit secretsの'GCS_SERVICE_ACCOUNT_JSON'の形式が不正です: {e}")
        st.stop()

    # 一時ファイルに認証情報を書き出し
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8") as tmp_file:
        tmp_file.write(clean_json_str)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name
    temp_gcs_key_path = tmp_file.name

    # --- ロギング設定 ---
    # Streamlitの再実行時にハンドラーが重複して追加されるのを防ぐ
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        
        # ファイルハンドラー (ローカル用)
        fh = logging.FileHandler("app.log", encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # ストリームハンドラー (Streamlitコンソール用)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # --- ★ デバッグメッセージ追加 ---
        # Google Cloud Loggingハンドラーの追加
        st.info("Google Cloud Loggingに接続試行中...")
        try:
            client = google_logging.Client(project=project_id)
            handler = CloudLoggingHandler(client, name="franken-ai-prompt-test")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # ★ UIとログの両方に成功メッセージを表示
            success_message = "Google Cloud Loggingに接続しました。"
            st.success(success_message)
            logger.info(success_message)
            
        except Exception as e:
            # ★ UIとログの両方に失敗メッセージを表示
            error_message = f"Google Cloud Loggingとの連携に失敗しました: {e}"
            st.warning(error_message)
            logger.warning(error_message)

except KeyError as e:
    st.error(
        f"必要な設定が secrets.toml に見つかりません: {e}。"
        "GOOGLE_API_KEY, OPENAI_API_KEY, GCS_BUCKET_NAME, GCS_INDEX_PREFIX, GCS_SERVICE_ACCOUNT_JSON を設定してください。"
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
    """
    if os.path.exists(LOCAL_INDEX_DIR):
        shutil.rmtree(LOCAL_INDEX_DIR)
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

    with st.spinner("初回読み込み中... (約1分かかります)"):
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

            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", dimensions=3072)
            st.info(f"埋め込みモデル: **{Settings.embed_model.model_name}** (次元: **{Settings.embed_model.dimensions}**) を使用します。")
            
            # 埋め込みモデルのテスト
            try:
                test_embedding = Settings.embed_model.get_text_embedding("これはテスト文字列です。")
                expected_dimension = 3072
                if len(test_embedding) != expected_dimension:
                    st.error(f"埋め込みモデルが期待される {expected_dimension} 次元ではなく、{len(test_embedding)} 次元を返しました。")
                    return None
                st.success("埋め込みモデルが正常に動作することを確認しました。")
            except Exception as e:
                st.error(f"埋め込みモデルの初期テスト中にエラーが発生しました: {e}")
                return None

            storage_context = StorageContext.from_defaults(persist_dir=LOCAL_INDEX_DIR)
            index = load_index_from_storage(storage_context)
            st.success("インデックスのロードが完了しました。質問を入力できます。")
            return index
        except Exception as e:
            st.error(f"GCSからのインデックスロード中にエラーが発生しました: {e}")
            st.exception(e)
            return None
        finally:
            if temp_gcs_key_path and os.path.exists(temp_gcs_key_path):
                os.remove(temp_gcs_key_path)

def get_response_from_llm(index: VectorStoreIndex, query: str, n_value: int, custom_qa_template_str: str):
    """
    LLMを使用して、LlamaIndexのインデックスから回答を生成します。
    """
    try:
        llm = GoogleGenAI(model="gemini-1.5-flash-latest") 
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
        return None

# --- 3. Streamlit UIの構築 ---
def main():
    """
    StreamlitアプリケーションのメインUIを構築します。
    """
    st.title("🤖 プロンプトテスト")
    st.markdown("""
    ## 初回読み込みに1分ほどかかります
    このアプリは、フランケンラジオAIのプロンプトテストが出来ます。
    左のサイドバーの設定で質問応答の挙動を調整できます。
    1.  **類似ドキュメント検索数**: 回答の基になる情報の量を調整します。
    2.  **カスタムQAプロンプト**: AIへの指示を細かく変更できます。
    3.  **質問入力**: 設定後、質問を入力してEnterキーを押してください。
    """)
    st.markdown("---")

    llama_index = load_llama_index_from_gcs()

    if llama_index:
        st.sidebar.header("⚙️ 高度な設定")
        n_value = st.sidebar.slider(
            "類似ドキュメント検索数 (N値)", 1, 10, 3, 1,
            help="回答生成の際に参照する、関連性の高いドキュメントの数を指定します。約3分の内容がドキュメント1つに相当します"
        )
        st.sidebar.info(f"現在、上位 **{n_value}** 個の関連内容を使用して回答を生成します。")

        st.sidebar.subheader("📝 カスタムQAプロンプト")
        custom_prompt_text = st.sidebar.text_area(
            "プロンプトを編集してください:",
            DEFAULT_QA_PROMPT, height=350,
            help="AIへの指示です。`{context_str}` (参照情報) と `{query_str}` (ユーザーの質問) は必ず含めてください。"
        )
        st.sidebar.markdown("---")
        st.sidebar.caption("© 2024 RAG Demo")

        st.header("💬 質問を入力してください")
        user_query = st.text_input(
            "フランケンAIに聞きたいことをここに入力してください:",
            placeholder="例: 今後のキャリアはどうしたらいいでしょうか？",
            key="user_query_input"
        )

        if user_query and user_query != st.session_state.last_query:
            st.session_state.feedback_submitted = False
            st.session_state.last_query = user_query

        if user_query:
            if "{context_str}" not in custom_prompt_text or "{query_str}" not in custom_prompt_text:
                st.warning("プロンプトには`{context_str}`と`{query_str}`の両方を含めてください。")
            else:
                logger.info("="*50)
                logger.info("新しいクエリの処理を開始します。")
                logger.info(f"[入力文] {user_query}")
                logger.info(f"[チャンク選択数] {n_value}")

                response = get_response_from_llm(llama_index, user_query, n_value, custom_prompt_text)

                if response:
                    st.session_state.last_response = str(response)

                    st.subheader("🤖 AIからの回答")
                    st.write(str(response))
                    
                    logger.info(f"[LLMからの回答] {str(response)}")

                    if response.source_nodes:
                        logger.info("--- 選択されたチャンク（回答根拠） ---")
                        with st.expander("参照されたソースを確認"):
                            for i, node in enumerate(response.source_nodes):
                                source_text = f"ソース {i+1} (関連度: {node.score:.4f})"
                                st.markdown(f"--- **{source_text}** ---")
                                st.text_area(
                                    label=f"ソース {i+1} の内容",
                                    value=node.text,
                                    height=150,
                                    disabled=True,
                                    key=f"chunk_{i}"
                                )
                                logger.info(f"[{source_text}] {node.text.replace('\n', ' ')}")
                        logger.info("--- チャンクのログ記録終了 ---")
                    else:
                        logger.warning("参照されたソースノードが見つかりませんでした。")

                    st.markdown("---")
                    st.subheader("📝 この回答についての感想")

                    if st.session_state.feedback_submitted:
                        st.success("フィードバックを記録しました。ありがとうございます！")
                    else:
                        with st.form(key='feedback_form'):
                            st.write("各項目について5段階で評価してください。")
                            
                            busso_doai = st.slider("1. 物騒度合い (1: 穏やか 〜 5: 過激)", 1, 5, 3)
                            datousei = st.slider("2. 質問への返答の妥当性 (1: 不適切 〜 5: 完璧)", 1, 5, 3)
                            igaisei = st.slider("3. 意外性 (1: 予測通り 〜 5: 驚き)", 1, 5, 3)
                            humor = st.slider("4. ユーモア (1: 皆無 〜 5: 爆笑)", 1, 5, 3)

                            feedback_comment = st.text_area(
                                "その他、コメントがあればご記入ください:",
                                placeholder="例：回答が的確だった、もっと具体的にしてほしかったなど"
                            )
                            submit_button = st.form_submit_button(label='フィードバックを送信')

                            if submit_button:
                                logger.info("--- ユーザーからの感想 ---")
                                logger.info(f"[対象の質問] {st.session_state.last_query}")
                                logger.info(f"[対象の回答] {st.session_state.last_response}")
                                logger.info(f"[評価 - 物騒度合い] {busso_doai}")
                                logger.info(f"[評価 - 妥当性] {datousei}")
                                logger.info(f"[評価 - 意外性] {igaisei}")
                                logger.info(f"[評価 - ユーモア] {humor}")
                                logger.info(f"[コメント] {feedback_comment.replace('\n', ' ')}")
                                logger.info("--- 感想のログ記録終了 ---")
                                
                                st.session_state.feedback_submitted = True
                                st.rerun()
                else:
                    logger.error("LLMからの応答がありませんでした。")

    else:
        st.error(
            "インデックスのロードに失敗したため、QAシステムを起動できませんでした。"
            "ページ上部のエラーメッセージを確認し、設定やGCSの状態を見直してください。"
        )

if __name__ == "__main__":
    main()
