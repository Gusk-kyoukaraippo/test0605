import os
import shutil
import streamlit as st
import json
import tempfile
import numpy as np # この行は使用されていないようですが、元のコードにあったので残します。
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    Settings,
)
from llama_index.llms.google_genai import GoogleGenAI
# OpenAIEmbedding をインポートするために、GoogleGenAIEmbedding のインポートをコメントアウトまたは削除し、OpenAIEmbedding を追加します。
# from llama_index.embeddings.google_genai import GoogleGenAIEmbedding 
from llama_index.embeddings.openai import OpenAIEmbedding # ★ここをOpenAIEmbeddingに変更

from google.cloud import storage

# --- ページ設定 (最初に一度だけ呼び出す) ---
st.set_page_config(page_title="フランケンAIプロンプトテスト", layout="wide")

# --- ロギング設定 ---
LOG_FILE = "app.log"
# 日本語を含むログメッセージを正しくファイルに出力するために encoding='utf-8' を指定
# Streamlitの再実行時にハンドラーが重複して追加されるのを防ぐため、ロガーの既存ハンドラーをクリア
logger = logging.getLogger(__name__)
if not logger.handlers: # ハンドラーがまだ設定されていない場合のみ設定
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # ファイルハンドラー
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # ストリームハンドラー (コンソール出力)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)


# --- 定数定義 ---
LOCAL_INDEX_DIR = "downloaded_storage_openai_embed" # インデックス保存ディレクトリ名を変更し、区別します
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

# --- 1. APIキーとGCS認証情報の設定 ---
temp_gcs_key_path = None
try:
    # Streamlit secretsから設定を読み込み
    # GOOGLE_API_KEYが存在するか確認 (Gemini LLM用)
    if "GOOGLE_API_KEY" not in st.secrets:
        raise KeyError("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    # ★OpenAI APIキーの追加
    if "OPENAI_API_KEY" not in st.secrets:
        raise KeyError("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


    # GCS関連のシークレットが存在するか確認
    if "GCS_BUCKET_NAME" not in st.secrets:
        raise KeyError("GCS_BUCKET_NAME")
    GCS_BUCKET_NAME = st.secrets["GCS_BUCKET_NAME"]

    if "GCS_INDEX_PREFIX" not in st.secrets:
        raise KeyError("GCS_INDEX_PREFIX")
    GCS_INDEX_PREFIX = st.secrets["GCS_INDEX_PREFIX"]

    if "GCS_SERVICE_ACCOUNT_JSON" not in st.secrets:
        raise KeyError("GCS_SERVICE_ACCOUNT_JSON")
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
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name # temp_gcs_key_path は不要になったため直接代入
    temp_gcs_key_path = tmp_file.name # 後でファイルを削除するためにパスを保持

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
    インデックスはローカルディレクトリにキャッシュされます。
    """
    if os.path.exists(LOCAL_INDEX_DIR):
        shutil.rmtree(LOCAL_INDEX_DIR)
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

    # 初回読み込み時間の提示
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

            # ★埋め込みモデルを設定をOpenAI Embeddingに変更
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", dimensions=3072)
            st.info(f"埋め込みモデル: **{Settings.embed_model.model_name}** (次元: **{Settings.embed_model.dimensions}**) を使用します。")


            # 埋め込みモデルが機能するかテスト
            try:
                test_embedding = Settings.embed_model.get_text_embedding("これはテスト文字列です。")
                if not isinstance(test_embedding, list) or len(test_embedding) == 0:
                    st.error("埋め込みモデルが有効な埋め込みを生成できませんでした。APIキーとモデルへのアクセスを確認してください。")
                    return None

                expected_dimension = 3072 # ★期待される次元数を3072に変更
                if len(test_embedding) != expected_dimension:
                    st.error(
                        f"埋め込みモデルが期待される {expected_dimension} 次元ではなく、"
                        f"{len(test_embedding)} 次元を返しました。モデル設定またはAPIの制限を確認してください。"
                    )
                    st.info(
                        "`text-embedding-3-large` モデルが実際に3072次元の出力をサポートしているか、"
                        "またはその次元で利用可能か確認してください。"
                    )
                    return None
                st.success("埋め込みモデルが正常に動作することを確認しました。")
            except Exception as e:
                st.error(f"埋め込みモデルの初期テスト中にエラーが発生しました: {e}")
                st.info("APIキー ('OPENAI_API_KEY') が正しく設定されているか、OpenAI Embedding APIへのアクセスが許可されているか確認してください。")
                return None

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
        finally:
            # 一時ファイルを削除 (念のため、Streamlitアプリの実行が終了するまで残る可能性があります)
            if temp_gcs_key_path and os.path.exists(temp_gcs_key_path):
                os.remove(temp_gcs_key_path)

def get_response_from_llm(index: VectorStoreIndex, query: str, n_value: int, custom_qa_template_str: str):
    """
    LLMを使用して、LlamaIndexのインデックスから回答を生成します。
    """
    try:
        # LLMはGeminiのまま変更なし
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
        st.info("Gemini APIキーが有効か、選択したLLMモデルが利用可能か確認してください。また、LlamaIndexが適切に埋め込みを生成できているか確認してください。")
        return None

# --- 3. Streamlit UIの構築 ---
def main():
    """
    StreamlitアプリケーションのメインUIを構築します。
    """
    st.title("🤖 プロンプトテスト")
    st.markdown("""
                ##初回読み込みに3分ほどかかります\n
    このアプリは、フランケンラジオAIのプロンプトテストが出来ます。\n
    notebookLMと違い、左のサイドバーの設定で質問応答の挙動を調整できます。\n
    設定項目は3つあります。\n
    1:左上段は、参照情報を何か所入れ込むか設定できます。(1カ所でラジオ約3分)\n
    2:左した段ではプロンプトを設定できます。\n
    3:左の設定後、ユーザーの質問に記入してエンターを押してください
    """)
    st.markdown("---")

    llama_index = load_llama_index_from_gcs()

    if llama_index:
        st.sidebar.header("⚙️ 高度な設定")
        n_value = st.sidebar.slider(
            "類似ドキュメント検索数 (N値)", 1, 10, 3, 1,
            help="回答生成の際に参照する、関連性の高いドキュメントの数を指定します。約3分の内容がドキュメント1つに相当します"
        )
        st.sidebar.info(f"ドキュメント1つが約3分の内容に相当します。現在、上位 **{n_value}** 個の関連内容を使用して回答を生成します。")

        st.sidebar.subheader("📝 カスタムQAプロンプト")
        custom_prompt_text = st.sidebar.text_area(
            "プロンプトを編集してください:",
            DEFAULT_QA_PROMPT, height=350,
            help="AIへの指示です。**`{context_str}` (参照情報)** と **`{query_str}` (ユーザーの質問)** は必ず含めてください。これらが含まれていないと、AIは適切に回答を生成できません。"
        )
        st.sidebar.markdown("---")
        st.sidebar.caption("© 2024 RAG Demo")

        st.header("💬 質問を入力してください")
        user_query = st.text_input(
            "フランケンAIに聞きたいことをここに入力してください:",
            placeholder="例: 今後のキャリアはどうしたらいいでしょうか？",
            help="質問を入力してEnterキーを押すか、少し待つと回答が生成されます。"
        )

        if user_query:
            if "{context_str}" not in custom_prompt_text or "{query_str}" not in custom_prompt_text:
                st.warning("プロンプトには`{context_str}`と`{query_str}`の両方を含めてください。これらはAIが参照情報と質問を認識するために必須です。")
            else:
                response = get_response_from_llm(llama_index, user_query, n_value, custom_prompt_text)
                if response:
                    st.subheader("🤖 AIからの回答")
                    st.write(str(response))
                    
                    # 参照ソースの表示
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
            "ページ上部のエラーメッセージを確認し、**設定やGCSの状態を見直してください**。特に、`secrets.toml`のキーと値が正しいか再確認してください。"
        )

if __name__ == "__main__":
    main()
