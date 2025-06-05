import os
import shutil
import streamlit as st
import json # jsonモジュールをインポート
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.gemini import GeminiEmbedding
from google.cloud import storage
import tempfile

    #一番最初にst.set_page_config(page_title="RAGベースQAウェブアプリ (GCS対応)", layout="wide")
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
temp_gcs_key_path = None # finallyブロックでもアクセスできるように初期化

try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    GCS_BUCKET_NAME = st.secrets["GCS_BUCKET_NAME"]
    GCS_INDEX_PREFIX = st.secrets["GCS_INDEX_PREFIX"]

    gcs_service_account_json_str = st.secrets["GCS_SERVICE_ACCOUNT_JSON"]

    st.header("🔑 GCS 認証情報デバッグ")
    st.write("Streamlit secretsから読み込んだ生文字列:")
    st.code(gcs_service_account_json_str) # 生文字列を表示
    st.write(f"文字列の長さ: {len(gcs_service_account_json_str)}")

    # エラーメッセージの char 176 付近を特に確認
    # (例: "private_key"の行あたり)
    error_char_index = 175 # char 176 は0-indexedで175
    context_length = 50 # 前後50文字を表示
    start_index = max(0, error_char_index - context_length)
    end_index = min(len(gcs_service_account_json_str), error_char_index + context_length)

    st.write(f"エラー位置 (char 176) 付近の文字列（エスケープ表示）:")
    st.code(repr(gcs_service_account_json_str[start_index:end_index]))
    st.write(f"完全な文字列のエスケープ表示（デバッグ用、長いです）:")
    st.text_area("Full Raw String (repr)", repr(gcs_service_account_json_str), height=300)


    try:
        parsed_json_data = json.loads(gcs_service_account_json_str)
        clean_gcs_service_account_json_str = json.dumps(parsed_json_data, indent=2, ensure_ascii=False)
        st.success("GCS_SERVICE_ACCOUNT_JSONのパースと整形に成功しました。")

    except json.JSONDecodeError as e:
        st.error(f"Streamlit secretsの'GCS_SERVICE_ACCOUNT_JSON'が不正なJSON形式です: {e}")
        st.info(
            "secrets.tomlに記載されているGCPサービスアカウントJSON文字列に、"
            "余分な文字（改行、スペース、不正なエスケープシーケンスなど）が含まれていないか、"
            "あるいは正しくJSON形式として記述されているか確認してください。"
            "特に、`key = \"\"\"...\"\"\"`のように3つの引用符で囲んで複数行文字列として定義してください。"
            "**上記デバッグ表示の 'エラー位置付近の文字列' に、予期しない文字がないか確認してください。**"
        )
        st.stop() # アプリケーションを停止
    except Exception as e:
        st.error(f"サービスアカウントJSONの処理中に予期せぬエラーが発生しました: {e}")
        st.exception(e)
        st.stop()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8") as tmp_file:
        tmp_file.write(clean_gcs_service_account_json_str)
        temp_gcs_key_path = tmp_file.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_gcs_key_path

except KeyError as e:
    st.error(
        f"必要な設定が secrets.toml に見つかりません: {e}。"
        "アプリケーションを正しく実行するには、Streamlitのsecretsに必要な情報を設定してください。"
        "例: GOOGLE_API_KEY, GCS_BUCKET_NAME, GCS_INDEX_PREFIX, GCS_SERVICE_ACCOUNT_JSON"
    )
    st.stop()
except Exception as e:
    st.error(f"設定の読み込み中に予期せぬエラーが発生しました: {e}")
    st.exception(e)
    st.stop()
finally:
    # 実際の本番環境では一時ファイルの削除を慎重に検討しますが、
    # デバッグ中は残しておいても問題ありません。
    # st.experimental_singleton などを使わない場合、セッションごとに実行されるため、
    # ここで削除すると次の実行時にファイルがない状態になる可能性があります。
    # 通常はOSによるクリーンアップに任せるか、アプリ終了時に明示的に削除するフックを用意します。
    pass


    # tempfileを使用して一時ファイルを作成し、クリーンなJSON文字列を書き込む
    # delete=False にすることで、ファイルが閉じられた後も自動で削除されず、
    # 環境変数 GOOGLE_APPLICATION_CREDENTIALS がそのパスを参照できるようにします。
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8") as tmp_file:
        tmp_file.write(clean_gcs_service_account_json_str)
        temp_gcs_key_path = tmp_file.name

    # 環境変数に一時ファイルのパスを設定
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_gcs_key_path

    # 注意: temp_gcs_key_path で作成された一時ファイルは、delete=False のため自動削除されません。
    # Streamlitの実行モデルでは、アプリケーションのプロセスが終了するときにOSによってクリーンアップされることが期待されますが、
    # 必要に応じて、アプリケーションのシャットダウンフックなどで明示的に削除することも検討できます。


# --- 2. LlamaIndex関連の関数 ---
@st.cache_resource
def load_llama_index_from_gcs():
    """
    Google Cloud Storage (GCS) からLlamaIndexのインデックスをダウンロードし、ロードします。
    インデックスはローカルディレクトリにキャッシュされます。
    """
    # 既存のローカルインデックスディレクトリをクリア
    if os.path.exists(LOCAL_INDEX_DIR):
        st.write(f"既存のローカルインデックスディレクトリ '{LOCAL_INDEX_DIR}' をクリアします...")
        shutil.rmtree(LOCAL_INDEX_DIR)
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

    st.info(f"GCSバケット '{GCS_BUCKET_NAME}' からインデックスファイルをダウンロード中... (プレフィックス: '{GCS_INDEX_PREFIX}')")

    try:
        # GCSクライアントを初期化（環境変数 GOOGLE_APPLICATION_CREDENTIALS を参照）
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        # 指定されたプレフィックスに一致するすべてのブロブ（ファイル）をリストアップ
        blobs = list(bucket.list_blobs(prefix=GCS_INDEX_PREFIX))

        # インデックスファイルが見つからない場合の警告
        if not blobs or all(blob.name == GCS_INDEX_PREFIX and blob.size == 0 for blob in blobs):
            st.warning(
                f"GCSバケット '{GCS_BUCKET_NAME}' の '{GCS_INDEX_PREFIX}' パスにインデックスファイルが見つかりませんでした。"
                "インデックスがアップロードされているか、パスが正しいか確認してください。"
            )
            return None

        download_count = 0
        # 各ブロブをローカルディレクトリにダウンロード
        for blob in blobs:
            # プレフィックス自体やディレクトリを示すブロブはスキップ
            if blob.name == GCS_INDEX_PREFIX or blob.name.endswith('/'):
                continue
            # ローカルでの保存パスを構築
            relative_path = os.path.relpath(blob.name, GCS_INDEX_PREFIX)
            local_file_path = os.path.join(LOCAL_INDEX_DIR, relative_path)
            # サブディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            # ファイルをダウンロード
            blob.download_to_filename(local_file_path)
            download_count += 1

        # ダウンロードされたファイルがない場合の警告
        if download_count == 0:
            st.warning(
                f"GCSバケット '{GCS_BUCKET_NAME}' の '{GCS_INDEX_PREFIX}' パスにダウンロード可能なファイルが見つかりませんでした。"
            )
            return None

        st.success(f"{download_count} 個のインデックスファイルがGCSから正常にダウンロードされました。")
        # ダウンロードしたローカルインデックスをロード
        storage_context = StorageContext.from_defaults(persist_dir=LOCAL_INDEX_DIR)
        index = load_index_from_storage(storage_context)
        st.success("LlamaIndexがローカルのインデックスを正常にロードしました。")
        return index
    except Exception as e:
        # GCSからのロード中にエラーが発生した場合
        st.error(f"GCSからのインデックスロード中にエラーが発生しました: {e}")
        st.exception(e) # 詳細なトレースバックを表示
        st.info(
            "以下の点を確認してください:\n"
            f"- GCSバケット名 ('{GCS_BUCKET_NAME}') とプレフィックス ('{GCS_INDEX_PREFIX}') が正しいか。\n"
            "- Streamlit Secretsの 'GCS_SERVICE_ACCOUNT_JSON' が正しく設定され、適切な権限を持っているか。\n"
            "- インターネット接続が安定しているか。"
        )
        return None

def get_response_from_llm(index: VectorStoreIndex, query: str, n_value: int, custom_qa_template_str: str):
    """
    LLM (gemini-2.5-flash-preview-05-20) を使用して、LlamaIndexのインデックスから回答を生成します。
    """
    try:
        # GoogleGenAI LLMを初期化
        llm = GoogleGenAI(model="gemini-2.5-flash-preview-05-20")
        # カスタムQAプロンプトを適用
        qa_template = PromptTemplate(custom_qa_template_str)
        # クエリエンジンを設定
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=n_value, # 類似ドキュメント検索数
            text_qa_template=qa_template
        )
        st.info(f"クエリを実行中: '{query}' (類似検索数: {n_value})")
        # 回答生成中のスピナー表示
        with st.spinner('AIが回答を生成中です...しばらくお待ちください。'):
            response = query_engine.query(query)
        return response
    except Exception as e:
        # LLMからの応答取得中にエラーが発生した場合
        st.error(f"LLMからの応答取得中にエラーが発生しました: {e}")
        st.exception(e) # 詳細なトレースバックを表示
        st.info(
            "Gemini APIキーが有効であるか、または選択したモデルが利用可能か確認してください。"
            "問題が解決しない場合は、プロンプトの内容やN値を見直してみてください。"
        )
        return None

# --- 3. Streamlit UIの構築 ---
def main():
    """
    StreamlitアプリケーションのメインUIを構築します。
    """
    #st.set_page_config(page_title="RAGベースQAウェブアプリ (GCS対応)", layout="wide")
    st.title("📚 ドキュメントQAボット (GCS連携)")
    st.markdown("""
    このアプリケーションは、Google Cloud Storage (GCS) に保存されたドキュメントのインデックスを利用して、
    アップロードされたドキュメントの内容に関する質問に回答します。
    左側のサイドバーで、検索するドキュメントの数 (`N値`) やAIへの指示（プロンプト）を調整できます。
    """)
    st.markdown("---")

    # GCSからLlamaIndexをロード
    llama_index = load_llama_index_from_gcs()

    if llama_index:
        # インデックスが正常にロードされた場合、UIを表示
        st.sidebar.header("⚙️ 高度な設定")
        n_value = st.sidebar.slider(
            "類似ドキュメント検索数 (N値)", 1, 10, 3, 1,
            help="回答を生成する際に考慮する、最も関連性の高いドキュメントチャンクの数を指定します。"
        )
        st.sidebar.info(f"現在、上位 **{n_value}** 個の関連チャンクを使用して回答を生成します。")
        st.sidebar.subheader("📝 カスタムQAプロンプト")
        custom_prompt_text = st.sidebar.text_area(
            "プロンプトを編集 (プレースホルダー {context_str} と {query_str} は必須):",
            DEFAULT_QA_PROMPT, height=350,
            help="AIへの指示プロンプトです。`{context_str}` (参照情報) と `{query_str}` (質問) を含めてください。"
        )
        st.sidebar.markdown("---")
        st.sidebar.caption("© 2024 RAG Demo with LlamaIndex & Streamlit")

        st.header("💬 質問を入力してください")
        user_query = st.text_input(
            "ドキュメントに関する質問をここに入力してください:",
            placeholder="例: このドキュメントの主要なテーマは何ですか？"
        )

        if user_query:
            # プロンプトに必須のプレースホルダーが含まれているかチェック
            if "{context_str}" not in custom_prompt_text or "{query_str}" not in custom_prompt_text:
                st.warning(
                    "カスタムプロンプトには `{context_str}` と `{query_str}` の両方のプレースホルダーを含める必要があります。"
                    "サイドバーでプロンプトを確認・修正してください。"
                )
            else:
                st.info(f"送信された質問: **{user_query}**")
                # LLMから応答を取得
                response = get_response_from_llm(llama_index, user_query, n_value, custom_prompt_text)
                if response:
                    st.subheader("🤖 AIからの回答")
                    st.write(str(response))
                    # デバッグ情報と参照ドキュメントのチャンクを表示
                    with st.expander("詳細情報を見る (デバッグ用)"):
                        st.write("**使用されたプロンプト:**")
                        st.text(custom_prompt_text.format(context_str="<参照情報>", query_str=user_query))
                        st.write(f"**使用されたN値:** {n_value}")
                        if hasattr(response, 'source_nodes') and response.source_nodes:
                            st.write("**参照されたドキュメントのチャンク:**")
                            for i, node in enumerate(response.source_nodes):
                                st.markdown(f"--- チャンク {i+1} (類似度スコア: {node.score:.4f}) ---")
                                st.text_area(f"チャンク {i+1} 内容", node.text, height=150, disabled=True, key=f"chunk_text_{i}")
                                if node.metadata:
                                    st.write(f"**メタデータ (チャンク {i+1}):**")
                                    st.json(node.metadata)
                        else:
                            st.info("この応答では、参照されたソースドキュメントの情報は利用できません。")
    else:
        # インデックスのロードに失敗した場合のエラーメッセージ
        st.error(
            "インデックスのロードに失敗したため、QAシステムを起動できませんでした。"
            "ページ上部のエラーメッセージやログを確認し、設定やGCSの状態を見直してください。"
        )

if __name__ == "__main__":
    main()
