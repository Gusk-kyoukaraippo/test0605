import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
# from llama_index.embeddings.google_genai import GoogleGenerativeAIEmbedding # 現在のLlamaIndexではSettings.embed_modelで明示的に設定しなくても良い場合が多いです

# --- Gemini APIキーの設定 ---
# Streamlit SecretsからAPIキーを取得することを推奨
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Gemini APIキーが設定されていません。'.streamlit/secrets.toml' ファイルに GOOGLE_API_KEY を設定してください。")
    st.stop() # APIキーがない場合はアプリの実行を停止

# インデックスが保存されているディレクトリ
INDEX_DIR = "storage"

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

# インデックスのロード (キャッシュして高速化)
# @st.cache_resource デコレータは、関数が初めて実行されたときに結果をキャッシュし、
# 次回以降の実行ではキャッシュされた結果を再利用することで、ロード時間を短縮します。
@st.cache_resource
def load_llama_index(index_dir: str):
    """
    保存されたインデックスをロードします。インデックスが存在しない場合はエラーメッセージを表示します。
    """
    index_path = os.path.join(index_dir, "docstore.json")
    if not os.path.exists(index_path):
        st.error(f"エラー: '{index_dir}' にインデックスが見つかりませんでした。")
        st.info("インデックスを作成するには、データディレクトリにドキュメントを配置し、元のコードを一度実行してインデックスを生成してください。")
        return None
    
    st.spinner(f"'{index_dir}' からインデックスをロード中です...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        st.success("インデックスが正常にロードされました。")
        return index
    except Exception as e:
        st.error(f"インデックスのロード中にエラーが発生しました: {e}")
        st.info("インデックスファイルが破損している可能性があります。再生成を試みてください。")
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

    with st.spinner('AIが回答を生成中です...'):
        response = query_engine.query(query)
    return response

# --- Streamlit UIの構築 ---
st.set_page_config(page_title="RAGベースQAウェブアプリ", layout="wide")
st.title("📚 ドキュメントQAボット")

st.markdown("""
このアプリは、既存のドキュメントインデックスに基づいて質問に回答します。
左側のサイドバーでプロンプトや関連度 (`N` 値) を調整して、AIの応答を試すことができます。
---
""")

# インデックスのロード
llama_index = load_llama_index(INDEX_DIR)

if llama_index:
    # サイドバーでの設定
    st.sidebar.header("⚙️ 設定")

    # N値の調整スライダー
    n_value = st.sidebar.slider(
        "類似度トップK (N値)", 
        min_value=1, 
        max_value=20, 
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