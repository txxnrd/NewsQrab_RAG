import os
# transformers 가 torchvision 을 import 하지 않도록 설정 (MPS/arm 오류 방지)
os.environ["DISABLE_TORCHVISION_IMPORTS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (필수 import)
from sklearn.decomposition import PCA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import matplotlib.font_manager as fm

VECTORSTORE_PATH = "faiss_index"


embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

from collections import Counter
type_counter = Counter(
    doc.metadata.get("type", "unknown")
    for doc in vectorstore.docstore._dict.values()
)
print(type_counter)
