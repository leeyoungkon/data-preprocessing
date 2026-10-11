from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
인공지능은 인간의 지능적인 행동을 컴퓨터가 수행하도록 하는 기술이다.

머신러닝은 데이터를 통해 패턴을 학습한다.

딥러닝은 인공신경망을 이용하여 복잡한 패턴을 학습한다.

생성형 AI는 새로운 텍스트, 이미지, 음악 등을 생성할 수 있다.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}")
    print(chunk)