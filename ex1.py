text = """
인공지능은 인간의 지능적인 행동을 컴퓨터가 수행하도록 하는 기술이다.
머신러닝은 데이터를 통해 패턴을 학습한다.
딥러닝은 인공신경망을 이용하여 복잡한 패턴을 학습한다.
생성형 AI는 학습된 데이터를 기반으로 새로운 콘텐츠를 생성한다.
"""

chunk_size = 50

chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

for chunk in chunks:
    print(chunk)
    print("------")
