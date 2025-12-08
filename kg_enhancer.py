import spacy

class KGEnhancer:
    def __init__(self):
        print("正在加载 Spacy 和 Entity Linker (首次加载可能需要几秒)...")
        # 禁用不需要的组件以提升速度
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
        self.nlp.add_pipe("entityLinker", last=True)

    def enhance_text(self, text):
        """
        功能：将实体描述拼接到原始文本后面
        """
        try:
            doc = self.nlp(text)
        except Exception:
            return text

        descriptions = []
        seen_entities = set()

        for entity in doc._.linkedEntities:
            # 过滤策略：
            # 1. 置信度 < 0.5 丢弃
            # 2. 没有描述的丢弃
            print(entity)

            desc = entity.get_description()
            if not desc:
                continue

            label = entity.get_label()
            # 3. 去重
            if label in seen_entities:
                continue

            seen_entities.add(label)
            descriptions.append(f"{label}: {desc}")

        if descriptions:
            # 使用 [SEP] 分隔符连接原文和知识
            enhanced_part = " ; ".join(descriptions)
            return f"{text} [SEP] {enhanced_part}"
        else:
            return text